"""Standalone evolution loop for the MVP.

In Sprint 3 this will be replaced (or wrapped over) by the vendored OpenEvolve
controller, which adds program-database curation, diff-based mutation, and
island-model parallelism. For Sprint 1 we just need an end-to-end loop that
proves the LLM-Provider + Benchmark + Manifest stack works.

Loop:
    1. Seed the population with the benchmark's seed_program.
    2. Repeat for N generations:
        a. Pick the current best.
        b. Ask the LLM to mutate it.
        c. Evaluate the child.
        d. If fitness improved, accept; else keep parent.
        e. Append a row to mutations.jsonl (Phase-4-ready).
        f. Append a row to events.jsonl (dashboard-friendly).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from science_game.benchmarks import Benchmark, EvalResult, get_benchmark
from science_game.llm import LLMProvider, MutationRequest, get_provider


@dataclass
class EvolutionConfig:
    benchmark: str
    llm_provider: str = "ollama-qwen"
    llm_model: str | None = None
    generations: int = 5
    temperature: float = 0.8
    seed: int = 42
    run_dir: Path = field(default_factory=lambda: Path("runs/dev"))


@dataclass
class Individual:
    code: str
    result: EvalResult
    generation: int


class Engine:
    def __init__(self, config: EvolutionConfig, benchmark: Benchmark, provider: LLMProvider) -> None:
        self.config = config
        self.benchmark = benchmark
        self.provider = provider
        self.run_dir = Path(config.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.run_dir / "events.jsonl"
        self.mutations_path = self.run_dir / "mutations.jsonl"
        self.best_dir = self.run_dir / "best"
        self.best_dir.mkdir(exist_ok=True)

    def run(self) -> Individual:
        seed_code = self.benchmark.seed_program
        seed_result = self.benchmark.evaluate(seed_code)
        if not seed_result.correct and seed_result.error:
            # Seed must run, otherwise the benchmark is broken — fail loud.
            raise RuntimeError(
                f"benchmark {self.config.benchmark!r} seed program failed evaluation: "
                f"{seed_result.error}"
            )

        best = Individual(code=seed_code, result=seed_result, generation=0)
        self._log_event({
            "kind": "init",
            "generation": 0,
            "fitness": seed_result.fitness,
            "metrics": seed_result.metrics,
        })
        self._save_best(best)

        consecutive_provider_errors = 0
        max_consecutive_errors = 3

        for gen in range(1, self.config.generations + 1):
            t0 = time.time()
            request = MutationRequest(
                parent_code=best.code,
                task_description=self.benchmark.task_description,
                benchmark=self.config.benchmark,
                temperature=self.config.temperature,
                seed=self.config.seed + gen,
            )
            response = self.provider.mutate(request)
            llm_seconds = time.time() - t0

            if not response.child_code:
                err = response.meta.get("error", "empty response")
                self._log_event({
                    "kind": "llm_error",
                    "generation": gen,
                    "error": err,
                    "llm_seconds": llm_seconds,
                })
                consecutive_provider_errors += 1
                if consecutive_provider_errors >= max_consecutive_errors:
                    raise RuntimeError(
                        f"LLM provider {self.config.llm_provider!r} failed "
                        f"{consecutive_provider_errors} times in a row. "
                        f"Last error: {err}. "
                        f"Run `science-game doctor` to diagnose."
                    )
                continue
            consecutive_provider_errors = 0

            child_result = self.benchmark.evaluate(response.child_code)
            improved = child_result.fitness > best.result.fitness

            self._log_mutation(
                generation=gen,
                parent_code=best.code,
                child_code=response.child_code,
                fitness_before=best.result.fitness,
                fitness_after=child_result.fitness,
                accepted=improved,
                provider=response.provider,
                model=response.model,
                tokens_in=response.tokens_in,
                tokens_out=response.tokens_out,
            )

            self._log_event({
                "kind": "generation",
                "generation": gen,
                "fitness_parent": best.result.fitness,
                "fitness_child": child_result.fitness,
                "accepted": improved,
                "child_metrics": child_result.metrics,
                "child_error": child_result.error,
                "llm_seconds": llm_seconds,
                "tokens_in": response.tokens_in,
                "tokens_out": response.tokens_out,
            })

            if improved:
                best = Individual(code=response.child_code, result=child_result, generation=gen)
                self._save_best(best)

        return best

    def _log_event(self, event: dict) -> None:
        with self.events_path.open("a") as f:
            f.write(json.dumps(event, default=str) + "\n")

    def _log_mutation(self, **fields) -> None:
        with self.mutations_path.open("a") as f:
            f.write(json.dumps(fields, default=str) + "\n")

    def _save_best(self, ind: Individual) -> None:
        path = self.best_dir / f"gen{ind.generation:04d}_fit{ind.result.fitness:.6f}.py"
        path.write_text(ind.code)
        latest = self.best_dir / "latest.py"
        latest.write_text(ind.code)


def run_evolution(config: EvolutionConfig) -> Individual:
    benchmark = get_benchmark(config.benchmark)
    provider_kwargs = {}
    if config.llm_model:
        provider_kwargs["model"] = config.llm_model
    provider = get_provider(config.llm_provider, **provider_kwargs)
    engine = Engine(config, benchmark, provider)
    return engine.run()
