"""Adapter that lets a Science Game `Benchmark` run inside OpenEvolve.

OpenEvolve is the open-source reimplementation of AlphaEvolve. It uses MAP-
Elites + island-based evolution + diff-based mutation — substantially more
sophisticated than our standalone hill-climber. This adapter bridges the
two so you can flip between engines from the CLI:

    science-game run matmul --engine standalone   # MVP loop (default)
    science-game run matmul --engine openevolve   # MAP-Elites + islands

What this adapter does:
- Wraps `benchmark.seed_program` with `# EVOLVE-BLOCK-START / END` markers.
- Builds an `openevolve.Config` whose LLM section points at our chosen
  provider (Ollama via OpenAI-compatible `http://localhost:11434/v1`,
  Anthropic, or OpenAI itself).
- Wraps `benchmark.evaluate(code)` as the OpenEvolve callable evaluator.
- Runs `openevolve.run_evolution(...)` and turns its result into our
  Individual + writes the standard manifest/best layout under `runs/<id>/`.

What this adapter does NOT do (yet):
- It doesn't produce our `mutations.jsonl` Phase-4 schema. OpenEvolve's
  program-database state is on disk in its own format; parsing it into
  parent/child pairs is a Sprint-5 task. Until then, Phase-4 datasets
  should be built from standalone-engine runs.

Optional dependency:
    uv sync --extra engine-openevolve
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from science_game.benchmarks import get_benchmark
from science_game.engine import EvolutionConfig, Individual

EVOLVE_START = "# EVOLVE-BLOCK-START"
EVOLVE_END = "# EVOLVE-BLOCK-END"


def wrap_seed_with_markers(code: str) -> str:
    """Prepend/append OpenEvolve's evolve-block markers around the seed code."""
    if EVOLVE_START in code:
        return code  # already wrapped
    return f"{EVOLVE_START}\n{code}\n{EVOLVE_END}\n"


def _make_openai_compatible_llm(config: EvolutionConfig) -> dict[str, Any]:
    """Map our provider names to OpenEvolve's LLM config payload.

    All paths use OpenEvolve's OpenAI-compatible client; Ollama exposes one
    natively, so we just point api_base at its port."""
    provider = config.llm_provider
    model = config.llm_model

    if provider in ("ollama", "ollama-qwen"):
        return {
            "name": model or "qwen2.5-coder:7b",
            "api_base": "http://localhost:11434/v1",
            "api_key": "ollama",  # any non-empty string
        }
    if provider == "openai":
        return {
            "name": model or "gpt-4o",
            "api_base": "https://api.openai.com/v1",
            "api_key": "${OPENAI_API_KEY}",
        }
    if provider == "anthropic":
        # Anthropic via OpenAI-compatible proxy isn't supported natively by
        # OpenEvolve. We document this and reject for now.
        raise ValueError(
            "anthropic provider is not yet supported with --engine openevolve. "
            "Use --engine standalone, or use an OpenAI-compatible proxy in front "
            "of Anthropic."
        )
    raise ValueError(f"unknown llm provider for openevolve adapter: {provider!r}")


def build_openevolve_config(
    config: EvolutionConfig,
    iterations: int | None = None,
) -> Any:
    """Build an openevolve.Config from our EvolutionConfig.

    Raises ImportError with installation guidance if openevolve isn't installed."""
    try:
        from openevolve.config import Config, LLMModelConfig
    except ImportError as e:
        raise ImportError(
            "openevolve is not installed. Run: uv sync --extra engine-openevolve"
        ) from e

    llm_payload = _make_openai_compatible_llm(config)
    cfg = Config()
    cfg.llm.api_base = llm_payload["api_base"]
    cfg.llm.api_key = llm_payload["api_key"]
    cfg.llm.temperature = config.temperature
    cfg.llm.models = [LLMModelConfig(name=llm_payload["name"], weight=1.0)]
    cfg.llm.evaluator_models = list(cfg.llm.models)
    if iterations is not None:
        cfg.max_iterations = iterations
    else:
        cfg.max_iterations = config.generations
    cfg.random_seed = config.seed
    return cfg


def make_evaluator(benchmark_name: str):
    """Return a callable suitable for openevolve.run_evolution(evaluator=...)."""
    bench = get_benchmark(benchmark_name)

    def evaluator(program_path: str) -> dict[str, Any]:
        code = Path(program_path).read_text()
        result = bench.evaluate(code)
        out: dict[str, Any] = {"score": float(result.fitness), "correct": bool(result.correct)}
        for k, v in (result.metrics or {}).items():
            if isinstance(v, (int, float, bool, str)):
                out[k] = v
        if result.error:
            out["error"] = result.error
        return out

    return evaluator


@dataclass
class OpenEvolveRunOutput:
    best_score: float
    best_code: str
    output_dir: str | None
    metrics: dict


def run_with_openevolve(
    config: EvolutionConfig,
    iterations: int | None = None,
) -> Individual:
    """Run an evolution through OpenEvolve, mirror artifacts to our run layout."""
    try:
        from openevolve import run_evolution
    except ImportError as e:
        raise ImportError(
            "openevolve is not installed. Run: uv sync --extra engine-openevolve"
        ) from e

    bench = get_benchmark(config.benchmark)
    seed_code = wrap_seed_with_markers(bench.seed_program)

    run_dir = Path(config.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    oe_output_dir = run_dir / "openevolve"

    oe_config = build_openevolve_config(config, iterations=iterations)
    result = run_evolution(
        initial_program=seed_code,
        evaluator=make_evaluator(config.benchmark),
        config=oe_config,
        iterations=iterations or config.generations,
        output_dir=str(oe_output_dir),
        cleanup=False,
    )

    # Mirror to our standard layout.
    best_dir = run_dir / "best"
    best_dir.mkdir(exist_ok=True)
    best_path = best_dir / f"openevolve_best_fit{result.best_score:.6f}.py"
    best_path.write_text(result.best_code)
    (best_dir / "latest.py").write_text(result.best_code)

    (run_dir / "openevolve_summary.json").write_text(
        json.dumps({
            "best_score": float(result.best_score),
            "metrics": result.metrics,
            "output_dir": str(oe_output_dir),
        }, indent=2, default=str)
    )

    # Re-evaluate against our benchmark so the returned Individual carries
    # correct/metrics in our schema.
    final_eval = bench.evaluate(result.best_code)
    return Individual(code=result.best_code, result=final_eval, generation=-1)
