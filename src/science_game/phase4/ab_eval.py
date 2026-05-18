"""A/B evaluation: run the same benchmark + seed pair twice — once with the
base mutator, once with the fine-tuned one — and report whether the
fine-tuned model is winning.

This is the post-fine-tune validation step in the Phase-4 playbook:
without it we can't claim the loop actually self-improved.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from science_game.engine import EvolutionConfig, run_evolution


@dataclass
class AbPairResult:
    seed: int
    base_fitness: float
    finetuned_fitness: float
    base_correct: bool
    finetuned_correct: bool
    winner: str  # "base" | "finetuned" | "tie"


@dataclass
class AbReport:
    benchmark: str
    base_provider: str
    base_model: str | None
    finetuned_provider: str
    finetuned_model: str | None
    generations: int
    seeds: list[int]
    pairs: list[AbPairResult]

    @property
    def wins_finetuned(self) -> int:
        return sum(1 for p in self.pairs if p.winner == "finetuned")

    @property
    def wins_base(self) -> int:
        return sum(1 for p in self.pairs if p.winner == "base")

    @property
    def ties(self) -> int:
        return sum(1 for p in self.pairs if p.winner == "tie")

    @property
    def avg_delta(self) -> float:
        if not self.pairs:
            return 0.0
        return sum(p.finetuned_fitness - p.base_fitness for p in self.pairs) / len(self.pairs)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["summary"] = {
            "wins_finetuned": self.wins_finetuned,
            "wins_base": self.wins_base,
            "ties": self.ties,
            "avg_delta": self.avg_delta,
        }
        return d


def _decide_winner(base_fit: float, ft_fit: float, eps: float = 1e-9) -> str:
    if abs(base_fit - ft_fit) <= eps:
        return "tie"
    return "finetuned" if ft_fit > base_fit else "base"


def run_ab(
    benchmark: str,
    base_provider: str,
    finetuned_provider: str,
    base_model: str | None = None,
    finetuned_model: str | None = None,
    seeds: list[int] | None = None,
    generations: int = 20,
    runs_root: Path = Path("runs/ab"),
    temperature: float = 0.8,
) -> AbReport:
    """Run N seeded pairs sequentially. Returns an AbReport."""
    seeds = seeds if seeds is not None else [0, 1, 2, 3, 4]
    runs_root = Path(runs_root)
    pairs: list[AbPairResult] = []

    for seed in seeds:
        base_run = runs_root / f"seed{seed}-base"
        ft_run = runs_root / f"seed{seed}-finetuned"

        base = run_evolution(EvolutionConfig(
            benchmark=benchmark, llm_provider=base_provider, llm_model=base_model,
            generations=generations, temperature=temperature,
            seed=seed, run_dir=base_run,
        ))
        ft = run_evolution(EvolutionConfig(
            benchmark=benchmark, llm_provider=finetuned_provider, llm_model=finetuned_model,
            generations=generations, temperature=temperature,
            seed=seed, run_dir=ft_run,
        ))
        pairs.append(AbPairResult(
            seed=seed,
            base_fitness=base.result.fitness,
            finetuned_fitness=ft.result.fitness,
            base_correct=base.result.correct,
            finetuned_correct=ft.result.correct,
            winner=_decide_winner(base.result.fitness, ft.result.fitness),
        ))

    return AbReport(
        benchmark=benchmark,
        base_provider=base_provider, base_model=base_model,
        finetuned_provider=finetuned_provider, finetuned_model=finetuned_model,
        generations=generations, seeds=seeds, pairs=pairs,
    )


def write_report(report: AbReport, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2, default=str), encoding="utf-8")
    return path
