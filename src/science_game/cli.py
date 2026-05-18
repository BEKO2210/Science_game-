"""Typer CLI: launch evolution runs from the terminal.

Usage:
    science-game run matmul --provider ollama-qwen --generations 5
    science-game run matmul --from-manifest runs/matmul-xxxx/manifest.json
    science-game list-benchmarks
    science-game build-mutator-dataset --runs-root runs --out dataset.jsonl
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from science_game.benchmarks import list_benchmarks
from science_game.engine import EvolutionConfig, run_evolution
from science_game.publish.manifest import Manifest, write_manifest

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()


@app.command("list-benchmarks")
def list_benchmarks_cmd() -> None:
    table = Table(title="Available benchmarks")
    table.add_column("name")
    for name in list_benchmarks():
        table.add_row(name)
    console.print(table)


@app.command("run")
def run_cmd(
    benchmark: str = typer.Argument(None, help="Benchmark name (e.g. matmul)"),
    provider: str = typer.Option("ollama-qwen", "--provider", "-p"),
    model: str | None = typer.Option(None, "--model", "-m"),
    generations: int = typer.Option(5, "--generations", "-g"),
    temperature: float = typer.Option(0.8, "--temperature", "-t"),
    seed: int = typer.Option(42, "--seed", "-s"),
    runs_root: Path = typer.Option(Path("runs"), "--runs-root"),
    from_manifest: Path | None = typer.Option(
        None, "--from-manifest",
        help="Reproduce a previous run from its manifest.json. All other flags are ignored.",
    ),
) -> None:
    """Launch one evolution run, writing artifacts to runs/<run_id>/."""
    if from_manifest is not None:
        cfg_dict = json.loads(from_manifest.read_text())["config"]
        cfg_dict["run_dir"] = Path(cfg_dict["run_dir"])
        # Re-run gets a fresh run_id so we don't overwrite the original.
        new_run_id = f"{cfg_dict['benchmark']}-rerun-{uuid.uuid4().hex[:8]}"
        cfg_dict["run_dir"] = runs_root / new_run_id
        config = EvolutionConfig(**cfg_dict)
        run_id = new_run_id
        console.print(f"[bold yellow]Re-run from manifest:[/] {from_manifest}")
    else:
        if benchmark is None:
            raise typer.BadParameter("benchmark is required unless --from-manifest is given")
        run_id = f"{benchmark}-{uuid.uuid4().hex[:8]}"
        run_dir = runs_root / run_id
        config = EvolutionConfig(
            benchmark=benchmark,
            llm_provider=provider,
            llm_model=model,
            generations=generations,
            temperature=temperature,
            seed=seed,
            run_dir=run_dir,
        )

    manifest = Manifest.create(run_id, config=asdict(config))
    write_manifest(manifest, config.run_dir / "manifest.json")

    console.print(f"[bold green]Run[/] {run_id} → {config.run_dir}")
    console.print(
        f"[dim]benchmark={config.benchmark} provider={config.llm_provider} "
        f"gens={config.generations} seed={config.seed}[/]"
    )

    best = run_evolution(config)

    console.print(
        f"\n[bold]Best fitness:[/] {best.result.fitness:.6f} "
        f"(gen {best.generation}, correct={best.result.correct})"
    )
    if best.result.metrics:
        console.print(f"[dim]metrics:[/] {best.result.metrics}")


@app.command("build-mutator-dataset")
def build_mutator_dataset_cmd(
    runs_root: Path = typer.Option(Path("runs"), "--runs-root"),
    out: Path = typer.Option(Path("mutator-dataset.jsonl"), "--out", "-o"),
    mode: str = typer.Option("sft", "--mode", help="sft | dpo"),
    min_delta: float = typer.Option(0.0, "--min-delta", help="Only accepted mutations with fitness_after - fitness_before > min_delta"),
) -> None:
    """Aggregate runs/*/mutations.jsonl into a fine-tuning dataset (Phase 4)."""
    from science_game.phase4.dataset import build_dataset

    n = build_dataset(runs_root, out, mode=mode, min_delta=min_delta)
    console.print(f"[bold green]Wrote[/] {n} examples → {out}")


if __name__ == "__main__":
    app()
