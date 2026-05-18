"""Typer CLI: launch evolution runs from the terminal.

Usage:
    science-game run matmul --provider ollama-qwen --generations 5
    science-game list-benchmarks
"""

from __future__ import annotations

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
    benchmark: str = typer.Argument(..., help="Benchmark name (e.g. matmul)"),
    provider: str = typer.Option("ollama-qwen", "--provider", "-p"),
    model: str | None = typer.Option(None, "--model", "-m"),
    generations: int = typer.Option(5, "--generations", "-g"),
    temperature: float = typer.Option(0.8, "--temperature", "-t"),
    seed: int = typer.Option(42, "--seed", "-s"),
    runs_root: Path = typer.Option(Path("runs"), "--runs-root"),
) -> None:
    """Launch one evolution run, writing artifacts to runs/<run_id>/."""
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
    write_manifest(manifest, run_dir / "manifest.json")

    console.print(f"[bold green]Run[/] {run_id} → {run_dir}")
    console.print(f"[dim]benchmark={benchmark} provider={provider} gens={generations} seed={seed}[/]")

    best = run_evolution(config)

    console.print(
        f"\n[bold]Best fitness:[/] {best.result.fitness:.6f} "
        f"(gen {best.generation}, correct={best.result.correct})"
    )
    if best.result.metrics:
        console.print(f"[dim]metrics:[/] {best.result.metrics}")


if __name__ == "__main__":
    app()
