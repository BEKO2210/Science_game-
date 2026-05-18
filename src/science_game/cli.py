"""Typer CLI: launch evolution runs from the terminal.

Usage:
    science-game run matmul --provider ollama-qwen --generations 5
    science-game run matmul --from-manifest runs/matmul-xxxx/manifest.json
    science-game list-benchmarks
    science-game build-mutator-dataset --runs-root runs --out dataset.jsonl
    science-game phase4 prepare --dataset datasets/m-v1.jsonl --gguf-name forge-v1.gguf
    science-game phase4 evaluate --benchmark matmul --finetuned-model forge-mutator-v1
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
phase4_app = typer.Typer(add_completion=False, no_args_is_help=True, help="Phase-4 helpers.")
app.add_typer(phase4_app, name="phase4")
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
    engine: str = typer.Option(
        "standalone", "--engine",
        help="standalone (MVP hill-climber) | openevolve (MAP-Elites + islands)",
    ),
    from_manifest: Path | None = typer.Option(
        None, "--from-manifest",
        help="Reproduce a previous run from its manifest.json. All other flags are ignored.",
    ),
) -> None:
    """Launch one evolution run, writing artifacts to runs/<run_id>/."""
    if from_manifest is not None:
        cfg_dict = json.loads(from_manifest.read_text())["config"]
        # Restore the engine choice from the manifest if present.
        engine = cfg_dict.pop("engine", engine)
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

    manifest_cfg = asdict(config)
    manifest_cfg["engine"] = engine
    manifest = Manifest.create(run_id, config=manifest_cfg)
    write_manifest(manifest, config.run_dir / "manifest.json")

    console.print(f"[bold green]Run[/] {run_id} → {config.run_dir}")
    console.print(
        f"[dim]benchmark={config.benchmark} provider={config.llm_provider} "
        f"engine={engine} gens={config.generations} seed={config.seed}[/]"
    )

    if engine == "openevolve":
        from science_game.openevolve_adapter import run_with_openevolve

        best = run_with_openevolve(config)
    elif engine == "standalone":
        best = run_evolution(config)
    else:
        raise typer.BadParameter(f"unknown engine: {engine!r}. Use 'standalone' or 'openevolve'.")

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


COLAB_URL = (
    "https://colab.research.google.com/github/unslothai/unsloth/blob/main"
    "/studio/Unsloth_Studio_Colab.ipynb"
)


@phase4_app.command("prepare")
def phase4_prepare_cmd(
    dataset: Path = typer.Option(..., "--dataset", "-d", help="Path to mutator-dataset.jsonl"),
    gguf_name: str = typer.Option(
        "forge-mutator-v1.gguf", "--gguf-name",
        help="Filename of the GGUF you'll download after Colab fine-tuning.",
    ),
    out_dir: Path = typer.Option(Path("phase4-out"), "--out-dir"),
    base_model: str = typer.Option(
        "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit", "--base-model",
        help="Unsloth-compatible base model spec.",
    ),
    hf_dataset_repo: str = typer.Option(
        "Beko2210/algorithm-forge-mutations-v1", "--hf-dataset-repo",
    ),
    hf_model_repo: str = typer.Option(
        "Beko2210/algorithm-forge-mutator-qwen-v1", "--hf-model-repo",
    ),
    ollama_model_name: str = typer.Option(
        "algorithm-forge-mutator", "--ollama-name",
    ),
) -> None:
    """Pre-fine-tune helper: validate the dataset, emit a Modelfile, print the Colab URL."""
    import json

    from science_game.phase4.modelfile import write_modelfile

    if not dataset.exists():
        raise typer.BadParameter(f"dataset not found: {dataset}")

    examples = 0
    with dataset.open() as f:
        for line in f:
            if line.strip():
                try:
                    json.loads(line)
                    examples += 1
                except json.JSONDecodeError as e:
                    raise typer.BadParameter(f"malformed line in {dataset}: {e}") from e
    if examples == 0:
        raise typer.BadParameter(f"dataset {dataset} is empty")

    out_dir.mkdir(parents=True, exist_ok=True)
    modelfile_path = out_dir / "Modelfile"
    write_modelfile(modelfile_path, gguf_path=f"./{gguf_name}")

    instructions_path = out_dir / "README.md"
    instructions_path.write_text(_phase4_readme(
        dataset=dataset, examples=examples, gguf_name=gguf_name,
        base_model=base_model, hf_dataset_repo=hf_dataset_repo,
        hf_model_repo=hf_model_repo, ollama_model_name=ollama_model_name,
    ))

    table = Table(title="Phase 4 prepare")
    table.add_column("key")
    table.add_column("value")
    table.add_row("dataset", str(dataset))
    table.add_row("examples", str(examples))
    table.add_row("Modelfile", str(modelfile_path))
    table.add_row("Instructions", str(instructions_path))
    table.add_row("Colab", COLAB_URL)
    console.print(table)
    console.print(
        "\n[bold green]Next:[/] open the Colab URL above, run all cells, "
        "then in Studio select the base model and your HF dataset."
    )


@phase4_app.command("evaluate")
def phase4_evaluate_cmd(
    benchmark: str = typer.Option("matmul", "--benchmark", "-b"),
    base_provider: str = typer.Option("ollama-qwen", "--base-provider"),
    base_model: str | None = typer.Option(None, "--base-model"),
    finetuned_provider: str = typer.Option("ollama-qwen", "--finetuned-provider"),
    finetuned_model: str | None = typer.Option("algorithm-forge-mutator", "--finetuned-model"),
    seeds: str = typer.Option("0,1,2,3,4", "--seeds", help="Comma-separated seeds"),
    generations: int = typer.Option(20, "--generations", "-g"),
    runs_root: Path = typer.Option(Path("runs/ab"), "--runs-root"),
    out: Path = typer.Option(Path("phase4-out/ab-report.json"), "--out", "-o"),
) -> None:
    """A/B-compare base mutator vs. fine-tuned mutator on the same benchmark + seeds."""
    from science_game.phase4.ab_eval import run_ab, write_report

    seed_list = [int(s.strip()) for s in seeds.split(",") if s.strip()]
    console.print(
        f"[bold]A/B[/] benchmark={benchmark}, "
        f"base={base_provider}({base_model or 'default'}), "
        f"finetuned={finetuned_provider}({finetuned_model or 'default'}), "
        f"seeds={seed_list}, gens={generations}"
    )
    report = run_ab(
        benchmark=benchmark,
        base_provider=base_provider, base_model=base_model,
        finetuned_provider=finetuned_provider, finetuned_model=finetuned_model,
        seeds=seed_list, generations=generations, runs_root=runs_root,
    )
    write_report(report, out)

    table = Table(title="A/B Report")
    table.add_column("metric")
    table.add_column("value")
    table.add_row("wins finetuned", str(report.wins_finetuned))
    table.add_row("wins base", str(report.wins_base))
    table.add_row("ties", str(report.ties))
    table.add_row("avg delta (finetuned - base)", f"{report.avg_delta:+.6f}")
    table.add_row("report file", str(out))
    console.print(table)


def _phase4_readme(
    dataset: Path, examples: int, gguf_name: str,
    base_model: str, hf_dataset_repo: str, hf_model_repo: str,
    ollama_model_name: str,
) -> str:
    return (
        f"# Phase 4 prep for `{ollama_model_name}`\n\n"
        f"Built from dataset `{dataset}` ({examples} examples).\n\n"
        "## Steps\n\n"
        f"1. `hf upload-dataset {hf_dataset_repo} {dataset}` (or push via huggingface_hub).\n"
        f"2. Open [Unsloth Studio Colab]({COLAB_URL}); Runtime → T4 GPU; Run all.\n"
        f"3. In Studio: base model = `{base_model}`, dataset = `{hf_dataset_repo}`.\n"
        "4. Recipe: LoRA 4-bit, rank 16, alpha 16, lr 2e-4, 2-3 epochs.\n"
        f"5. Export GGUF (Q4_K_M) and push to `{hf_model_repo}`.\n"
        f"6. Download the GGUF locally and place it next to this Modelfile, named `{gguf_name}`.\n"
        f"7. `ollama create {ollama_model_name} -f Modelfile`.\n"
        f"8. `science-game phase4 evaluate --finetuned-model {ollama_model_name}` to A/B-compare.\n"
    )


if __name__ == "__main__":
    app()
