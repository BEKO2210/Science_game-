"""Helpers for reading runs/ from disk — shared between dashboard pages.

Kept dependency-light (stdlib + pandas) so unit tests don't need Streamlit.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DEFAULT_RUNS_ROOT = Path("runs")


@dataclass
class RunSummary:
    run_id: str
    path: Path
    manifest: dict
    best_fitness: float | None
    generations_logged: int

    @property
    def benchmark(self) -> str:
        return self.manifest.get("config", {}).get("benchmark", "?")

    @property
    def provider(self) -> str:
        return self.manifest.get("config", {}).get("llm_provider", "?")


def list_runs(runs_root: Path = DEFAULT_RUNS_ROOT) -> list[RunSummary]:
    """Discover all runs under `runs_root`. A run is any dir with manifest.json."""
    if not runs_root.exists():
        return []
    out: list[RunSummary] = []
    for child in sorted(runs_root.iterdir()):
        if not child.is_dir():
            continue
        mpath = child / "manifest.json"
        if not mpath.exists():
            continue
        try:
            manifest = json.loads(mpath.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        df = load_events(child)
        best = float(df["fitness"].max()) if not df.empty and "fitness" in df.columns else None
        gens = int(df["generation"].max()) if not df.empty and "generation" in df.columns else 0
        out.append(
            RunSummary(
                run_id=child.name,
                path=child,
                manifest=manifest,
                best_fitness=best,
                generations_logged=gens,
            )
        )
    out.sort(key=lambda r: r.manifest.get("created_at", ""), reverse=True)
    return out


def load_events(run_dir: Path) -> pd.DataFrame:
    """Read events.jsonl into a long-format DataFrame with a `fitness` column.

    Combines init + per-generation rows. Returns empty DataFrame if no events yet.
    """
    path = run_dir / "events.jsonl"
    if not path.exists():
        return pd.DataFrame()

    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = rec.get("kind")
            gen = rec.get("generation", 0)
            if kind == "init":
                rows.append({
                    "generation": gen,
                    "fitness": rec.get("fitness"),
                    "kind": kind,
                    "accepted": True,
                })
            elif kind == "generation":
                rows.append({
                    "generation": gen,
                    "fitness": (
                        rec.get("fitness_child")
                        if rec.get("accepted")
                        else rec.get("fitness_parent")
                    ),
                    "fitness_child": rec.get("fitness_child"),
                    "fitness_parent": rec.get("fitness_parent"),
                    "accepted": rec.get("accepted"),
                    "kind": kind,
                    "llm_seconds": rec.get("llm_seconds"),
                    "tokens_in": rec.get("tokens_in"),
                    "tokens_out": rec.get("tokens_out"),
                })
            elif kind == "llm_error":
                rows.append({
                    "generation": gen,
                    "kind": kind,
                    "error": rec.get("error"),
                })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("generation").reset_index(drop=True)
        # Best-so-far line for nice monotonic plots
        if "fitness" in df.columns:
            df["best_so_far"] = df["fitness"].cummax()
    return df


def load_mutations(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "mutations.jsonl"
    if not path.exists():
        return pd.DataFrame()
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(rows)


def list_best_files(run_dir: Path) -> list[Path]:
    best_dir = run_dir / "best"
    if not best_dir.exists():
        return []
    files = [p for p in best_dir.iterdir() if p.is_file() and p.suffix == ".py" and p.name != "latest.py"]
    return sorted(files)
