"""Tests for the dashboard's pure-Python helpers (no Streamlit needed)."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

from science_game.benchmarks import get_benchmark
from science_game.dashboard.runs_io import (
    list_best_files,
    list_runs,
    load_events,
    load_mutations,
)
from science_game.engine import Engine, EvolutionConfig
from science_game.llm.base import LLMProvider, MutationRequest, MutationResponse


class ScriptedProvider(LLMProvider):
    name = "scripted"

    def __init__(self, programs: list[str]) -> None:
        self.programs = programs
        self.idx = 0

    def mutate(self, request: MutationRequest) -> MutationResponse:
        code = self.programs[self.idx % len(self.programs)]
        self.idx += 1
        return MutationResponse(
            child_code=code, raw_text=code, provider=self.name, model="scripted-1",
        )


def _seed_run(run_dir: Path) -> None:
    """Drive a tiny end-to-end run so we have real artifacts to inspect."""
    bench = get_benchmark("matmul")
    strassen = textwrap.dedent(
        """
        def matmul2x2(A, B, mul):
            a, b = A[0][0], A[0][1]
            c, d = A[1][0], A[1][1]
            e, f = B[0][0], B[0][1]
            g, h = B[1][0], B[1][1]
            m1 = mul(a + d, e + h)
            m2 = mul(c + d, e)
            m3 = mul(a, f - h)
            m4 = mul(d, g - e)
            m5 = mul(a + b, h)
            m6 = mul(c - a, e + f)
            m7 = mul(b - d, g + h)
            return [[m1 + m4 - m5 + m7, m3 + m5], [m2 + m4, m1 - m2 + m3 + m6]]
        """
    ).strip()
    provider = ScriptedProvider([strassen])
    config = EvolutionConfig(
        benchmark="matmul", llm_provider="scripted",
        generations=2, seed=0, run_dir=run_dir,
    )
    Engine(config, bench, provider).run()
    # Minimal manifest so list_runs picks it up.
    (run_dir / "manifest.json").write_text(json.dumps({
        "run_id": run_dir.name,
        "created_at": "2026-05-18T00:00:00+00:00",
        "config": {"benchmark": "matmul", "llm_provider": "scripted"},
        "git_sha": "deadbeef",
        "git_dirty": False,
    }))


def test_list_runs_empty(tmp_path: Path):
    assert list_runs(tmp_path) == []


def test_list_runs_and_load(tmp_path: Path):
    run_dir = tmp_path / "matmul-test01"
    run_dir.mkdir()
    _seed_run(run_dir)

    runs = list_runs(tmp_path)
    assert len(runs) == 1
    r = runs[0]
    assert r.run_id == "matmul-test01"
    assert r.benchmark == "matmul"
    assert r.provider == "scripted"
    assert r.best_fitness is not None
    assert r.best_fitness > 0
    assert r.generations_logged >= 1


def test_load_events_columns(tmp_path: Path):
    run_dir = tmp_path / "r1"
    run_dir.mkdir()
    _seed_run(run_dir)

    df = load_events(run_dir)
    assert not df.empty
    assert {"generation", "fitness", "best_so_far"}.issubset(df.columns)
    # Monotonic best-so-far
    assert df["best_so_far"].is_monotonic_increasing


def test_load_mutations_has_phase4_fields(tmp_path: Path):
    run_dir = tmp_path / "r1"
    run_dir.mkdir()
    _seed_run(run_dir)

    mdf = load_mutations(run_dir)
    assert not mdf.empty
    for col in ("parent_code", "child_code", "fitness_before", "fitness_after", "accepted"):
        assert col in mdf.columns, f"missing {col} — Phase 4 needs it"


def test_list_best_files_excludes_latest(tmp_path: Path):
    run_dir = tmp_path / "r1"
    run_dir.mkdir()
    _seed_run(run_dir)

    files = list_best_files(run_dir)
    assert all(p.name != "latest.py" for p in files)
    assert len(files) >= 1
