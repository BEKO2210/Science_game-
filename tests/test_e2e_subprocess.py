"""End-to-end tests: spawn the actual `science-game` CLI as a subprocess
and verify that every artifact the dashboard depends on is produced.

These are the tests that would have caught the "looks like everything is
fine but nothing actually runs" failure mode. They use the mock provider
so they don't need Ollama, GPU, or API keys.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


def _run_cli(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    cmd = [sys.executable, "-m", "science_game.cli", *args]
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=60)


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def test_e2e_sort_run_writes_full_artifact_set(repo_root: Path, tmp_path: Path):
    runs_root = tmp_path / "runs"
    result = _run_cli(
        ["run", "sort", "--provider", "mock", "--generations", "3", "--runs-root", str(runs_root)],
        cwd=repo_root,
    )
    assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"

    run_dirs = list(runs_root.glob("sort-*"))
    assert len(run_dirs) == 1
    run = run_dirs[0]

    # Manifest
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["run_id"] == run.name
    assert manifest["config"]["benchmark"] == "sort"
    assert manifest["config"]["llm_provider"] == "mock"
    assert manifest["config"]["engine"] == "standalone"
    assert manifest["git_sha"]  # non-empty

    # Events: 1 init + 3 generations
    events_lines = (run / "events.jsonl").read_text().strip().splitlines()
    assert len(events_lines) == 4
    init_evt = json.loads(events_lines[0])
    assert init_evt["kind"] == "init"
    assert init_evt["metrics"]["comparators"] == 28

    # Mutations: 3 rows, Phase-4-shaped
    mutations_lines = (run / "mutations.jsonl").read_text().strip().splitlines()
    assert len(mutations_lines) == 3
    first_mut = json.loads(mutations_lines[0])
    for col in ("parent_code", "child_code", "fitness_before", "fitness_after", "accepted"):
        assert col in first_mut, f"mutations.jsonl missing required field {col!r}"
    assert first_mut["accepted"] is True

    # Best snapshots: gen0 (28-comparator seed) + gen1 (mock's 19-comp answer) + latest.py
    best_files = sorted((run / "best").iterdir())
    best_names = [p.name for p in best_files]
    assert "latest.py" in best_names
    assert any("gen0000" in n for n in best_names)
    assert any("gen0001" in n for n in best_names)


def test_e2e_matmul_run_finds_strassen(repo_root: Path, tmp_path: Path):
    runs_root = tmp_path / "runs"
    result = _run_cli(
        ["run", "matmul", "--provider", "mock", "--generations", "2", "--runs-root", str(runs_root)],
        cwd=repo_root,
    )
    assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"

    run = next(runs_root.glob("matmul-*"))
    events = [json.loads(line) for line in (run / "events.jsonl").read_text().strip().splitlines()]
    # init has the 8-mult naive, accepted gen has 7-mult Strassen.
    init = events[0]
    assert init["metrics"]["mult_count"] == 8
    accepted_gen = next(e for e in events[1:] if e["accepted"])
    assert accepted_gen["child_metrics"]["mult_count"] == 7
    assert accepted_gen["child_metrics"]["beats_naive"] is True


def test_e2e_doctor_command_runs(repo_root: Path):
    result = _run_cli(["doctor"], cwd=repo_root)
    # doctor exits non-zero only on REQUIRED failures. In our test env all
    # required things are installed; ollama is optional so it can be missing.
    assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
    assert "All required checks passed" in result.stdout


def test_e2e_build_mutator_dataset_from_real_run(repo_root: Path, tmp_path: Path):
    runs_root = tmp_path / "runs"
    # Generate two runs to aggregate.
    _run_cli(["run", "sort", "--provider", "mock", "-g", "2", "--runs-root", str(runs_root)], cwd=repo_root)
    _run_cli(["run", "matmul", "--provider", "mock", "-g", "2", "--runs-root", str(runs_root)], cwd=repo_root)

    out = tmp_path / "ds.jsonl"
    result = _run_cli(
        ["build-mutator-dataset", "--runs-root", str(runs_root), "--out", str(out), "--mode", "sft"],
        cwd=repo_root,
    )
    assert result.returncode == 0, result.stderr

    lines = out.read_text().strip().splitlines()
    assert len(lines) >= 2  # at least one accepted improvement per benchmark
    for line in lines:
        row = json.loads(line)
        assert "messages" in row and len(row["messages"]) == 2
        assert row["messages"][0]["role"] == "user"
        assert row["messages"][1]["role"] == "assistant"
