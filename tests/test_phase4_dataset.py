"""Tests for the Phase 4 mutator-dataset builder."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from science_game.phase4.dataset import build_dataset


def _write_mutations(run_dir: Path, mutations: list[dict]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "mutations.jsonl").open("w") as f:
        for m in mutations:
            f.write(json.dumps(m) + "\n")


def test_sft_keeps_only_accepted_improving(tmp_path: Path):
    runs_root = tmp_path / "runs"
    _write_mutations(runs_root / "r1", [
        {"parent_code": "def p(): pass", "child_code": "def c1(): pass",
         "fitness_before": 0.1, "fitness_after": 0.2, "accepted": True,
         "provider": "ollama", "model": "qwen"},
        {"parent_code": "def p(): pass", "child_code": "def c2(): pass",
         "fitness_before": 0.1, "fitness_after": 0.0, "accepted": False,
         "provider": "ollama", "model": "qwen"},
    ])

    out = tmp_path / "ds.jsonl"
    n = build_dataset(runs_root, out, mode="sft")
    assert n == 1
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 1
    ex = json.loads(lines[0])
    assert ex["messages"][0]["role"] == "user"
    assert "def p()" in ex["messages"][0]["content"]
    assert "```python" in ex["messages"][1]["content"]
    assert ex["meta"]["delta"] == pytest.approx(0.1)


def test_sft_respects_min_delta(tmp_path: Path):
    runs_root = tmp_path / "runs"
    _write_mutations(runs_root / "r1", [
        {"parent_code": "p", "child_code": "c1",
         "fitness_before": 0.10, "fitness_after": 0.11, "accepted": True},
        {"parent_code": "p", "child_code": "c2",
         "fitness_before": 0.10, "fitness_after": 0.50, "accepted": True},
    ])

    out = tmp_path / "ds.jsonl"
    n = build_dataset(runs_root, out, mode="sft", min_delta=0.05)
    assert n == 1
    ex = json.loads(out.read_text().strip())
    assert "c2" in ex["messages"][1]["content"]


def test_dpo_pairs_best_and_worst_per_parent(tmp_path: Path):
    runs_root = tmp_path / "runs"
    _write_mutations(runs_root / "r1", [
        {"parent_code": "p", "child_code": "c1", "fitness_after": 0.3, "accepted": True},
        {"parent_code": "p", "child_code": "c2", "fitness_after": 0.5, "accepted": True},
        {"parent_code": "p", "child_code": "c3", "fitness_after": 0.1, "accepted": False},
        {"parent_code": "q", "child_code": "d1", "fitness_after": 0.4, "accepted": True},
    ])

    out = tmp_path / "ds.jsonl"
    n = build_dataset(runs_root, out, mode="dpo")
    assert n == 1  # q has only one sibling -> skipped
    ex = json.loads(out.read_text().strip())
    assert "c2" in ex["chosen"]
    assert "c3" in ex["rejected"]
    assert ex["meta"]["chosen_fitness"] == 0.5


def test_unknown_mode_raises(tmp_path: Path):
    (tmp_path / "runs").mkdir()
    with pytest.raises(ValueError):
        build_dataset(tmp_path / "runs", tmp_path / "x.jsonl", mode="??")


def test_aggregates_across_runs(tmp_path: Path):
    runs_root = tmp_path / "runs"
    _write_mutations(runs_root / "r1", [
        {"parent_code": "p", "child_code": "c1",
         "fitness_before": 0.1, "fitness_after": 0.2, "accepted": True},
    ])
    _write_mutations(runs_root / "r2", [
        {"parent_code": "q", "child_code": "d1",
         "fitness_before": 0.1, "fitness_after": 0.3, "accepted": True},
    ])
    out = tmp_path / "ds.jsonl"
    n = build_dataset(runs_root, out, mode="sft")
    assert n == 2
