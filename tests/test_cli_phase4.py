"""CLI tests for `science-game phase4 prepare`."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from science_game.cli import app


def test_phase4_prepare_rejects_missing_dataset(tmp_path: Path):
    runner = CliRunner()
    result = runner.invoke(app, [
        "phase4", "prepare",
        "--dataset", str(tmp_path / "nope.jsonl"),
        "--out-dir", str(tmp_path / "out"),
    ])
    assert result.exit_code != 0


def test_phase4_prepare_rejects_empty_dataset(tmp_path: Path):
    ds = tmp_path / "empty.jsonl"
    ds.write_text("")
    runner = CliRunner()
    result = runner.invoke(app, [
        "phase4", "prepare", "--dataset", str(ds),
        "--out-dir", str(tmp_path / "out"),
    ])
    assert result.exit_code != 0


def test_phase4_prepare_writes_artifacts(tmp_path: Path):
    ds = tmp_path / "ds.jsonl"
    ds.write_text(json.dumps({"messages": [
        {"role": "user", "content": "p"},
        {"role": "assistant", "content": "c"},
    ]}) + "\n")
    out_dir = tmp_path / "out"
    runner = CliRunner()
    result = runner.invoke(app, [
        "phase4", "prepare",
        "--dataset", str(ds),
        "--out-dir", str(out_dir),
        "--gguf-name", "demo.gguf",
    ])
    assert result.exit_code == 0, result.output

    modelfile = out_dir / "Modelfile"
    readme = out_dir / "README.md"
    assert modelfile.exists()
    assert readme.exists()

    mf = modelfile.read_text()
    assert "FROM ./demo.gguf" in mf
    assert "SYSTEM" in mf

    rd = readme.read_text()
    assert "demo.gguf" in rd
    assert "Unsloth Studio Colab" in rd
    assert "ollama create" in rd
