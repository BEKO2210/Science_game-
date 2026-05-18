"""Regression test: CLI must work when stdout is non-UTF-8 (Windows default).

The original bug: on Windows PowerShell, when pytest spawns the CLI as a
subprocess, sys.stdout.encoding becomes 'cp1252' and Rich's print() crashes
on any non-ASCII character (we had a `→` arrow in three places).

We simulate this by forcing PYTHONIOENCODING=cp1252 on the subprocess.
Run on every platform — protects Linux/macOS CI from regressions that
would only manifest on Windows.

Also covers file writes: phase4 prepare writes a README via Path.write_text
which defaults to the platform encoding; that broke on cp1252.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def _run_cp1252(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    """Spawn CLI with cp1252 stdout to simulate Windows. Returns bytes — the
    test is about whether the subprocess crashed, not about decoding its
    output cleanly."""
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "cp1252"
    env["PYTHONUTF8"] = "0"  # disable UTF-8 mode that's default since 3.15
    cmd = [sys.executable, "-m", "science_game.cli", *args]
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, env=env, timeout=60)
    # Decode with replacement so tests can still inspect output.
    proc.stdout = proc.stdout.decode("cp1252", errors="replace") if proc.stdout else ""
    proc.stderr = proc.stderr.decode("cp1252", errors="replace") if proc.stderr else ""
    return proc


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def test_cli_run_works_with_cp1252_stdout(repo_root: Path, tmp_path: Path):
    """Reproduces the original Windows failure: CLI printed `\\u2192` and died."""
    runs_root = tmp_path / "runs"
    result = _run_cp1252(
        ["run", "sort", "--provider", "mock", "--generations", "2", "--runs-root", str(runs_root)],
        cwd=repo_root,
    )
    assert result.returncode == 0, (
        f"CLI crashed under cp1252 stdout — "
        f"the Windows bug is back.\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    )


def test_cli_doctor_works_with_cp1252_stdout(repo_root: Path):
    result = _run_cp1252(["doctor"], cwd=repo_root)
    # doctor exits 0 when no REQUIRED checks fail. Even if Ollama is missing
    # on the test machine, the CLI must at least print without UnicodeError.
    assert "UnicodeEncodeError" not in result.stderr


def test_cli_build_mutator_dataset_works_with_cp1252_stdout(repo_root: Path, tmp_path: Path):
    runs_root = tmp_path / "runs"
    # Make a run first.
    _run_cp1252(
        ["run", "sort", "--provider", "mock", "-g", "2", "--runs-root", str(runs_root)],
        cwd=repo_root,
    )
    out = tmp_path / "ds.jsonl"
    result = _run_cp1252(
        ["build-mutator-dataset", "--runs-root", str(runs_root), "--out", str(out), "--mode", "sft"],
        cwd=repo_root,
    )
    assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"


def test_phase4_prepare_writes_readme_as_utf8(repo_root: Path, tmp_path: Path):
    """The README contains paths with potential non-ASCII chars (user folder
    names on Windows can have umlauts). Must be written as UTF-8."""
    ds = tmp_path / "ds.jsonl"
    ds.write_text(
        json.dumps({"messages": [
            {"role": "user", "content": "p"},
            {"role": "assistant", "content": "c"},
        ]}) + "\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"
    result = _run_cp1252(
        ["phase4", "prepare", "--dataset", str(ds), "--out-dir", str(out_dir), "--gguf-name", "demo.gguf"],
        cwd=repo_root,
    )
    assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
    # File must be readable as UTF-8 — the original bug was a charmap-encode crash.
    readme = (out_dir / "README.md").read_text(encoding="utf-8")
    assert "Phase 4 prep" in readme
