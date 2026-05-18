"""Test that --from-manifest reproduces a run.

Drives the CLI in-process via Typer's runner with a scripted provider
plumbed in through a small monkey-patch — no Ollama, no API.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

from typer.testing import CliRunner

from science_game.cli import app
from science_game.llm.base import LLMProvider, MutationRequest, MutationResponse


class _Scripted(LLMProvider):
    name = "scripted"

    def __init__(self, **_: object) -> None:
        self.code = textwrap.dedent(
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

    def mutate(self, request: MutationRequest) -> MutationResponse:
        return MutationResponse(
            child_code=self.code, raw_text=self.code,
            provider=self.name, model="scripted",
        )


def test_from_manifest_round_trips(tmp_path: Path, monkeypatch):
    runs_root = tmp_path / "runs"
    runs_root.mkdir()

    # Force the CLI's provider factory to return our scripted one.
    from science_game.llm import __init__ as llm_pkg  # noqa: F401

    def fake_get_provider(name, **kw):
        return _Scripted(**kw)

    monkeypatch.setattr("science_game.engine.get_provider", fake_get_provider)

    runner = CliRunner()

    # First run
    r1 = runner.invoke(app, [
        "run", "matmul", "--provider", "scripted",
        "--generations", "1", "--seed", "7",
        "--runs-root", str(runs_root),
    ])
    assert r1.exit_code == 0, r1.output

    # Find the manifest produced
    manifests = list(runs_root.glob("*/manifest.json"))
    assert len(manifests) == 1
    first_manifest = manifests[0]

    # Re-run from it
    r2 = runner.invoke(app, [
        "run",
        "--from-manifest", str(first_manifest),
        "--runs-root", str(runs_root),
    ])
    assert r2.exit_code == 0, r2.output

    # We now have two run dirs; the rerun one has 'rerun' in the name
    rerun = next(p for p in runs_root.iterdir() if "rerun" in p.name)
    rerun_manifest = json.loads((rerun / "manifest.json").read_text())
    orig_manifest = json.loads(first_manifest.read_text())
    # Same config fields, different run_id + run_dir
    assert rerun_manifest["config"]["benchmark"] == orig_manifest["config"]["benchmark"]
    assert rerun_manifest["config"]["seed"] == orig_manifest["config"]["seed"]
    assert rerun_manifest["run_id"] != orig_manifest["run_id"]
