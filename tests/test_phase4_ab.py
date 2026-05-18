"""Tests for the A/B evaluator — uses the scripted provider trick from
test_engine_smoke so we don't need Ollama."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

from science_game.llm.base import LLMProvider, MutationRequest, MutationResponse
from science_game.phase4.ab_eval import _decide_winner, run_ab, write_report


class _AlwaysBestProvider(LLMProvider):
    """Mutator that always returns the Strassen 7-mult algorithm."""
    name = "best"
    _code = textwrap.dedent(
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
            return [[m1+m4-m5+m7, m3+m5], [m2+m4, m1-m2+m3+m6]]
        """
    ).strip()

    def __init__(self, **_): pass
    def mutate(self, request: MutationRequest) -> MutationResponse:
        return MutationResponse(
            child_code=self._code, raw_text=self._code,
            provider=self.name, model="best",
        )


class _AlwaysNoopProvider(LLMProvider):
    """Mutator that returns the seed unchanged — never improves."""
    name = "noop"

    def __init__(self, **_): pass
    def mutate(self, request: MutationRequest) -> MutationResponse:
        return MutationResponse(
            child_code=request.parent_code, raw_text=request.parent_code,
            provider=self.name, model="noop",
        )


def _patch_providers(monkeypatch):
    def fake_get_provider(name, **kw):
        if name == "best":
            return _AlwaysBestProvider(**kw)
        if name == "noop":
            return _AlwaysNoopProvider(**kw)
        raise ValueError(name)
    monkeypatch.setattr("science_game.engine.get_provider", fake_get_provider)


def test_decide_winner():
    assert _decide_winner(0.1, 0.2) == "finetuned"
    assert _decide_winner(0.2, 0.1) == "base"
    assert _decide_winner(0.1, 0.1) == "tie"


def test_ab_eval_finetuned_wins(tmp_path: Path, monkeypatch):
    _patch_providers(monkeypatch)
    report = run_ab(
        benchmark="matmul",
        base_provider="noop", finetuned_provider="best",
        seeds=[0, 1, 2], generations=1, runs_root=tmp_path,
    )
    assert report.wins_finetuned == 3
    assert report.wins_base == 0
    assert report.avg_delta > 0


def test_ab_eval_writes_report(tmp_path: Path, monkeypatch):
    _patch_providers(monkeypatch)
    report = run_ab(
        benchmark="matmul",
        base_provider="best", finetuned_provider="noop",  # base wins
        seeds=[0, 1], generations=1, runs_root=tmp_path / "runs",
    )
    out = tmp_path / "report.json"
    write_report(report, out)
    data = json.loads(out.read_text())
    assert "summary" in data
    assert data["summary"]["wins_base"] == 2


def test_ab_eval_tie_when_both_noop(tmp_path: Path, monkeypatch):
    _patch_providers(monkeypatch)
    report = run_ab(
        benchmark="matmul",
        base_provider="noop", finetuned_provider="noop",
        seeds=[0, 1], generations=1, runs_root=tmp_path,
    )
    assert report.ties == 2
