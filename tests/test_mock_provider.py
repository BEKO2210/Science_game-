"""Tests for the mock LLM provider — used by E2E smoke tests."""

from __future__ import annotations

from science_game.benchmarks import get_benchmark
from science_game.llm import get_provider
from science_game.llm.base import MutationRequest


def test_mock_returns_strassen_for_matmul():
    p = get_provider("mock")
    bench = get_benchmark("matmul")
    resp = p.mutate(MutationRequest(
        parent_code=bench.seed_program,
        task_description=bench.task_description,
        benchmark="matmul",
    ))
    assert resp.child_code
    result = bench.evaluate(resp.child_code)
    assert result.correct
    # Strassen uses 7 multiplications instead of the naive 8.
    assert result.metrics["mult_count"] == 7
    assert result.metrics["beats_naive"] is True
    assert result.metrics["beats_strassen"] is False  # 7 == 7, not <


def test_mock_returns_knuth19_for_sort():
    p = get_provider("mock")
    bench = get_benchmark("sort")
    resp = p.mutate(MutationRequest(
        parent_code=bench.seed_program,
        task_description=bench.task_description,
        benchmark="sort",
    ))
    result = bench.evaluate(resp.child_code)
    assert result.correct
    assert result.metrics["comparators"] == 19


def test_mock_echoes_parent_for_unknown_benchmark():
    p = get_provider("mock")
    resp = p.mutate(MutationRequest(
        parent_code="def foo(): return 42",
        task_description="t",
        benchmark="who-knows",
    ))
    assert resp.child_code == "def foo(): return 42"
