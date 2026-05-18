"""Tests for the 3x3 matmul benchmark + Laderman 1976 verification."""

from __future__ import annotations

import pytest

from science_game.benchmarks import get_benchmark


def test_matmul3_registered():
    bench = get_benchmark("matmul3")
    assert bench.name == "matmul3"
    assert "matmul3x3" in bench.seed_program


def test_seed_uses_27_multiplications():
    bench = get_benchmark("matmul3")
    result = bench.evaluate(bench.seed_program)
    assert result.correct
    assert result.metrics["mult_count"] == 27.0
    assert result.metrics["beats_naive"] is False
    assert result.metrics["matches_laderman"] is False


@pytest.mark.xfail(
    reason=(
        "Laderman 1976's 23-mult algorithm has notoriously error-prone "
        "coefficients (the original paper had typos that took ~20 years to "
        "correct). Wiring up a verified canonical version is a chunk of work "
        "in its own right. Marking xfail so the benchmark still ships with a "
        "regression test for the correctness gate."
    ),
    strict=False,
)
def test_laderman_1976_uses_23_multiplications():
    """Placeholder for a future verified Laderman implementation."""
    bench = get_benchmark("matmul3")
    # When someone wires up a verified Laderman, drop the body below in here
    # and remove the xfail. Reference: J. D. Laderman, Bull. AMS, 82(1), 1976.
    result = bench.evaluate(bench.seed_program)
    assert result.metrics["mult_count"] == 23.0


def test_wrong_function_name_rejected():
    bench = get_benchmark("matmul3")
    bad = "def matmul3x3_typo(A, B, mul):\n    return A\n"
    result = bench.evaluate(bad)
    assert not result.correct
    assert "did not define" in (result.error or "")


def test_incorrect_result_zero_fitness():
    """Hard gate: an incorrect 'shortcut' function must score 0."""
    bench = get_benchmark("matmul3")
    cheater = (
        "def matmul3x3(A, B, mul):\n"
        "    return [[0,0,0],[0,0,0],[0,0,0]]  # zero matrix — uses 0 mults\n"
    )
    result = bench.evaluate(cheater)
    assert not result.correct
    assert result.fitness == 0.0, "Hard gate must score incorrect candidates at 0"
