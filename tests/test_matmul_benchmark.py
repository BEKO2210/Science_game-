"""Smoke tests for the matmul benchmark.

These verify the evaluator itself — not the LLM. They run in <1s, no GPU,
no network. The point is: when the LLM evolves code in a real run, the
evaluator must correctly score it. We check three reference programs:
    1. The seed program (naive 8 mults, must score correct=True).
    2. A wrong program (must score correct=False, fitness 0).
    3. Strassen's 7-mult algorithm (must score correct=True, mult_count=7).
"""

from __future__ import annotations

import textwrap

from science_game.benchmarks import get_benchmark


def test_seed_program_is_correct_naive():
    bench = get_benchmark("matmul")
    result = bench.evaluate(bench.seed_program)
    assert result.correct, f"seed must be correct, got error={result.error}"
    assert result.metrics["mult_count"] == 8
    assert result.metrics["beats_naive"] is False
    assert result.metrics["beats_strassen"] is False
    assert result.fitness == 1.0 / 8


def test_broken_program_scores_zero():
    bench = get_benchmark("matmul")
    bad = textwrap.dedent(
        """
        def matmul2x2(A, B, mul):
            return [[0, 0], [0, 0]]
        """
    ).strip()
    result = bench.evaluate(bad)
    assert not result.correct
    assert result.fitness == 0.0


def test_strassen_beats_naive():
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

            c00 = m1 + m4 - m5 + m7
            c01 = m3 + m5
            c10 = m2 + m4
            c11 = m1 - m2 + m3 + m6
            return [[c00, c01], [c10, c11]]
        """
    ).strip()
    result = bench.evaluate(strassen)
    assert result.correct, f"Strassen must be correct, got error={result.error}"
    assert result.metrics["mult_count"] == 7
    assert result.metrics["beats_naive"] is True
    assert result.metrics["beats_strassen"] is False  # ties, doesn't strictly beat
    assert result.fitness == 1.0 / 7


def test_missing_function_scores_zero():
    bench = get_benchmark("matmul")
    result = bench.evaluate("x = 1\n")
    assert not result.correct
    assert result.fitness == 0.0
    assert "did not define matmul2x2" in (result.error or "")


def test_compile_error_handled():
    bench = get_benchmark("matmul")
    result = bench.evaluate("def matmul2x2(A, B, mul:\n")  # syntax error
    assert not result.correct
    assert result.fitness == 0.0
    assert result.error is not None
