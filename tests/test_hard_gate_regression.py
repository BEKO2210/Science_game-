"""Regression tests for the hard correctness gate.

Backstory: the original fitness function was `correctness * 1/comparators`
which lets the optimizer prefer partial-correct shorter solutions (e.g.
70%-correct 19-comparator network > 100%-correct 28-comparator network).
The hard gate sets fitness to 0 unless correctness == 1.0, which is the
ONLY thing that makes the LLM-driven loop converge in practice.

If these tests start failing, someone removed the gate — don't merge.
"""

from __future__ import annotations

import textwrap

from science_game.benchmarks import get_benchmark

# --- sort --------------------------------------------------------------------

def test_sort_hard_gate_partial_correct_scores_zero():
    """A network that sorts 99% of inputs must still score 0."""
    bench = get_benchmark("sort")
    # Drop the last pair of the correct all-pairs sort -> not quite correct
    almost = textwrap.dedent(
        """
        def build_network():
            N = 8
            pairs = []
            for i in range(N):
                for j in range(i + 1, N):
                    pairs.append((i, j))
            return pairs[:-1]
        """
    ).strip()
    result = bench.evaluate(almost)
    # Verify it's actually partial-correct (i.e. the test is meaningful).
    assert result.metrics["correctness"] < 1.0
    assert result.metrics["correctness"] > 0.0
    # The whole point of the hard gate:
    assert result.fitness == 0.0
    assert not result.correct


def test_sort_hard_gate_correct_wins_over_smaller_buggy():
    """The 28-comparator correct seed must outscore a smaller broken network."""
    bench = get_benchmark("sort")
    correct_28 = bench.evaluate(bench.seed_program).fitness
    broken_small = bench.evaluate(
        "def build_network():\n    return [(0,1), (2,3), (4,5)]\n"
    ).fitness
    assert correct_28 > broken_small
    assert broken_small == 0.0


# --- matmul ------------------------------------------------------------------

def test_matmul_hard_gate_partial_correct_scores_zero():
    """A buggy 'Strassen-like' that sometimes mis-multiplies must score 0."""
    bench = get_benchmark("matmul")
    # Replace one of the eight required multiplications with a constant.
    buggy = textwrap.dedent(
        """
        def matmul2x2(A, B, mul):
            a, b = A[0][0], A[0][1]
            c, d = A[1][0], A[1][1]
            e, f = B[0][0], B[0][1]
            g, h = B[1][0], B[1][1]
            return [[mul(a, e) + mul(b, g), mul(a, f) + 0.0],
                    [mul(c, e) + mul(d, g), mul(c, f) + mul(d, h)]]
        """
    ).strip()
    result = bench.evaluate(buggy)
    assert result.metrics["correctness"] < 1.0
    assert result.fitness == 0.0


def test_matmul_hard_gate_correct_seed_beats_buggy_strassen():
    """Correct 8-mult seed must outscore a buggy 7-mult."""
    bench = get_benchmark("matmul")
    seed_fit = bench.evaluate(bench.seed_program).fitness
    assert seed_fit > 0.0
    # A "Strassen" that swaps two formulas — fewer mults but wrong.
    wrong_strassen = textwrap.dedent(
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
            # Bug: swapped m4 and m5 in c[0][0]
            return [[m1 + m5 - m4 + m7, m3 + m5], [m2 + m4, m1 - m2 + m3 + m6]]
        """
    ).strip()
    wrong_fit = bench.evaluate(wrong_strassen).fitness
    assert wrong_fit == 0.0


# --- matmul3 -----------------------------------------------------------------

def test_matmul3_hard_gate_partial_correct_scores_zero():
    bench = get_benchmark("matmul3")
    buggy = textwrap.dedent(
        """
        def matmul3x3(A, B, mul):
            # Returns A unchanged — coincidentally correct when B == I.
            return A
        """
    ).strip()
    result = bench.evaluate(buggy)
    assert result.fitness == 0.0
