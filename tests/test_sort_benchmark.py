"""Tests for the sorting-network benchmark."""

from __future__ import annotations

import textwrap

from science_game.benchmarks import get_benchmark


def test_seed_program_is_bubble_sort():
    bench = get_benchmark("sort")
    result = bench.evaluate(bench.seed_program)
    assert result.correct
    assert result.metrics["comparators"] == 28
    assert result.metrics["beats_bubble"] is False
    assert result.metrics["matches_knuth_optimum"] is False


def test_knuth_19_comparator_network_for_n8_beats_seed():
    """The known-optimal N=8 19-comparator sorting network."""
    bench = get_benchmark("sort")
    optimal = textwrap.dedent(
        """
        def build_network():
            return [
                (0, 1), (2, 3), (4, 5), (6, 7),
                (0, 2), (1, 3), (4, 6), (5, 7),
                (1, 2), (5, 6), (0, 4), (3, 7),
                (1, 5), (2, 6),
                (1, 4), (3, 6),
                (2, 4), (3, 5),
                (3, 4),
            ]
        """
    ).strip()
    result = bench.evaluate(optimal)
    assert result.correct, f"expected correct, got {result.error}"
    assert result.metrics["comparators"] == 19
    assert result.metrics["beats_bubble"] is True
    assert result.metrics["matches_knuth_optimum"] is True


def test_invalid_pair_format_rejected():
    bench = get_benchmark("sort")
    bad = "def build_network():\n    return [(0, 0)]\n"  # i >= j
    result = bench.evaluate(bad)
    assert not result.correct
    assert "invalid pair" in (result.error or "")


def test_missing_function_rejected():
    bench = get_benchmark("sort")
    result = bench.evaluate("x = 1\n")
    assert not result.correct
    assert "did not define" in (result.error or "")


def test_incorrect_network_scores_low():
    """A trivial 'do nothing' returns the input unsorted -> 1/256 zero-one tests pass."""
    bench = get_benchmark("sort")
    nothing = "def build_network():\n    return []\n"
    result = bench.evaluate(nothing)
    assert not result.correct
    # The all-zeros and all-ones binary inputs are already sorted, so a few cases pass.
    assert result.metrics["correctness"] < 0.1
