"""Tests for the MNIST NAS benchmark.

We deliberately do not download MNIST in tests (slow + flaky network). The
checks here verify the wiring: torch availability detection, error paths,
seed-program-text shape, and registration. The full train+eval loop is
exercised on the user PC via scripts/run_local.sh.
"""

from __future__ import annotations

import importlib.util

import pytest

from science_game.benchmarks import get_benchmark

HAS_TORCH = importlib.util.find_spec("torch") is not None


def test_mnist_nas_registered():
    bench = get_benchmark("mnist_nas")
    assert bench.name == "mnist_nas"
    assert "Net" in bench.seed_program
    assert "nn.Module" in bench.seed_program


@pytest.mark.skipif(HAS_TORCH, reason="this path only triggers without torch")
def test_mnist_nas_without_torch_returns_clean_error():
    bench = get_benchmark("mnist_nas")
    result = bench.evaluate(bench.seed_program)
    assert result.fitness == 0.0
    assert not result.correct
    assert "torch" in (result.error or "")


@pytest.mark.skipif(not HAS_TORCH, reason="torch needed")
def test_mnist_nas_rejects_missing_net_class():
    bench = get_benchmark("mnist_nas")
    result = bench.evaluate("x = 1\n")
    assert result.fitness == 0.0
    assert "did not define" in (result.error or "")


@pytest.mark.skipif(not HAS_TORCH, reason="torch needed")
def test_mnist_nas_rejects_syntax_error():
    bench = get_benchmark("mnist_nas")
    result = bench.evaluate("class Net(nn.Module:\n")  # syntax error
    assert result.fitness == 0.0
    assert result.error is not None
