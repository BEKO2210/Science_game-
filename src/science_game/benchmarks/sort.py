"""Sorting-network benchmark.

Goal: evolve a fixed sequence of compare-and-swap operations that sorts any
length-N input list (default N=8). The candidate exposes a single function
`build_network()` returning a list of `(i, j)` pairs; the evaluator runs every
pair as `if a[i] > a[j]: swap(a[i], a[j])` against a battery of random and
adversarial inputs, then scores:

    fitness = correctness * (1 / max(comparators, 1))

Reference best for N=8: 19 comparators (Knuth, TAOCP vol 3). The seed
program is the textbook bubble-sort network with N*(N-1)/2 = 28
comparators — beating it down toward 19 is the AlphaEvolve-style success
criterion.

Compared to matmul, this benchmark:
- has many more correct solutions (the search space is forgiving)
- evaluates in milliseconds per candidate
- is great for fast iteration loops and dataset-bootstrap for Phase 4
"""

from __future__ import annotations

import textwrap
import traceback
from typing import Any

import numpy as np

from science_game.benchmarks.base import Benchmark, EvalResult, register_benchmark

DEFAULT_N = 8

SEED_PROGRAM = textwrap.dedent(
    """
    \"\"\"All-pairs (insertion-sort-style) network: 28 comparators for N=8.

    `build_network()` must return a list of (i, j) integer pairs with
    0 <= i < j < N. The evaluator applies them in order as
    `if a[i] > a[j]: swap`. Find a shorter sequence that still sorts
    every input correctly. The known optimum for N=8 is 19 (Knuth, TAOCP
    vol 3, problem 5.3.4-44).
    \"\"\"

    def build_network():
        N = 8
        pairs = []
        for i in range(N):
            for j in range(i + 1, N):
                pairs.append((i, j))
        return pairs
    """
).strip()


SAFE_BUILTINS = {
    "len": len, "range": range, "abs": abs, "min": min, "max": max,
    "sum": sum, "float": float, "int": int, "list": list, "tuple": tuple,
    "enumerate": enumerate, "zip": zip, "sorted": sorted, "any": any, "all": all,
}


def apply_network(values: list[int], pairs: list[tuple[int, int]]) -> list[int]:
    out = list(values)
    for i, j in pairs:
        if out[i] > out[j]:
            out[i], out[j] = out[j], out[i]
    return out


@register_benchmark("sort")
class SortingNetworkBenchmark(Benchmark):
    task_description = (
        f"Evolve a Python function build_network() that returns a list of "
        f"(i, j) tuples encoding a sorting network for length-{DEFAULT_N} "
        "inputs. Every pair runs as `if a[i] > a[j]: swap`. The network "
        "must sort EVERY permutation correctly. Minimize the number of "
        "pairs (comparators). The textbook bubble-sort uses 28; the known "
        "optimum for N=8 is 19."
    )

    def __init__(
        self,
        n: int = DEFAULT_N,
        num_random_cases: int = 256,
        seed: int = 0,
    ) -> None:
        self.n = n
        self.num_random_cases = num_random_cases
        self.seed = seed

    @property
    def seed_program(self) -> str:
        return SEED_PROGRAM

    def evaluate(self, program_code: str) -> EvalResult:
        namespace: dict[str, Any] = {"__builtins__": SAFE_BUILTINS}
        try:
            exec(compile(program_code, "<candidate>", "exec"), namespace)
        except Exception as e:
            return EvalResult(
                fitness=0.0, correct=False,
                error=f"compile/exec: {e!r}",
                metrics={"traceback": traceback.format_exc(limit=3)},
            )

        builder = namespace.get("build_network")
        if not callable(builder):
            return EvalResult(
                fitness=0.0, correct=False,
                error="program did not define build_network()",
            )

        try:
            pairs = builder()
        except Exception as e:
            return EvalResult(
                fitness=0.0, correct=False,
                error=f"build_network() raised: {e!r}",
            )

        if not isinstance(pairs, list):
            return EvalResult(
                fitness=0.0, correct=False,
                error="build_network() must return a list",
            )

        # Validate every pair.
        for pair in pairs:
            if (
                not isinstance(pair, tuple) or len(pair) != 2
                or not all(isinstance(x, int) for x in pair)
                or pair[0] < 0 or pair[1] >= self.n or pair[0] >= pair[1]
            ):
                return EvalResult(
                    fitness=0.0, correct=False,
                    error=f"invalid pair {pair!r}; need (i, j) with 0 <= i < j < {self.n}",
                )

        # Test on the "zero-one principle" inputs (sufficient to prove sorting-
        # network correctness): all 2^N binary vectors. Plus random ints.
        # For N up to 12 this is cheap (4096 inputs).
        cases = []
        for mask in range(1 << self.n):
            cases.append([(mask >> bit) & 1 for bit in range(self.n)])

        rng = np.random.default_rng(self.seed)
        for _ in range(self.num_random_cases):
            cases.append(rng.integers(-100, 100, self.n).tolist())

        correct = 0
        for case in cases:
            if apply_network(case, pairs) == sorted(case):
                correct += 1
        correctness = correct / len(cases)

        comparators = len(pairs)
        # Hard correctness gate: a sorting network that doesn't sort every
        # input is worthless, no matter how few comparators it uses. Without
        # this, the optimizer happily accepts partial-correctness 19-comparator
        # networks (fitness 0.70/19 > 1.0/28) and then gets stuck there because
        # the LLM keeps reproducing the same buggy variant.
        if correctness < 1.0:
            fitness = 0.0
        else:
            fitness = 1.0 / max(comparators, 1)

        return EvalResult(
            fitness=fitness,
            correct=correctness == 1.0,
            metrics={
                "correctness": correctness,
                "comparators": comparators,
                "n": self.n,
                "num_cases": len(cases),
                "beats_bubble": correctness == 1.0 and comparators < 28,
                "matches_knuth_optimum": correctness == 1.0 and comparators <= 19,
            },
        )
