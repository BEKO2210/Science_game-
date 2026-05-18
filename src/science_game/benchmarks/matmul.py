"""Matmul benchmark.

Goal: evolve a 2x2 matrix-multiplication function that uses as few scalar
multiplications as possible while remaining correct. The naive algorithm uses 8;
Strassen's famous 1969 result uses 7. A successful run finds a 7-mult variant
(the AlphaEvolve-style sanity check, but in miniature).

Scoring:
    fitness = correctness * (1 / mult_count)
    correctness = fraction of test cases where the candidate matches numpy's @ operator
                  within an absolute tolerance.

Mult counting: we wrap Python's `*` for the duration of evaluation via a Counter
class that the candidate is REQUIRED to use for all scalar multiplications. The
seed program demonstrates the pattern.

Safety: the candidate is executed in a restricted namespace with no builtins
beyond a tiny whitelist. This is NOT a real sandbox — when running outside the
local trusted dev loop, wrap evaluation in firejail/nsjail.
"""

from __future__ import annotations

import math
import textwrap
import traceback
from typing import Any

import numpy as np

from science_game.benchmarks.base import Benchmark, EvalResult, register_benchmark

SEED_PROGRAM = textwrap.dedent(
    '''
    """Naive 2x2 matrix multiplication using 8 scalar multiplications.

    The Counter parameter is a callable: every scalar multiplication MUST go
    through `mul(a, b)` instead of `a * b`. The evaluator uses this to count
    multiplications. Adders/subtracts are free.

    Returns the 2x2 product as a list of lists.
    """

    def matmul2x2(A, B, mul):
        a, b, c, d = A[0][0], A[0][1], A[1][0], A[1][1]
        e, f, g, h = B[0][0], B[0][1], B[1][0], B[1][1]

        c00 = mul(a, e) + mul(b, g)
        c01 = mul(a, f) + mul(b, h)
        c10 = mul(c, e) + mul(d, g)
        c11 = mul(c, f) + mul(d, h)

        return [[c00, c01], [c10, c11]]
    '''
).strip()


SAFE_BUILTINS = {
    "len": len,
    "range": range,
    "abs": abs,
    "min": min,
    "max": max,
    "sum": sum,
    "float": float,
    "int": int,
    "list": list,
    "tuple": tuple,
    "enumerate": enumerate,
    "zip": zip,
}


class MultCounter:
    __slots__ = ("count",)

    def __init__(self) -> None:
        self.count = 0

    def __call__(self, a: float, b: float) -> float:
        self.count += 1
        return a * b


@register_benchmark("matmul")
class Matmul2x2Benchmark(Benchmark):
    task_description = (
        "Evolve a Python function matmul2x2(A, B, mul) that multiplies two 2x2 "
        "matrices using as few scalar multiplications as possible. Every scalar "
        "multiplication must go through the provided `mul` callable. Addition and "
        "subtraction are free. The naive algorithm uses 8 multiplications; Strassen "
        "showed in 1969 that 7 suffice. Beat 8."
    )

    def __init__(self, num_cases: int = 500, tol: float = 1e-6, seed: int = 0) -> None:
        self.num_cases = num_cases
        self.tol = tol
        self.seed = seed

    @property
    def seed_program(self) -> str:
        return SEED_PROGRAM

    def evaluate(self, program_code: str) -> EvalResult:
        # 1. Compile the candidate into an isolated namespace.
        namespace: dict[str, Any] = {"__builtins__": SAFE_BUILTINS}
        try:
            exec(compile(program_code, "<candidate>", "exec"), namespace)
        except Exception as e:
            return EvalResult(
                fitness=0.0,
                correct=False,
                error=f"compile/exec: {e!r}",
                metrics={"traceback": traceback.format_exc(limit=3)},
            )

        fn = namespace.get("matmul2x2")
        if not callable(fn):
            return EvalResult(
                fitness=0.0,
                correct=False,
                error="program did not define matmul2x2(A, B, mul)",
            )

        # 2. Run test cases.
        rng = np.random.default_rng(self.seed)
        correct = 0
        total = self.num_cases
        mult_counts: list[int] = []

        for _ in range(total):
            A = rng.standard_normal((2, 2))
            B = rng.standard_normal((2, 2))
            counter = MultCounter()
            try:
                result = fn(A.tolist(), B.tolist(), counter)
            except Exception as e:
                return EvalResult(
                    fitness=0.0,
                    correct=False,
                    error=f"runtime: {e!r}",
                    metrics={"traceback": traceback.format_exc(limit=3)},
                )

            mult_counts.append(counter.count)
            expected = A @ B
            if not self._matches(result, expected):
                continue
            correct += 1

        correctness = correct / total
        avg_mults = float(np.mean(mult_counts)) if mult_counts else math.inf
        # Mult count must be stable across inputs to be meaningful — penalize variance.
        if mult_counts and len(set(mult_counts)) > 1:
            avg_mults = max(mult_counts)  # be pessimistic

        # Hard correctness gate: a matmul algorithm that returns the wrong
        # product is useless no matter how few multiplications it uses.
        # Without this, partial-correct 7-mult variants beat the seed
        # 8-mult algorithm and trap the search in a local optimum.
        if correctness < 1.0 or avg_mults <= 0:
            fitness = 0.0
        else:
            fitness = 1.0 / avg_mults

        return EvalResult(
            fitness=fitness,
            correct=correctness == 1.0,
            metrics={
                "correctness": correctness,
                "mult_count": avg_mults,
                "num_cases": total,
                "beats_naive": correctness == 1.0 and avg_mults < 8,
                "beats_strassen": correctness == 1.0 and avg_mults < 7,
            },
        )

    def _matches(self, candidate: Any, expected: np.ndarray) -> bool:
        try:
            arr = np.asarray(candidate, dtype=float)
        except (TypeError, ValueError):
            return False
        if arr.shape != expected.shape:
            return False
        return bool(np.allclose(arr, expected, atol=self.tol))
