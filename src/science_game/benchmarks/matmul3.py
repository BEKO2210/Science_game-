"""3x3 matrix-multiplication benchmark.

Naive triple loop uses 27 scalar multiplications. The best known result
for 3x3 over a commutative ring is 23 (Laderman, 1976). Anything below
23 would be a novel result. Smirnov found 21 for 3x3 over C; over the
reals the lower bound is still open.

Same `mul` counter mechanism as the 2x2 benchmark. Hard correctness gate
because partial-correct 23-mult solutions would otherwise dominate the
correct 27-mult seed.
"""

from __future__ import annotations

import math
import textwrap
import traceback
from typing import Any

import numpy as np

from science_game.benchmarks.base import Benchmark, EvalResult, register_benchmark
from science_game.benchmarks.matmul import SAFE_BUILTINS, MultCounter

SEED_PROGRAM = textwrap.dedent(
    '''
    """Naive 3x3 matrix multiplication using 27 scalar multiplications.

    Every scalar multiplication MUST go through `mul(a, b)` instead of `a * b`
    so the evaluator can count them. Addition and subtraction are free.

    Returns the 3x3 product as a list of lists.
    """

    def matmul3x3(A, B, mul):
        C = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        for i in range(3):
            for j in range(3):
                s = mul(A[i][0], B[0][j])
                s = s + mul(A[i][1], B[1][j])
                s = s + mul(A[i][2], B[2][j])
                C[i][j] = s
        return C
    '''
).strip()


@register_benchmark("matmul3")
class Matmul3x3Benchmark(Benchmark):
    task_description = (
        "Evolve a Python function matmul3x3(A, B, mul) that multiplies two 3x3 "
        "matrices using as few scalar multiplications as possible. Every scalar "
        "multiplication must go through the provided `mul` callable. Addition and "
        "subtraction are free. The naive algorithm uses 27 multiplications; "
        "Laderman (1976) showed 23 suffice. Beat 27, ideally tie or beat 23."
    )

    def __init__(self, num_cases: int = 500, tol: float = 1e-6, seed: int = 0) -> None:
        self.num_cases = num_cases
        self.tol = tol
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
                fitness=0.0,
                correct=False,
                error=f"compile/exec: {e!r}",
                metrics={"traceback": traceback.format_exc(limit=3)},
            )

        fn = namespace.get("matmul3x3")
        if not callable(fn):
            return EvalResult(
                fitness=0.0,
                correct=False,
                error="program did not define matmul3x3(A, B, mul)",
            )

        rng = np.random.default_rng(self.seed)
        correct = 0
        total = self.num_cases
        mult_counts: list[int] = []

        for _ in range(total):
            A = rng.standard_normal((3, 3))
            B = rng.standard_normal((3, 3))
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
        if mult_counts and len(set(mult_counts)) > 1:
            avg_mults = max(mult_counts)

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
                "beats_naive": correctness == 1.0 and avg_mults < 27,
                "matches_laderman": correctness == 1.0 and avg_mults <= 23,
                "beats_laderman": correctness == 1.0 and avg_mults < 23,
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
