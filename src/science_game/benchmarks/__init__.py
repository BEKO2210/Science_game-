from science_game.benchmarks.base import (
    Benchmark,
    EvalResult,
    register_benchmark,
    get_benchmark,
    list_benchmarks,
)
from science_game.benchmarks import matmul  # noqa: F401 — registers

__all__ = [
    "Benchmark",
    "EvalResult",
    "register_benchmark",
    "get_benchmark",
    "list_benchmarks",
]
