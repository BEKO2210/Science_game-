from science_game.benchmarks import (
    matmul,  # noqa: F401 — registers
    mnist_nas,  # noqa: F401 — registers
    sort,  # noqa: F401 — registers
)
from science_game.benchmarks.base import (
    Benchmark,
    EvalResult,
    get_benchmark,
    list_benchmarks,
    register_benchmark,
)

__all__ = [
    "Benchmark",
    "EvalResult",
    "get_benchmark",
    "list_benchmarks",
    "register_benchmark",
]
