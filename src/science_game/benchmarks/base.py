from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class EvalResult:
    """Result of running a single program through the benchmark evaluator."""

    fitness: float
    correct: bool
    metrics: dict = field(default_factory=dict)
    error: str | None = None


class Benchmark(ABC):
    """A benchmark exposes a seed program and an evaluator. Engine drives mutation
    of `seed_program`; each candidate gets scored via `evaluate`."""

    name: str = "base"
    task_description: str = ""

    @property
    @abstractmethod
    def seed_program(self) -> str:
        """Initial Python source code the engine starts mutating from."""

    @abstractmethod
    def evaluate(self, program_code: str) -> EvalResult:
        """Run a candidate program and return its fitness + metrics."""


_REGISTRY: dict[str, Callable[[], Benchmark]] = {}


def register_benchmark(name: str) -> Callable[[type[Benchmark]], type[Benchmark]]:
    def deco(cls: type[Benchmark]) -> type[Benchmark]:
        _REGISTRY[name] = cls
        cls.name = name
        return cls

    return deco


def get_benchmark(name: str) -> Benchmark:
    if name not in _REGISTRY:
        raise KeyError(f"unknown benchmark: {name!r}. known: {sorted(_REGISTRY)}")
    return _REGISTRY[name]()


def list_benchmarks() -> list[str]:
    return sorted(_REGISTRY)
