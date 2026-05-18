from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class MutationRequest:
    """A request to an LLM mutator: turn `parent_code` into a (hopefully better) variant.

    The benchmark name and task description give the LLM context. The system prompt is
    typically a generic "you are an evolutionary code mutator" header; the user prompt
    is per-mutation. Temperature and seed control sampling.
    """

    parent_code: str
    task_description: str
    benchmark: str
    system_prompt: str = ""
    user_prompt: str = ""
    temperature: float = 0.8
    seed: int | None = None
    max_tokens: int = 2048


@dataclass
class MutationResponse:
    child_code: str
    raw_text: str
    provider: str
    model: str
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    meta: dict = field(default_factory=dict)


class LLMProvider(ABC):
    """Provider interface — every backend implements `mutate`."""

    name: str = "base"

    @abstractmethod
    def mutate(self, request: MutationRequest) -> MutationResponse:
        """Run one mutation. Implementations must not raise on transport errors —
        instead return a MutationResponse with empty `child_code` and the failure
        recorded in `meta['error']`. The engine decides whether to retry."""
        raise NotImplementedError

    def health_check(self) -> bool:
        """Cheap probe — does the backend respond at all? Default: True."""
        return True
