"""Anthropic provider with prompt caching.

The system prompt is large and stable across mutations within a run, so we
mark it with cache_control to cut tokens-billed drastically. Per
https://docs.claude.com/en/docs/build-with-claude/prompt-caching the
ephemeral 5-minute cache is free to use and ideal for back-to-back
mutations in an evolution loop.

API key is read from $ANTHROPIC_API_KEY by the SDK.
"""

from __future__ import annotations

import os

from science_game.llm.base import LLMProvider, MutationRequest, MutationResponse
from science_game.llm.ollama_provider import extract_code

DEFAULT_MODEL = "claude-opus-4-7"
DEFAULT_SYSTEM = (
    "You are an evolutionary code mutator. Given a Python program and a "
    "benchmark description, you produce an improved variant. Output ONLY a "
    "single fenced Python code block containing the full new program — no "
    "prose, no explanation, no surrounding text."
)


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None) -> None:
        try:
            import anthropic  # noqa: F401 — checked at construction
        except ImportError as e:
            raise ImportError(
                "anthropic SDK not installed. Run: uv sync --extra api-llm"
            ) from e

        from anthropic import Anthropic

        self.model = model
        self._client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    def mutate(self, request: MutationRequest) -> MutationResponse:
        system_text = request.system_prompt or DEFAULT_SYSTEM
        user_text = request.user_prompt or self._default_user_prompt(request)

        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                system=[
                    {
                        "type": "text",
                        "text": system_text,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": user_text}],
            )
        except Exception as e:
            return MutationResponse(
                child_code="", raw_text="", provider=self.name, model=self.model,
                meta={"error": f"transport: {e!r}"},
            )

        raw = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )
        usage = response.usage
        return MutationResponse(
            child_code=extract_code(raw),
            raw_text=raw,
            provider=self.name,
            model=self.model,
            tokens_in=getattr(usage, "input_tokens", 0),
            tokens_out=getattr(usage, "output_tokens", 0),
            meta={
                "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", 0),
                "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", 0),
                "stop_reason": response.stop_reason,
            },
        )

    @staticmethod
    def _default_user_prompt(request: MutationRequest) -> str:
        return (
            f"Task: {request.task_description}\n"
            f"Benchmark: {request.benchmark}\n\n"
            "Below is the current best program. Produce an improved variant.\n\n"
            "```python\n"
            f"{request.parent_code}\n"
            "```\n"
        )
