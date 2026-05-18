"""OpenAI provider.

API key is read from $OPENAI_API_KEY by the SDK. Implicit prompt caching
applies to repeated prefixes automatically (see OpenAI prompt-caching docs);
no special wiring is needed on our side.
"""

from __future__ import annotations

import os

from science_game.llm.base import LLMProvider, MutationRequest, MutationResponse
from science_game.llm.ollama_provider import extract_code

DEFAULT_MODEL = "gpt-4o"
DEFAULT_SYSTEM = (
    "You are an evolutionary code mutator. Given a Python program and a "
    "benchmark description, you produce an improved variant. Output ONLY a "
    "single fenced Python code block containing the full new program — no "
    "prose, no explanation, no surrounding text."
)


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None) -> None:
        try:
            import openai  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "openai SDK not installed. Run: uv sync --extra api-llm"
            ) from e

        from openai import OpenAI

        self.model = model
        self._client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    def mutate(self, request: MutationRequest) -> MutationResponse:
        system_text = request.system_prompt or DEFAULT_SYSTEM
        user_text = request.user_prompt or self._default_user_prompt(request)

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                seed=request.seed,
                messages=[
                    {"role": "system", "content": system_text},
                    {"role": "user", "content": user_text},
                ],
            )
        except Exception as e:
            return MutationResponse(
                child_code="", raw_text="", provider=self.name, model=self.model,
                meta={"error": f"transport: {e!r}"},
            )

        choice = response.choices[0]
        raw = choice.message.content or ""
        usage = response.usage
        return MutationResponse(
            child_code=extract_code(raw),
            raw_text=raw,
            provider=self.name,
            model=self.model,
            tokens_in=usage.prompt_tokens if usage else 0,
            tokens_out=usage.completion_tokens if usage else 0,
            meta={
                "cached_tokens": (
                    usage.prompt_tokens_details.cached_tokens
                    if usage and getattr(usage, "prompt_tokens_details", None)
                    else 0
                ),
                "finish_reason": choice.finish_reason,
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
