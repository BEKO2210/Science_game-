from __future__ import annotations

import re

import httpx

from science_game.llm.base import LLMProvider, MutationRequest, MutationResponse

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5-coder:7b"

CODE_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


def extract_code(text: str) -> str:
    """Pull the first fenced code block; fall back to the full text."""
    match = CODE_FENCE_RE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_OLLAMA_URL,
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def health_check(self) -> bool:
        try:
            r = httpx.get(f"{self.base_url}/api/tags", timeout=5.0)
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    def mutate(self, request: MutationRequest) -> MutationResponse:
        prompt = self._build_prompt(request)
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_tokens,
            },
        }
        if request.seed is not None:
            payload["options"]["seed"] = request.seed
        if request.system_prompt:
            payload["system"] = request.system_prompt

        try:
            r = httpx.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
            )
            r.raise_for_status()
            data = r.json()
        except httpx.HTTPError as e:
            return MutationResponse(
                child_code="",
                raw_text="",
                provider=self.name,
                model=self.model,
                meta={"error": f"transport: {e!r}"},
            )

        raw = data.get("response", "")
        return MutationResponse(
            child_code=extract_code(raw),
            raw_text=raw,
            provider=self.name,
            model=self.model,
            tokens_in=data.get("prompt_eval_count", 0),
            tokens_out=data.get("eval_count", 0),
            meta={
                "total_duration_ns": data.get("total_duration", 0),
                "eval_duration_ns": data.get("eval_duration", 0),
            },
        )

    def _build_prompt(self, request: MutationRequest) -> str:
        if request.user_prompt:
            return request.user_prompt
        return (
            f"Task: {request.task_description}\n"
            f"Benchmark: {request.benchmark}\n\n"
            "Below is the current best program. Produce an improved variant. "
            "Return ONLY a single fenced Python code block with the full new program.\n\n"
            "```python\n"
            f"{request.parent_code}\n"
            "```\n"
        )
