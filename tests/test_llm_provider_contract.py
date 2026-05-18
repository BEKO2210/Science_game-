"""Contract tests for the LLM provider interface.

These don't hit a real LLM — they check the interface and the code-extraction
helper. The real Ollama smoke test lives in scripts/run_local.sh and is run
on the user's machine where ollama is available.
"""

from __future__ import annotations

from science_game.llm import LLMProvider, MutationRequest, MutationResponse
from science_game.llm.ollama_provider import OllamaProvider, extract_code


def test_extract_code_python_fence():
    text = "Sure! Here you go:\n\n```python\ndef f():\n    return 1\n```\n\nDone."
    assert extract_code(text) == "def f():\n    return 1"


def test_extract_code_bare_fence():
    text = "```\nprint('hi')\n```"
    assert extract_code(text) == "print('hi')"


def test_extract_code_no_fence_returns_raw():
    text = "def f():\n    return 1\n"
    assert extract_code(text) == "def f():\n    return 1"


def test_ollama_health_check_offline():
    """When ollama isn't running, health_check returns False, not raises."""
    p = OllamaProvider(base_url="http://127.0.0.1:1")  # bogus port
    assert p.health_check() is False


def test_ollama_mutate_offline_returns_error_response():
    """Transport failures must be reported via MutationResponse.meta, not raised."""
    p = OllamaProvider(base_url="http://127.0.0.1:1", timeout=1.0)
    req = MutationRequest(
        parent_code="x = 1",
        task_description="noop",
        benchmark="test",
    )
    resp = p.mutate(req)
    assert resp.child_code == ""
    assert "error" in resp.meta


class FakeProvider(LLMProvider):
    name = "fake"

    def mutate(self, request: MutationRequest) -> MutationResponse:
        return MutationResponse(
            child_code="def f():\n    return 2",
            raw_text="```python\ndef f():\n    return 2\n```",
            provider=self.name,
            model="fake-1",
            tokens_in=10,
            tokens_out=8,
        )


def test_provider_protocol_smoke():
    p = FakeProvider()
    resp = p.mutate(MutationRequest(parent_code="x=1", task_description="t", benchmark="b"))
    assert resp.child_code.startswith("def f")
    assert resp.provider == "fake"
