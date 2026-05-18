"""Anthropic / OpenAI providers do lazy SDK imports. Verify behaviour
without the SDKs installed (transport-error reporting tested separately
on the user PC where keys live)."""

from __future__ import annotations

import importlib.util

import pytest

HAS_ANTHROPIC = importlib.util.find_spec("anthropic") is not None
HAS_OPENAI = importlib.util.find_spec("openai") is not None


@pytest.mark.skipif(HAS_ANTHROPIC, reason="anthropic IS installed; this checks the without-SDK path")
def test_anthropic_provider_clean_error_without_sdk():
    from science_game.llm.anthropic_provider import AnthropicProvider
    with pytest.raises(ImportError) as exc:
        AnthropicProvider()
    assert "uv sync --extra api-llm" in str(exc.value)


@pytest.mark.skipif(HAS_OPENAI, reason="openai IS installed; this checks the without-SDK path")
def test_openai_provider_clean_error_without_sdk():
    from science_game.llm.openai_provider import OpenAIProvider
    with pytest.raises(ImportError) as exc:
        OpenAIProvider()
    assert "uv sync --extra api-llm" in str(exc.value)


def test_get_provider_resolves_known_names():
    from science_game.llm import get_provider

    p = get_provider("ollama-qwen")
    assert p.name == "ollama"
    with pytest.raises(ValueError):
        get_provider("does-not-exist")
