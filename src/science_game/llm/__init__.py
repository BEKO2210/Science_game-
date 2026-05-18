from science_game.llm.base import LLMProvider, MutationRequest, MutationResponse
from science_game.llm.ollama_provider import OllamaProvider

__all__ = ["LLMProvider", "MutationRequest", "MutationResponse", "OllamaProvider"]


def get_provider(name: str, **kwargs) -> LLMProvider:
    """Factory: resolve a provider by short name."""
    if name in ("ollama", "ollama-qwen"):
        return OllamaProvider(**kwargs)
    if name == "anthropic":
        from science_game.llm.anthropic_provider import AnthropicProvider

        return AnthropicProvider(**kwargs)
    if name == "openai":
        from science_game.llm.openai_provider import OpenAIProvider

        return OpenAIProvider(**kwargs)
    raise ValueError(f"unknown llm provider: {name!r}")
