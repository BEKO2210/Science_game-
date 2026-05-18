"""Tests for the OpenEvolve adapter.

OpenEvolve is an optional extra (`uv sync --extra engine-openevolve`). These
tests cover both: with-and-without-OpenEvolve install paths. The pure
helpers (marker wrapping, LLM payload mapping, evaluator factory) work
without OpenEvolve. The full-loop test is skipped unless OpenEvolve is
installed AND an OpenAI-compatible LLM is reachable (CI doesn't have that).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from science_game.engine import EvolutionConfig
from science_game.openevolve_adapter import (
    EVOLVE_END,
    EVOLVE_START,
    _make_openai_compatible_llm,
    make_evaluator,
    wrap_seed_with_markers,
)

HAS_OPENEVOLVE = importlib.util.find_spec("openevolve") is not None


def test_wrap_seed_with_markers_basic():
    out = wrap_seed_with_markers("def f(): pass")
    assert out.startswith(EVOLVE_START)
    assert out.rstrip().endswith(EVOLVE_END)
    assert "def f(): pass" in out


def test_wrap_seed_is_idempotent():
    once = wrap_seed_with_markers("def f(): pass")
    twice = wrap_seed_with_markers(once)
    assert once == twice


def test_make_evaluator_returns_score_dict(tmp_path: Path):
    """The OpenEvolve evaluator contract: callable returning dict with 'score'."""
    eval_fn = make_evaluator("matmul")
    bench_seed = (tmp_path / "candidate.py")
    bench_seed.write_text(
        "def matmul2x2(A, B, mul):\n"
        "    return [[mul(A[0][0], B[0][0]), 0], [0, 0]]\n"  # wrong but compiles
    )
    out = eval_fn(str(bench_seed))
    assert "score" in out
    assert "correct" in out
    assert isinstance(out["score"], float)


def test_llm_payload_ollama():
    cfg = EvolutionConfig(benchmark="matmul", llm_provider="ollama-qwen")
    payload = _make_openai_compatible_llm(cfg)
    assert payload["api_base"] == "http://localhost:11434/v1"
    assert payload["name"] == "qwen2.5-coder:7b"


def test_llm_payload_custom_ollama_model():
    cfg = EvolutionConfig(benchmark="matmul", llm_provider="ollama", llm_model="codellama:13b")
    payload = _make_openai_compatible_llm(cfg)
    assert payload["name"] == "codellama:13b"


def test_llm_payload_openai_default():
    cfg = EvolutionConfig(benchmark="matmul", llm_provider="openai")
    payload = _make_openai_compatible_llm(cfg)
    assert payload["api_base"] == "https://api.openai.com/v1"
    assert payload["name"] == "gpt-4o"


def test_llm_payload_anthropic_rejected():
    cfg = EvolutionConfig(benchmark="matmul", llm_provider="anthropic")
    with pytest.raises(ValueError, match="not yet supported"):
        _make_openai_compatible_llm(cfg)


def test_llm_payload_unknown_provider_rejected():
    cfg = EvolutionConfig(benchmark="matmul", llm_provider="banana")
    with pytest.raises(ValueError, match="unknown llm provider"):
        _make_openai_compatible_llm(cfg)


@pytest.mark.skipif(HAS_OPENEVOLVE, reason="this path only fires without openevolve installed")
def test_build_config_without_openevolve_raises():
    from science_game.openevolve_adapter import build_openevolve_config

    cfg = EvolutionConfig(benchmark="matmul")
    with pytest.raises(ImportError, match="uv sync --extra engine-openevolve"):
        build_openevolve_config(cfg)


@pytest.mark.skipif(HAS_OPENEVOLVE, reason="this path only fires without openevolve installed")
def test_run_without_openevolve_raises():
    from science_game.openevolve_adapter import run_with_openevolve

    cfg = EvolutionConfig(benchmark="matmul")
    with pytest.raises(ImportError, match="uv sync --extra engine-openevolve"):
        run_with_openevolve(cfg)


@pytest.mark.skipif(not HAS_OPENEVOLVE, reason="openevolve needed")
def test_build_openevolve_config_smoke():
    """When openevolve IS installed, building a config should work."""
    from science_game.openevolve_adapter import build_openevolve_config

    cfg = EvolutionConfig(benchmark="matmul", llm_provider="ollama-qwen", generations=3)
    oe_cfg = build_openevolve_config(cfg)
    assert oe_cfg.llm.api_base == "http://localhost:11434/v1"
    assert oe_cfg.max_iterations == 3
    assert oe_cfg.random_seed == 42
