"""Tests for the Modelfile generator."""

from __future__ import annotations

from pathlib import Path

from science_game.phase4.modelfile import build_modelfile, write_modelfile


def test_build_modelfile_basic():
    text = build_modelfile(gguf_path="./forge.gguf")
    assert text.startswith("FROM ./forge.gguf")
    assert "PARAMETER temperature 0.8" in text
    assert "PARAMETER num_ctx 8192" in text
    assert "SYSTEM" in text


def test_build_modelfile_custom_overrides():
    text = build_modelfile(
        gguf_path="./x.gguf", temperature=0.3, num_ctx=4096,
        parameter_overrides={"top_p": "0.9", "repeat_penalty": "1.1"},
    )
    assert "PARAMETER temperature 0.3" in text
    assert "PARAMETER num_ctx 4096" in text
    assert "PARAMETER top_p 0.9" in text
    assert "PARAMETER repeat_penalty 1.1" in text


def test_write_modelfile_creates_parent(tmp_path: Path):
    out = tmp_path / "nested" / "Modelfile"
    write_modelfile(out, gguf_path="./y.gguf")
    assert out.exists()
    assert out.read_text().startswith("FROM ./y.gguf")
