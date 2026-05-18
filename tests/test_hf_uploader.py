"""Unit tests for the HF uploader helpers — no real HTTP calls."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from science_game.publish.hf_uploader import _build_readme, upload_run


def test_build_readme_contains_key_fields():
    manifest = {
        "run_id": "matmul-abc",
        "config": {
            "benchmark": "matmul", "llm_provider": "ollama-qwen",
            "llm_model": None, "generations": 50, "seed": 42, "temperature": 0.8,
        },
        "git_sha": "deadbeef" * 5,
        "git_dirty": True,
        "python_version": "3.11.15",
        "platform": "Linux 6.18.5 (x86_64)",
        "created_at": "2026-05-18T14:00:00+00:00",
    }
    md = _build_readme(Path("runs/matmul-abc"), manifest)
    assert "matmul-abc" in md
    assert "ollama-qwen" in md
    assert "dirty tree" in md
    assert "mutations.jsonl" in md
    assert "Phase-4" in md


def test_upload_run_errors_without_manifest(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        upload_run(tmp_path)


def test_upload_run_writes_readme_before_upload(tmp_path: Path, monkeypatch):
    """We monkey-patch create_repo + HfApi to verify the local README is generated."""
    run_dir = tmp_path / "matmul-fake"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(json.dumps({
        "run_id": "matmul-fake",
        "config": {"benchmark": "matmul", "llm_provider": "ollama-qwen"},
        "git_sha": "abc", "git_dirty": False,
        "python_version": "3.11.15", "platform": "Linux",
        "created_at": "2026-05-18T00:00:00+00:00",
    }))
    (run_dir / "events.jsonl").write_text("")
    (run_dir / "best").mkdir()
    (run_dir / "best" / "latest.py").write_text("def f(): pass\n")

    calls = {"create": 0, "upload": 0}

    class FakeApi:
        def __init__(self, token=None):
            pass
        def upload_folder(self, folder_path, repo_id, repo_type, ignore_patterns=None):
            calls["upload"] += 1
            assert (Path(folder_path) / "README.md").exists()

    def fake_create_repo(repo_id, repo_type, private, exist_ok, token):
        calls["create"] += 1

    monkeypatch.setattr("huggingface_hub.HfApi", FakeApi)
    monkeypatch.setattr("huggingface_hub.create_repo", fake_create_repo)

    result = upload_run(run_dir, repo_id="Beko2210/test-fake")
    assert result.repo_id == "Beko2210/test-fake"
    assert result.files_uploaded >= 3  # manifest + events + best/latest + README
    assert calls["create"] == 1
    assert calls["upload"] == 1
    assert (run_dir / "README.md").exists()
