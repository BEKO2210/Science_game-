from __future__ import annotations

import json
from pathlib import Path

from science_game.publish.manifest import Manifest, write_manifest


def test_manifest_create_and_write(tmp_path: Path):
    m = Manifest.create("run-abc", config={"benchmark": "matmul", "seed": 42})
    out = tmp_path / "manifest.json"
    write_manifest(m, out)

    data = json.loads(out.read_text())
    assert data["run_id"] == "run-abc"
    assert data["config"]["benchmark"] == "matmul"
    assert data["python_version"].startswith("3.")
    assert "platform" in data
