"""Run manifest — captures everything needed to reproduce a run."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from science_game import __version__


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None


def _git_dirty() -> bool:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True, capture_output=True, text=True, timeout=5,
        )
        return bool(out.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False


@dataclass
class Manifest:
    run_id: str
    created_at: str
    science_game_version: str
    python_version: str
    platform: str
    git_sha: str | None
    git_dirty: bool
    config: dict = field(default_factory=dict)

    @classmethod
    def create(cls, run_id: str, config: dict) -> Manifest:
        return cls(
            run_id=run_id,
            created_at=datetime.now(UTC).isoformat(),
            science_game_version=__version__,
            python_version=sys.version.split()[0],
            platform=f"{platform.system()} {platform.release()} ({platform.machine()})",
            git_sha=_git_sha(),
            git_dirty=_git_dirty(),
            config=config,
        )


def write_manifest(manifest: Manifest, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(manifest), indent=2, default=str))
