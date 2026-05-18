"""Preflight checks — `science-game doctor` runs all of these and reports.

Each check returns (ok: bool, summary: str, hint: str | None). They never
raise — the whole point is to give the user a complete diagnostic instead
of dying on the first problem.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CheckResult:
    name: str
    ok: bool
    summary: str
    hint: str | None = None


def check_python_version() -> CheckResult:
    import sys
    v = sys.version_info
    ok = (v.major, v.minor) >= (3, 11)
    return CheckResult(
        name="Python >= 3.11",
        ok=ok,
        summary=f"{v.major}.{v.minor}.{v.micro}",
        hint=None if ok else "Install Python 3.11+; uv handles this with `uv python install 3.11`.",
    )


def check_module(modname: str, extra_hint: str) -> CheckResult:
    found = importlib.util.find_spec(modname) is not None
    return CheckResult(
        name=f"import {modname}",
        ok=found,
        summary="ok" if found else "missing",
        hint=None if found else extra_hint,
    )


def check_dashboard_extras() -> CheckResult:
    have = all(importlib.util.find_spec(m) is not None for m in ("streamlit", "plotly", "pandas"))
    return CheckResult(
        name="dashboard extras",
        ok=have,
        summary="ok" if have else "missing",
        hint=None if have else "Run: uv sync --extra dashboard",
    )


def check_api_extras() -> CheckResult:
    have = all(importlib.util.find_spec(m) is not None for m in ("anthropic", "openai"))
    return CheckResult(
        name="api-llm extras (optional)",
        ok=have,
        summary="ok" if have else "not installed",
        hint=None if have else "Run: uv sync --extra api-llm (only needed for --provider anthropic|openai)",
    )


def check_nas_extras() -> CheckResult:
    have = all(importlib.util.find_spec(m) is not None for m in ("torch", "torchvision"))
    return CheckResult(
        name="nas extras (optional)",
        ok=have,
        summary="ok" if have else "not installed",
        hint=None if have else "Run: uv sync --extra nas (only needed for --benchmark mnist_nas)",
    )


def check_openevolve() -> CheckResult:
    have = importlib.util.find_spec("openevolve") is not None
    return CheckResult(
        name="openevolve engine (optional)",
        ok=have,
        summary="ok" if have else "not installed",
        hint=None if have else "Run: ./scripts/install_openevolve.sh (only needed for --engine openevolve)",
    )


def check_submodule() -> CheckResult:
    """Verify third_party/openevolve was actually checked out."""
    path = Path("third_party/openevolve/openevolve/__init__.py")
    ok = path.exists()
    return CheckResult(
        name="openevolve submodule checked out",
        ok=ok,
        summary="ok" if ok else "empty (git submodule not initialized)",
        hint=None if ok else "Run: git submodule update --init --recursive",
    )


def check_ollama_binary() -> CheckResult:
    found = shutil.which("ollama") is not None
    return CheckResult(
        name="ollama binary on PATH (optional)",
        ok=found,
        summary="ok" if found else "missing",
        hint=None if found else (
            "Install from https://ollama.com (only needed for --provider ollama-qwen)"
        ),
    )


def check_ollama_running(model: str = "qwen2.5-coder:7b") -> CheckResult:
    """Ping the Ollama HTTP API and verify the model is pulled."""
    try:
        import httpx

        url = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        r = httpx.get(f"{url}/api/tags", timeout=3.0)
        if r.status_code != 200:
            return CheckResult(
                name="ollama service reachable (optional)",
                ok=False,
                summary=f"HTTP {r.status_code}",
                hint="Start Ollama (it usually runs as a system service after install).",
            )
        models = [m.get("name", "") for m in r.json().get("models", [])]
        if model not in models:
            return CheckResult(
                name=f"ollama model {model!r} pulled (optional)",
                ok=False,
                summary=f"not pulled (have: {', '.join(models) or 'none'})",
                hint=f"Run: ollama pull {model}",
            )
        return CheckResult(
            name=f"ollama service + model {model!r}",
            ok=True,
            summary="ok",
        )
    except ImportError:
        return CheckResult(
            name="ollama service reachable (optional)", ok=False, summary="httpx missing",
            hint="Run: uv sync",
        )
    except Exception as e:
        return CheckResult(
            name="ollama service reachable (optional)",
            ok=False,
            summary=f"{type(e).__name__}: {e}",
            hint="Make sure Ollama is running (`ollama serve` or system service). Or use --provider mock for testing.",
        )


def check_cuda() -> CheckResult:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            return CheckResult(
                name="NVIDIA GPU detected",
                ok=True,
                summary=out.stdout.strip().splitlines()[0],
            )
    except Exception:
        pass
    return CheckResult(
        name="NVIDIA GPU detected (optional)",
        ok=False,
        summary="no nvidia-smi (CPU-only mode is fine for matmul + sort)",
        hint=None,
    )


def check_runs_dir() -> CheckResult:
    path = Path("runs")
    if not path.exists():
        return CheckResult(name="runs/ directory", ok=True, summary="will be created on first run")
    n = len(list(path.glob("*/manifest.json")))
    return CheckResult(
        name="runs/ directory",
        ok=True,
        summary=f"{n} previous run(s)",
    )


ALL_CHECKS = (
    check_python_version,
    check_module,  # placeholder; we expand inline below
)


def run_all_checks(ollama_model: str = "qwen2.5-coder:7b") -> list[CheckResult]:
    return [
        check_python_version(),
        check_submodule(),
        check_module("science_game", "Run: uv sync"),
        check_module("typer", "Run: uv sync"),
        check_module("rich", "Run: uv sync"),
        check_module("httpx", "Run: uv sync"),
        check_dashboard_extras(),
        check_api_extras(),
        check_nas_extras(),
        check_openevolve(),
        check_ollama_binary(),
        check_ollama_running(ollama_model),
        check_cuda(),
        check_runs_dir(),
    ]
