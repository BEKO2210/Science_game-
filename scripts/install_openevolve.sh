#!/usr/bin/env bash
# Install the vendored OpenEvolve into the project's venv as an editable
# dependency. This is split out from `pyproject.toml` because PEP 508 doesn't
# allow relative `file:` references in published metadata — but `pip install -e
# <path>` handles it fine.
#
# Run once after `uv sync`:
#   ./scripts/install_openevolve.sh
#
# Then run with the OpenEvolve engine:
#   uv run science-game run matmul --engine openevolve --provider ollama-qwen -g 50
set -euo pipefail
cd "$(dirname "$0")/.."

git submodule update --init --recursive third_party/openevolve
uv pip install -e third_party/openevolve

echo
echo "OpenEvolve installed. Smoke check:"
uv run python -c "import openevolve; print('openevolve', openevolve.__version__)"
