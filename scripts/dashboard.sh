#!/usr/bin/env bash
# Launch the Streamlit dashboard.
# Default port 8501; override with: ./scripts/dashboard.sh --port 8888
set -euo pipefail
cd "$(dirname "$0")/.."

uv run streamlit run src/science_game/dashboard/app.py "$@"
