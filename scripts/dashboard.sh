#!/usr/bin/env bash
# Launch the Streamlit dashboard (Sprint 2 — placeholder).
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -f "src/science_game/dashboard/app.py" ]; then
  echo "Dashboard not built yet — coming in Sprint 2."
  exit 1
fi

uv run streamlit run src/science_game/dashboard/app.py
