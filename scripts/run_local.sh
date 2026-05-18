#!/usr/bin/env bash
# Convenience wrapper for science-game run.
# Usage: ./scripts/run_local.sh [benchmark] [provider] [generations]
set -euo pipefail

BENCHMARK="${1:-matmul}"
PROVIDER="${2:-ollama-qwen}"
GENERATIONS="${3:-5}"

cd "$(dirname "$0")/.."

uv run science-game run "$BENCHMARK" \
  --provider "$PROVIDER" \
  --generations "$GENERATIONS"
