#!/usr/bin/env bash
# Bootstrap: install dependencies and pull the default Ollama model.
# Idempotent — safe to re-run.

set -euo pipefail

cd "$(dirname "$0")/.."

echo "[1/3] uv sync (Python deps)..."
uv sync --extra dashboard --extra api-llm --extra dev

echo "[2/3] check for ollama..."
if ! command -v ollama >/dev/null 2>&1; then
  echo "  WARN: ollama not installed. Install from https://ollama.com to use the local LLM provider."
  echo "  (You can still use --provider anthropic or --provider openai.)"
else
  echo "[3/3] ollama pull qwen2.5-coder:7b..."
  ollama pull qwen2.5-coder:7b
fi

echo
echo "Done. Try:"
echo "  uv run pytest"
echo "  uv run science-game list-benchmarks"
echo "  uv run science-game run matmul --provider ollama-qwen --generations 5"
