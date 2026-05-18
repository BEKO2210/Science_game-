#!/usr/bin/env bash
# Bootstrap: init submodules, install dependencies, pull the default Ollama model.
# Idempotent — safe to re-run.

set -euo pipefail

cd "$(dirname "$0")/.."

echo "[1/4] git submodule update (third_party/openevolve)..."
if [ -d .git ]; then
  git submodule update --init --recursive
else
  echo "  (not a git checkout — skipping submodule init)"
fi

echo "[2/4] uv sync (Python deps)..."
uv sync --extra dashboard --extra api-llm --extra dev

echo "[3/4] check for ollama..."
if ! command -v ollama >/dev/null 2>&1; then
  echo "  WARN: ollama not installed. Install from https://ollama.com to use the local LLM provider."
  echo "  (You can still use --provider anthropic, --provider openai, or --provider mock.)"
else
  echo "[4/4] ollama pull qwen2.5-coder:7b..."
  ollama pull qwen2.5-coder:7b
fi

echo
echo "Done. Verify the install:"
echo "  uv run science-game doctor"
echo
echo "Then try a smoke run (works without Ollama):"
echo "  uv run science-game run sort --provider mock --generations 1"
echo
echo "And the real thing (needs Ollama running):"
echo "  uv run science-game run matmul --provider ollama-qwen --generations 5"
