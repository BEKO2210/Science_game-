# Science Game — Algorithm Forge

> LLM-driven evolutionary search for code, models, and algorithms — with a research dashboard, GPU acceleration, and a full publish-ready pipeline.

**Status:** v0.0.1 — Walking Skeleton (Sprint 1)

Inspiration: DeepMind's [AlphaEvolve](https://deepmind.google/blog/alphevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/) (May 2025) used LLM-driven evolution to beat Strassen's 56-year-old record for 4×4 matrix multiplication. This project builds on [OpenEvolve](https://github.com/codelion/openevolve) — the open-source reimplementation — and adds a research-grade dashboard, hybrid local/API LLM mutation, reproducibility infrastructure, and a publish pipeline targeting Hugging Face Hub.

## Why

Evolution as a research tool is having a moment. Sakana AI's evolutionary model merging, OpenEvolve's AlphaEvolve replication, and the broader "AI evolves AI" wave all point at the same thing: small, well-instrumented evolutionary loops on accessible hardware can produce genuine scientific contributions. This repo tries to make that loop **playable** — you start a run from a dashboard, watch code mutate live, see fitness climb, and publish the result.

Phase 4 closes the loop: the mutation logs we accumulate become a fine-tuning dataset for a specialized **mutator LLM**, trained on a free Colab T4 via [Unsloth](https://github.com/unslothai/unsloth) and exported back into Ollama as a custom-tuned engine. AI evolves AI, literally.

## Quickstart (planned)

```bash
# clone with submodules
git clone --recursive https://github.com/beko2210/science_game-.git
cd science_game-

# install
uv sync
./scripts/bootstrap.sh    # pulls ollama models if local LLM mode

# run a 5-generation evolution on the matmul benchmark
./scripts/run_local.sh matmul ollama-qwen --generations 5

# open the dashboard
streamlit run src/science_game/dashboard/app.py
```

## Status

| Sprint | Scope | State |
|---|---|---|
| 1 | Walking skeleton: pyproject, OpenEvolve vendor, Ollama provider, matmul benchmark, smoke tests | **done** |
| 2 | Streamlit dashboard (Launcher / Live / Artifacts / Compare), DVC stub, live plots | **done** |
| 3 | MNIST NAS benchmark, HF Hub publisher, Anthropic + OpenAI providers (prompt caching), Quarto paper template | **done** |
| 4 | Reproducibility polish (`--from-manifest`), CI, first published demo runs, `v0.1.0` tag | pending |
| Phase 4 | Self-Improving Mutator via Unsloth (Colab) → fine-tuned Ollama model | post-MVP |

### Dashboard

```bash
uv sync --extra dev --extra dashboard
./scripts/dashboard.sh   # → http://localhost:8501
```

4 Seiten: **Home** (Run-Übersicht), **Launcher** (neuen Run starten — spawnt einen `science-game run`-Subprocess), **Live** (auto-refreshing Fitness-Plot + letzte Mutation + Best-of-Generation-Snapshots), **Compare** (mehrere Runs auf einem Plot). Auf der **Artifacts**-Seite kann jeder Run mit einem Klick als HF-Hub-Repo veröffentlicht werden.

### LLM-Provider

```bash
# Lokal mit Ollama (Default)
ollama pull qwen2.5-coder:7b
uv run science-game run matmul --provider ollama-qwen -g 20

# Anthropic (Prompt-Caching aktiv — drastisch günstiger bei vielen Generationen)
export ANTHROPIC_API_KEY=sk-ant-...
uv sync --extra api-llm
uv run science-game run matmul --provider anthropic -g 20

# OpenAI (implizites Prompt-Caching)
export OPENAI_API_KEY=sk-...
uv run science-game run matmul --provider openai -g 20
```

### Benchmarks

- **`matmul`** — 2×2-Matrix-Multiplikation mit minimaler Multiplikationszahl. Strassen-Sanity-Check (8 naiv → 7 evolved).
- **`mnist_nas`** — Mini-NAS auf MNIST-Subset (5k Samples, 1 Epoche). Fitness = `accuracy − 0.05·log₁₀(params)`. Braucht `uv sync --extra nas` (torch+torchvision).

### Publishing

```bash
hf auth login   # einmalig
# Dashboard → Artifacts → Publish-Button
# oder programmatisch:
uv run python -c "from science_game.publish import upload_run; \
    upload_run('runs/matmul-xxxxxx')"
```

Jeder Run wird als HF-Repo angelegt mit Manifest, Events, Mutationen (Phase-4-Trainingsdaten) und Best-of-Generation-Code.

Full design doc: see the plan file referenced in the project root, or `docs/plan.md` once mirrored in.

## License

Apache-2.0. Vendored OpenEvolve retains its own Apache-2.0 license under `third_party/openevolve`.
