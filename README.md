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
| 4 | Reproducibility (`--from-manifest`), GitHub Actions CI, mutator-dataset builder, Phase-4 docs | **done** |
| Phase 4 | Self-Improving Mutator via Unsloth (Colab) → fine-tuned Ollama model | docs ready, run on user PC |

**MVP scope is complete.** All four sprints shipped, 33 tests passing. See [`docs/phase4-unsloth-colab.md`](docs/phase4-unsloth-colab.md) for the Self-Improving-Mutator playbook.

### Reproducibility

```bash
# Reproduce any previous run from its manifest
uv run science-game run --from-manifest runs/matmul-abc12345/manifest.json
```

### Phase 4 — Schnellste Variante: One-Click Colab

Komplett-Pipeline (Runs generieren → Dataset → Fine-Tuning → A/B-Vergleich) in einem Notebook auf gratis Colab T4:

**Option A — Iterative Self-Improvement (empfohlen, resilient gegen Disconnects):**

`notebooks/phase4_iterative_loop.ipynb` öffnen in Colab via Direkt-URL:

```
https://colab.research.google.com/github/BEKO2210/Science_game-/blob/claude/evolution-game-concept-yN3Tn/notebooks/phase4_iterative_loop.ipynb
```

Diese Variante:
- Persistiert alles in Google Drive (`MyDrive/algorithm-forge/`)
- Kann an jeder Stelle weitermachen falls Colab disconnected
- Trainiert iterativ: V1 lernt von OpenAI, V2 lernt von V1+OpenAI, V3 von V2+V1+OpenAI, …
- Plot am Ende zeigt Fortschritt über Iterationen
- Default: 2 Iterationen ~3 Stunden, ~$1 OpenAI-Kosten

**Option B — Single-Shot (einfacher, kein Drive):**

`notebooks/phase4_full_pipeline.ipynb` — eine Iteration ohne Drive-Persistenz, ~90 min.

Für beide brauchst du in **Colab Secrets**: `HF_TOKEN` (Pflicht), `OPENAI_API_KEY` (für Bootstrap).

Output am Ende: A/B-Report ob deine fine-getunte Mutator-KI die Base-KI messbar schlägt. Dataset + Modell landen automatisch auf deinem HF-Hub als `Beko2210/algorithm-forge-*-vN`.

### Phase 4 — Self-Improving Mutator (manuelle Variante)

```bash
# 1. Aggregate Mutationen aus deinen Runs
uv run science-game build-mutator-dataset \
    --runs-root runs --out datasets/mutator-v1.jsonl --mode sft --min-delta 0.001

# 2. HF Hub push
hf upload-dataset Beko2210/algorithm-forge-mutations-v1 datasets/mutator-v1.jsonl

# 3. Modelfile + Anleitung generieren
uv run science-game phase4 prepare \
    --dataset datasets/mutator-v1.jsonl \
    --gguf-name model-Q4_K_M.gguf

# 4. notebooks/unsloth_finetune.ipynb in Colab öffnen, T4 GPU, Run all
#    → exportiert GGUF und pusht zu Beko2210/algorithm-forge-mutator-qwen-v1

# 5. GGUF lokal laden + Ollama registrieren
hf download Beko2210/algorithm-forge-mutator-qwen-v1 \
    --include '*.gguf' --local-dir ./phase4-out
cd phase4-out && ollama create algorithm-forge-mutator -f Modelfile

# 6. A/B-Compare Base vs. Fine-tuned
uv run science-game phase4 evaluate \
    --benchmark matmul \
    --finetuned-model algorithm-forge-mutator \
    --seeds 0,1,2,3,4 --generations 20
```

Output: `phase4-out/ab-report.json` mit Win/Tie/Loss-Statistik und avg-Fitness-Delta. Wenn der Fine-Tuned gewinnt: Loop ist geschlossen, das ist deine "AI evolves AI"-Demo.

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
- **`sort`** — Sortier-Netzwerk für N=8. Seed: 28 Comparatoren (All-Pairs), Knuth-Optimum: 19. Schnelles Feedback (~ms pro Kandidat).
- **`mnist_nas`** — Mini-NAS auf MNIST-Subset (5k Samples, 1 Epoche). Fitness = `accuracy − 0.05·log₁₀(params)`. Braucht `uv sync --extra nas` (torch+torchvision).

### Erstmal alles prüfen

```bash
# Preflight — sagt was installiert ist, was fehlt, was zu tun ist
uv run science-game doctor

# Smoke-Lauf OHNE Ollama/API (Mock-Provider liefert pre-baked Antworten)
uv run science-game run sort --provider mock --generations 3
# → findet das Knuth-19-Optimum innerhalb 1 Generation
```

Wenn `doctor` "All required checks passed" zeigt und der Mock-Run grün ist, läuft die komplette Pipeline (CLI → Engine → Benchmark → Events-Log → Best-Files → Dashboard-Reader). Erst dann lohnt sich Ollama-Setup.

### Engines

Zwei Optionen:

- **`standalone`** (default) — der MVP-Hill-Climber: 1 Population, 1 Parent, akzeptiert wenn besser. Schreibt vollständigen Mutations-Log (Phase-4-ready).
- **`openevolve`** — wrappt den vendor'ed [OpenEvolve](https://github.com/codelion/openevolve) Controller: MAP-Elites + Island-Model + Diff-Mutation. AlphaEvolve-style. Aktuell ohne `mutations.jsonl` (wird in einem späteren Sprint nachgereicht).

```bash
# OpenEvolve einmalig installieren
./scripts/install_openevolve.sh

# Mit OpenEvolve-Engine laufen lassen
uv run science-game run matmul --engine openevolve --provider ollama-qwen -g 100
```

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
