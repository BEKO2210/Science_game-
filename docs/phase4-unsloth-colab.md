# Phase 4 — Self-Improving Mutator via Unsloth on Colab

This document walks through closing the loop: take the mutation log your
runs produce, fine-tune a specialized mutator LLM, and load it back into
Ollama as a new provider.

## Prerequisites

- One or more completed Algorithm Forge runs with mutation logs in
  `runs/*/mutations.jsonl` (the more, the better — aim for >500 accepted
  mutations across runs).
- A Hugging Face account; you'll need an HF write token for the model
  upload step. You are signed in as `Beko2210`.
- A Google account (for free Colab T4 GPU access).

## Step 1 — Build the dataset

```bash
uv run science-game build-mutator-dataset \
    --runs-root runs \
    --out datasets/mutator-v1.jsonl \
    --mode sft \
    --min-delta 0.0001
```

This walks every `runs/*/mutations.jsonl`, keeps the accepted improving
mutations whose `fitness_after - fitness_before > min-delta`, and writes a
JSONL where each row is:

```json
{"messages": [
  {"role": "user", "content": "<prompt with parent code>"},
  {"role": "assistant", "content": "```python\n<child code>\n```"}
], "meta": {"fitness_before": 0.125, "fitness_after": 0.143, "delta": 0.018, ...}}
```

For DPO instead of SFT (recommended once you have enough siblings per parent):

```bash
uv run science-game build-mutator-dataset --mode dpo --out datasets/mutator-v1-dpo.jsonl
```

## Step 2 — Upload the dataset to HF Hub

```bash
hf auth login
hf upload-dataset Beko2210/algorithm-forge-mutations-v1 datasets/mutator-v1.jsonl
```

## Step 3 — Fine-tune via Unsloth Studio on Colab

1. Open the [Unsloth Studio Colab notebook](https://colab.research.google.com/github/unslothai/unsloth/blob/main/studio/Unsloth_Studio_Colab.ipynb).
2. Runtime → Change runtime type → **T4 GPU** (free tier).
3. Runtime → Run all to boot Unsloth Studio; click the UI link it prints.
4. In Studio:
   - **Base model**: Qwen2.5-Coder-7B (or Gemma 4 — both fit in 4-bit on T4).
   - **Dataset**: load `Beko2210/algorithm-forge-mutations-v1` from HF.
   - **Mode**: LoRA 4-bit; rank 16; alpha 16; lr 2e-4; 2-3 epochs.
   - Click **Train**, watch the loss curve.
5. **Export**: GGUF (Q4_K_M for speed, Q8_0 for fidelity) AND LoRA adapter.
6. **Push to Hub**: `Beko2210/algorithm-forge-mutator-qwen-v1`.

## Step 4 — Load into Ollama

```bash
# Download the GGUF (HF will give you a direct URL).
mkdir -p ~/ollama-models
curl -L -o ~/ollama-models/forge-mutator-v1.gguf \
    "https://huggingface.co/Beko2210/algorithm-forge-mutator-qwen-v1/resolve/main/model-Q4_K_M.gguf"

# Write a Modelfile.
cat > ~/ollama-models/Modelfile <<'EOF'
FROM ./forge-mutator-v1.gguf
PARAMETER temperature 0.8
PARAMETER num_ctx 8192
SYSTEM "You are an evolutionary code mutator. Given a Python program and a benchmark description, you produce an improved variant. Output ONLY a single fenced Python code block."
EOF

# Register.
ollama create algorithm-forge-mutator -f ~/ollama-models/Modelfile
```

## Step 5 — A/B compare

In the dashboard Launcher, run the same benchmark + seed twice — once
with `ollama-qwen` (base) and once with model override
`algorithm-forge-mutator`. Open the Compare page to overlay the
trajectories. A win shows up as a faster climb and/or a higher plateau.

## Iteration

Treat each Step-1-through-5 cycle as a versioned dataset + model:
v1 trained on data from base-Qwen runs; v2 trained on data that includes
v1-mutator runs; etc. The HF Hub repos `…-v1`, `…-v2`, … give you a
clean lineage to plot in the paper.
