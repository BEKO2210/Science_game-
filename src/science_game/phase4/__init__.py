"""Phase 4 — Self-Improving Mutator.

This package aggregates the per-run mutation logs into training data for
fine-tuning a mutator LLM via Unsloth on Colab T4. After fine-tuning, the
exported GGUF model is loaded into Ollama and used as the `ollama-forge-mutator`
provider.

See dataset.py for the aggregation logic. The actual fine-tuning happens
in the Unsloth Studio Colab notebook (see docs/phase4-unsloth-colab.md).
"""
