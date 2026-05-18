"""Generate Ollama Modelfiles for the fine-tuned mutator.

After fine-tuning on Colab via Unsloth, the user gets a GGUF file. To use
it as a regular Ollama model (and thus as a drop-in
`ollama-forge-mutator` provider in our system), they need to register it
with `ollama create`. This module produces the Modelfile.
"""

from __future__ import annotations

from pathlib import Path

DEFAULT_SYSTEM = (
    "You are an evolutionary code mutator. Given a Python program and a "
    "benchmark description, you produce an improved variant. Output ONLY a "
    "single fenced Python code block containing the full new program — no "
    "prose, no explanation, no surrounding text."
)


def build_modelfile(
    gguf_path: str,
    system_prompt: str = DEFAULT_SYSTEM,
    temperature: float = 0.8,
    num_ctx: int = 8192,
    parameter_overrides: dict[str, str] | None = None,
) -> str:
    """Return the Modelfile text. Pair with `ollama create <name> -f <file>`."""
    lines = [
        f"FROM {gguf_path}",
        f"PARAMETER temperature {temperature}",
        f"PARAMETER num_ctx {num_ctx}",
    ]
    if parameter_overrides:
        for k, v in parameter_overrides.items():
            lines.append(f"PARAMETER {k} {v}")
    lines.append(f'SYSTEM """{system_prompt}"""')
    return "\n".join(lines) + "\n"


def write_modelfile(path: Path, **kwargs) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_modelfile(**kwargs), encoding="utf-8")
    return path
