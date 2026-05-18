"""Build a mutator fine-tuning dataset from runs/*/mutations.jsonl.

Two modes:

**SFT** (supervised fine-tuning) — one example per accepted improving
mutation, formatted as instruction/output for `unsloth.chat_templates`:
    {"messages": [{"role": "user", "content": <prompt>},
                  {"role": "assistant", "content": <child_code>}]}

**DPO** (direct preference optimization) — paired examples per parent.
For each parent that appears with multiple children in the logs, we form
(chosen, rejected) pairs by ranking children by fitness_after:
    {"prompt": <prompt>, "chosen": <best_child>, "rejected": <worst_child>}

Both formats are compatible with Unsloth's HF Hub trainer recipes.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path


def _iter_mutations(runs_root: Path) -> Iterable[dict]:
    for run_dir in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        mfile = run_dir / "mutations.jsonl"
        if not mfile.exists():
            continue
        with mfile.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def _format_prompt(parent_code: str) -> str:
    return (
        "Below is the current best program. Produce an improved variant. "
        "Return ONLY a single fenced Python code block with the full new program.\n\n"
        "```python\n"
        f"{parent_code}\n"
        "```\n"
    )


def _build_sft(rows: list[dict], min_delta: float) -> list[dict]:
    examples = []
    for r in rows:
        if not r.get("accepted"):
            continue
        delta = float(r.get("fitness_after", 0)) - float(r.get("fitness_before", 0))
        if delta <= min_delta:
            continue
        parent = r.get("parent_code")
        child = r.get("child_code")
        if not parent or not child:
            continue
        examples.append({
            "messages": [
                {"role": "user", "content": _format_prompt(parent)},
                {"role": "assistant", "content": f"```python\n{child}\n```"},
            ],
            "meta": {
                "fitness_before": r.get("fitness_before"),
                "fitness_after": r.get("fitness_after"),
                "delta": delta,
                "provider": r.get("provider"),
                "model": r.get("model"),
            },
        })
    return examples


def _build_dpo(rows: list[dict]) -> list[dict]:
    by_parent: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        parent = r.get("parent_code")
        child = r.get("child_code")
        if not parent or not child:
            continue
        by_parent[parent].append(r)

    examples = []
    for parent, children in by_parent.items():
        if len(children) < 2:
            continue
        ranked = sorted(children, key=lambda r: float(r.get("fitness_after", 0)), reverse=True)
        best, worst = ranked[0], ranked[-1]
        if float(best.get("fitness_after", 0)) <= float(worst.get("fitness_after", 0)):
            continue
        examples.append({
            "prompt": _format_prompt(parent),
            "chosen": f"```python\n{best['child_code']}\n```",
            "rejected": f"```python\n{worst['child_code']}\n```",
            "meta": {
                "chosen_fitness": best.get("fitness_after"),
                "rejected_fitness": worst.get("fitness_after"),
                "num_siblings": len(children),
            },
        })
    return examples


def build_dataset(
    runs_root: Path,
    out: Path,
    mode: str = "sft",
    min_delta: float = 0.0,
) -> int:
    """Aggregate runs into a JSONL training file. Returns row count written."""
    runs_root = Path(runs_root)
    out = Path(out)
    if not runs_root.exists():
        raise FileNotFoundError(f"runs root not found: {runs_root}")

    rows = list(_iter_mutations(runs_root))

    if mode == "sft":
        examples = _build_sft(rows, min_delta=min_delta)
    elif mode == "dpo":
        examples = _build_dpo(rows)
    else:
        raise ValueError(f"unknown mode: {mode!r} — use 'sft' or 'dpo'")

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    return len(examples)
