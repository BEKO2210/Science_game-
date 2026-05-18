"""End-to-end smoke test using a deterministic in-process LLM stub.

This verifies the full engine loop — request -> mutate -> evaluate -> log —
without needing Ollama. It also asserts the artifacts (events.jsonl,
mutations.jsonl, best/*.py) are written.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

from science_game.benchmarks import get_benchmark
from science_game.engine import Engine, EvolutionConfig
from science_game.llm import LLMProvider, MutationRequest, MutationResponse


class ScriptedProvider(LLMProvider):
    """Replays a fixed sequence of child programs — perfect for deterministic tests."""

    name = "scripted"

    def __init__(self, programs: list[str]) -> None:
        self.programs = programs
        self.idx = 0

    def mutate(self, request: MutationRequest) -> MutationResponse:
        code = self.programs[self.idx % len(self.programs)]
        self.idx += 1
        return MutationResponse(
            child_code=code,
            raw_text=code,
            provider=self.name,
            model="scripted-1",
            tokens_in=0,
            tokens_out=0,
        )


def test_engine_accepts_improvement(tmp_path: Path):
    bench = get_benchmark("matmul")

    # A program identical to the seed should NOT improve fitness.
    same = bench.seed_program

    # Strassen — 7 mults, definitely improves.
    strassen = textwrap.dedent(
        """
        def matmul2x2(A, B, mul):
            a, b = A[0][0], A[0][1]
            c, d = A[1][0], A[1][1]
            e, f = B[0][0], B[0][1]
            g, h = B[1][0], B[1][1]
            m1 = mul(a + d, e + h)
            m2 = mul(c + d, e)
            m3 = mul(a, f - h)
            m4 = mul(d, g - e)
            m5 = mul(a + b, h)
            m6 = mul(c - a, e + f)
            m7 = mul(b - d, g + h)
            c00 = m1 + m4 - m5 + m7
            c01 = m3 + m5
            c10 = m2 + m4
            c11 = m1 - m2 + m3 + m6
            return [[c00, c01], [c10, c11]]
        """
    ).strip()

    provider = ScriptedProvider([same, strassen])
    config = EvolutionConfig(
        benchmark="matmul",
        llm_provider="scripted",
        generations=2,
        seed=0,
        run_dir=tmp_path,
    )
    engine = Engine(config, bench, provider)
    best = engine.run()

    assert best.result.correct
    assert best.result.metrics["mult_count"] == 7
    assert best.generation == 2  # Strassen accepted in gen 2

    # Artifacts exist
    events = (tmp_path / "events.jsonl").read_text().strip().splitlines()
    assert len(events) >= 3  # init + 2 generations
    mutations = (tmp_path / "mutations.jsonl").read_text().strip().splitlines()
    assert len(mutations) == 2

    # Mutation log has the Phase-4-ready fields.
    rec = json.loads(mutations[1])
    for field in (
        "generation", "parent_code", "child_code",
        "fitness_before", "fitness_after", "accepted",
        "provider", "model",
    ):
        assert field in rec, f"missing field {field}"
    assert rec["accepted"] is True

    # Best-of-run file is on disk.
    latest = (tmp_path / "best" / "latest.py").read_text()
    assert "def matmul2x2" in latest


def test_engine_rejects_regression(tmp_path: Path):
    bench = get_benchmark("matmul")
    # Always returns a broken program — fitness 0, never accepted.
    bad = "def matmul2x2(A, B, mul):\n    return [[0, 0], [0, 0]]\n"
    provider = ScriptedProvider([bad])
    config = EvolutionConfig(
        benchmark="matmul",
        llm_provider="scripted",
        generations=3,
        seed=0,
        run_dir=tmp_path,
    )
    engine = Engine(config, bench, provider)
    best = engine.run()

    # Best stays at the seed.
    assert best.generation == 0
    assert best.result.metrics["mult_count"] == 8

    mutations = (tmp_path / "mutations.jsonl").read_text().strip().splitlines()
    assert len(mutations) == 3
    for line in mutations:
        rec = json.loads(line)
        assert rec["accepted"] is False
