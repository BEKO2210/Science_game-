"""The engine must fail loudly when the provider is broken — not loop silently."""

from __future__ import annotations

from pathlib import Path

import pytest

from science_game.benchmarks import get_benchmark
from science_game.engine import Engine, EvolutionConfig
from science_game.llm.base import LLMProvider, MutationRequest, MutationResponse


class _AlwaysErrors(LLMProvider):
    name = "broken"

    def mutate(self, request: MutationRequest) -> MutationResponse:
        return MutationResponse(
            child_code="", raw_text="", provider=self.name, model="broken",
            meta={"error": "transport: simulated connection refused"},
        )


def test_engine_raises_after_three_consecutive_provider_errors(tmp_path: Path):
    bench = get_benchmark("matmul")
    config = EvolutionConfig(
        benchmark="matmul", llm_provider="broken",
        generations=10, run_dir=tmp_path / "fail",
    )
    engine = Engine(config, bench, _AlwaysErrors())
    with pytest.raises(RuntimeError, match="failed 3 times in a row"):
        engine.run()

    # All three errors should be in events.jsonl for forensics.
    events_text = (tmp_path / "fail" / "events.jsonl").read_text()
    assert events_text.count('"kind": "llm_error"') == 3
