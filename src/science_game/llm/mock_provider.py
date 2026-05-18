"""Deterministic mock provider for smoke tests + first-run validation.

Selects a pre-baked "good answer" based on the benchmark name. This lets us
verify the entire pipeline (CLI → engine → benchmark → events log → best
files → dashboard) without needing Ollama, an API key, or a GPU.

Use it via:
    science-game run matmul --provider mock --generations 1
    science-game run sort --provider mock --generations 1
"""

from __future__ import annotations

import textwrap

from science_game.llm.base import LLMProvider, MutationRequest, MutationResponse

# Strassen's 7-multiplication algorithm for 2x2 matmul.
_MATMUL_STRASSEN = textwrap.dedent(
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
        return [[m1 + m4 - m5 + m7, m3 + m5], [m2 + m4, m1 - m2 + m3 + m6]]
    """
).strip()

# Known-optimal 19-comparator sorting network for N=8 (Knuth TAOCP).
_SORT_KNUTH_19 = textwrap.dedent(
    """
    def build_network():
        return [
            (0, 1), (2, 3), (4, 5), (6, 7),
            (0, 2), (1, 3), (4, 6), (5, 7),
            (1, 2), (5, 6), (0, 4), (3, 7),
            (1, 5), (2, 6),
            (1, 4), (3, 6),
            (2, 4), (3, 5),
            (3, 4),
        ]
    """
).strip()

# Slightly-better-than-MLP CNN seed for the MNIST NAS benchmark.
_MNIST_CNN = textwrap.dedent(
    """
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
            self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
            self.pool = nn.MaxPool2d(2)
            self.fc = nn.Linear(16 * 7 * 7, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            return self.fc(x.view(x.size(0), -1))
    """
).strip()


_BY_BENCHMARK = {
    "matmul": _MATMUL_STRASSEN,
    "sort": _SORT_KNUTH_19,
    "mnist_nas": _MNIST_CNN,
}


class MockProvider(LLMProvider):
    """Returns a hard-coded improvement for known benchmarks; echoes parent otherwise."""

    name = "mock"

    def __init__(self, **_: object) -> None:
        pass

    def mutate(self, request: MutationRequest) -> MutationResponse:
        code = _BY_BENCHMARK.get(request.benchmark, request.parent_code)
        return MutationResponse(
            child_code=code,
            raw_text=f"```python\n{code}\n```",
            provider=self.name,
            model="mock-1",
            tokens_in=len(request.parent_code) // 4,
            tokens_out=len(code) // 4,
        )
