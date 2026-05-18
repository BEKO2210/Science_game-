"""MNIST NAS-light benchmark.

The evolution loop mutates a tiny PyTorch model definition; the evaluator
trains it for a small number of epochs on a MNIST subset and reports
(accuracy, param_count). Fitness is a single scalar that rewards both:

    fitness = accuracy - alpha * log10(max(params, 1))

so the engine moves along a Pareto front between accuracy and compactness.

Why this is interesting (Phase-4 angle): each accepted mutation produces a
small, named, evolved architecture — the kind of artifact that's easy to
publish as a Hugging Face model, and a great starting point for diverse
ablations.

Safety: candidate code is executed in a restricted namespace. The evaluator
expects a class `Net(nn.Module)` with a standard `forward(x)`. The torch
import is provided by the namespace so candidates don't need to manage it.

To run end-to-end you need `uv sync --extra nas` (installs torch +
torchvision). The pure-Python fitness function works without torch for
testing the wiring.
"""

from __future__ import annotations

import math
import textwrap
import traceback
from typing import Any

from science_game.benchmarks.base import Benchmark, EvalResult, register_benchmark

SEED_PROGRAM = textwrap.dedent(
    """
    \"\"\"Seed: a 1-layer MLP. ~7850 params, ~92% accuracy after 1 epoch.

    Mutator: improve accuracy and/or shrink params. Anything goes — extra
    layers, conv, BatchNorm, dropout, different activations — as long as you
    define `class Net(nn.Module)` with `forward(x)` taking a [B, 1, 28, 28]
    tensor and returning [B, 10] logits.

    Use only `nn`, `F`, `torch` — they're injected into the namespace.
    \"\"\"

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(28 * 28, 10)

        def forward(self, x):
            x = x.view(x.size(0), -1)
            return self.fc(x)
    """
).strip()


@register_benchmark("mnist_nas")
class MnistNasBenchmark(Benchmark):
    task_description = (
        "Evolve a small PyTorch model definition. Define `class Net(nn.Module)` "
        "with `forward(x)` taking input [B, 1, 28, 28] and returning [B, 10] "
        "logits. Maximize fitness = accuracy - 0.05 * log10(max(params, 1)). "
        "You have access to `torch`, `nn`, and `F` (nn.functional) in scope. "
        "Avoid huge models — Pareto-front matters."
    )

    def __init__(
        self,
        subset_size: int = 5000,
        test_size: int = 1000,
        epochs: int = 1,
        batch_size: int = 64,
        lr: float = 1e-3,
        alpha: float = 0.05,
        seed: int = 0,
        device: str | None = None,
    ) -> None:
        self.subset_size = subset_size
        self.test_size = test_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.alpha = alpha
        self.seed = seed
        self.device = device

    @property
    def seed_program(self) -> str:
        return SEED_PROGRAM

    def evaluate(self, program_code: str) -> EvalResult:
        try:
            import torch
            from torch import nn
            from torch.nn import functional as F
        except ImportError as e:
            return EvalResult(
                fitness=0.0, correct=False,
                error=f"torch not installed (uv sync --extra nas): {e!r}",
            )

        # Inject the standard torch namespace so candidates stay short.
        namespace: dict[str, Any] = {
            "__builtins__": __builtins__,
            "torch": torch, "nn": nn, "F": F,
        }
        try:
            exec(compile(program_code, "<candidate>", "exec"), namespace)
        except Exception as e:
            return EvalResult(
                fitness=0.0, correct=False,
                error=f"compile/exec: {e!r}",
                metrics={"traceback": traceback.format_exc(limit=3)},
            )

        Net = namespace.get("Net")
        if Net is None:
            return EvalResult(
                fitness=0.0, correct=False,
                error="program did not define a `Net` class",
            )

        try:
            return self._train_and_eval(Net, torch, nn, F)
        except Exception as e:
            return EvalResult(
                fitness=0.0, correct=False,
                error=f"train/eval: {e!r}",
                metrics={"traceback": traceback.format_exc(limit=3)},
            )

    def _train_and_eval(self, Net, torch, nn, F) -> EvalResult:
        device = torch.device(
            self.device if self.device
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        torch.manual_seed(self.seed)

        train_x, train_y, test_x, test_y = _mnist_subset(
            self.subset_size, self.test_size, seed=self.seed,
        )
        train_x, train_y = train_x.to(device), train_y.to(device)
        test_x, test_y = test_x.to(device), test_y.to(device)

        model = Net().to(device)
        params = sum(p.numel() for p in model.parameters())

        optim = torch.optim.Adam(model.parameters(), lr=self.lr)
        model.train()
        for _epoch in range(self.epochs):
            perm = torch.randperm(train_x.size(0), device=device)
            for i in range(0, train_x.size(0), self.batch_size):
                idx = perm[i : i + self.batch_size]
                logits = model(train_x[idx])
                loss = F.cross_entropy(logits, train_y[idx])
                optim.zero_grad()
                loss.backward()
                optim.step()

        model.eval()
        with torch.no_grad():
            preds = model(test_x).argmax(dim=1)
            acc = float((preds == test_y).float().mean().item())

        fitness = acc - self.alpha * math.log10(max(params, 1))

        return EvalResult(
            fitness=fitness, correct=True,
            metrics={
                "accuracy": acc, "params": params,
                "device": str(device), "epochs": self.epochs,
                "subset_size": self.subset_size,
            },
        )


def _mnist_subset(train_n: int, test_n: int, seed: int = 0):
    """Lazy-load MNIST through torchvision (cached locally), return tensors."""
    import torch
    from torchvision import datasets, transforms

    tfm = transforms.Compose([transforms.ToTensor()])
    root = "benchmarks/mnist"
    train = datasets.MNIST(root, train=True, download=True, transform=tfm)
    test = datasets.MNIST(root, train=False, download=True, transform=tfm)

    g = torch.Generator().manual_seed(seed)
    train_idx = torch.randperm(len(train), generator=g)[:train_n]
    test_idx = torch.randperm(len(test), generator=g)[:test_n]

    train_x = torch.stack([train[i][0] for i in train_idx])
    train_y = torch.tensor([train[i][1] for i in train_idx])
    test_x = torch.stack([test[i][0] for i in test_idx])
    test_y = torch.tensor([test[i][1] for i in test_idx])
    return train_x, train_y, test_x, test_y
