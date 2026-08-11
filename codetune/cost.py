"""Cost accounting — the axis the 2024 study left out.

The original paper compared four fine-tuning methods on accuracy alone, despite
efficiency being its subject. These four numbers are what actually differ
between the methods:

- **trainable parameters** — what the optimizer updates
- **peak GPU memory**      — whether it fits on the card you own
- **wall-clock train time** — what it costs to run
- **delta checkpoint size** — what you must store and ship per task

The last one is the honest version of PEFT's storage claim. The 2024 runs saved
a full ~500 MB ``model.safetensors`` for every method, so the saving was never
realised; here only the trainable tensors are counted and written.
"""

from __future__ import annotations

import time

import torch
from torch import nn


def count_parameters(model: nn.Module, base_total: int | None = None) -> dict:
    """Count trainable parameters against a fixed denominator.

    ``base_total`` should be the parameter count of the *pristine* model, taken
    before any method touched it. Without it, methods that restructure the module
    tree are measured against different denominators: ``peft`` keeps a frozen
    ``original_module`` copy of the classifier alongside the trainable one, which
    inflates LoRA's total by ~592 K parameters on CodeBERT and quietly makes its
    percentage incomparable with the other three methods.
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    live_total = sum(p.numel() for p in model.parameters())
    total = base_total if base_total is not None else live_total
    return {
        "trainable_params": trainable,
        "total_params": total,
        "live_total_params": live_total,
        "trainable_pct": round(100.0 * trainable / total, 4) if total else 0.0,
    }


def trainable_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Just the tensors a deployment would actually need to ship for this task."""
    trainable_names = {name for name, p in model.named_parameters() if p.requires_grad}
    return {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
        if name in trainable_names
    }


def delta_checkpoint_bytes(model: nn.Module) -> int:
    """Size on disk of a trainable-parameters-only checkpoint.

    Read from tensor metadata rather than by building the state dict: for full
    fine-tuning the latter copies ~476 MB device-to-host purely to sum its
    ``numel``, which is available without touching the data.
    """
    return sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)


class CostTracker:
    """Times a block and records peak CUDA allocation over it.

    On CPU the memory figure is reported as None rather than 0, so a CPU smoke
    run is never mistaken for a GPU run that used no memory.
    """

    def __init__(self, device: torch.device | str = "cpu"):
        self.device = torch.device(device)
        self.use_cuda = self.device.type == "cuda"
        self.seconds: float | None = None
        self.peak_memory_bytes: int | None = None
        self.peak_reserved_bytes: int | None = None

    def __enter__(self) -> "CostTracker":
        if self.use_cuda:
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc_info) -> None:
        if self.use_cuda:
            torch.cuda.synchronize(self.device)
            self.peak_memory_bytes = torch.cuda.max_memory_allocated(self.device)
            self.peak_reserved_bytes = torch.cuda.max_memory_reserved(self.device)
        self.seconds = time.perf_counter() - self._start

    def as_dict(self) -> dict:
        def mb(value: int | None) -> float | None:
            return round(value / 1024**2, 1) if value is not None else None

        return {
            "seconds": round(self.seconds, 2) if self.seconds is not None else None,
            "peak_memory_bytes": self.peak_memory_bytes,
            "peak_memory_mb": mb(self.peak_memory_bytes),
            # Reserved is what the card actually has to give up: allocated bytes
            # exclude fragmentation and cached-but-unused blocks, so on their own
            # they understate whether a run fits.
            "peak_reserved_bytes": self.peak_reserved_bytes,
            "peak_reserved_mb": mb(self.peak_reserved_bytes),
        }


def cost_summary(model: nn.Module, tracker: CostTracker, base_total: int | None = None) -> dict:
    delta_bytes = delta_checkpoint_bytes(model)
    return {
        **count_parameters(model, base_total),
        **tracker.as_dict(),
        "delta_checkpoint_bytes": delta_bytes,
        "delta_checkpoint_mb": round(delta_bytes / 1024**2, 2),
    }
