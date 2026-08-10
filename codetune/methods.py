"""The four fine-tuning strategies under comparison.

Every method exposes the same contract: take a ``RobertaForSequenceClassification``,
mark the right parameters trainable, and return the model. The caller then reads
the trainable-parameter count back off the model, so a method cannot misreport
what it actually trains.

- ``full``             every parameter (the baseline)
- ``bitfit``           bias terms only, plus the classification head
- ``lora``             low-rank updates on the attention query/value projections
- ``parallel_adapter`` a bottleneck MLP alongside each layer's feed-forward block
"""

from __future__ import annotations

import torch
from torch import nn

#: The classifier is randomly initialised, so it must be trainable under every
#: method or nothing can be learned at all.
CLASSIFIER_PREFIXES = ("classifier.", "score.")


def _freeze_all(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad_(False)


def _unfreeze_classifier(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        if any(key in name for key in CLASSIFIER_PREFIXES):
            param.requires_grad_(True)


def apply_full(model: nn.Module, cfg: dict) -> nn.Module:
    for param in model.parameters():
        param.requires_grad_(True)
    return model


def apply_bitfit(model: nn.Module, cfg: dict) -> nn.Module:
    """BitFit: train only bias vectors (Ben Zaken et al., ACL 2022)."""
    _freeze_all(model)
    for name, param in model.named_parameters():
        if name.endswith(".bias") or name == "bias":
            param.requires_grad_(True)
    _unfreeze_classifier(model)
    return model


def apply_lora(model: nn.Module, cfg: dict) -> nn.Module:
    """LoRA on the attention query/value projections (Hu et al., ICLR 2022)."""
    from peft import LoraConfig, TaskType, get_peft_model

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=int(cfg.get("lora_r", 8)),
        lora_alpha=int(cfg.get("lora_alpha", 16)),
        lora_dropout=float(cfg.get("lora_dropout", 0.1)),
        target_modules=list(cfg.get("lora_targets", ["query", "value"])),
        bias="none",
    )
    return get_peft_model(model, peft_config)


class Adapter(nn.Module):
    """Bottleneck down-project → ReLU → up-project.

    The up-projection starts at zero so the adapter is exactly the identity at
    initialisation and the pre-trained function is preserved on step 0.
    """

    def __init__(self, hidden_size: int, bottleneck: int, scale: float = 2.0):
        super().__init__()
        self.down = nn.Linear(hidden_size, bottleneck)
        self.up = nn.Linear(bottleneck, hidden_size)
        self.act = nn.ReLU()
        self.scale = scale
        nn.init.normal_(self.down.weight, std=1e-2)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.scale * self.up(self.act(self.down(hidden)))


class ParallelAdapterOutput(nn.Module):
    """Wraps a ``RobertaOutput`` to run an adapter parallel to the FFN block.

    ``RobertaOutput.forward(hidden_states, input_tensor)`` computes
    ``LayerNorm(dropout(dense(hidden_states)) + input_tensor)``, where
    ``input_tensor`` is the FFN's input. A *parallel* adapter reads that same
    input and its output is summed in before the residual LayerNorm — as opposed
    to a sequential adapter, which reads the FFN's output. This is the
    formulation from He et al., *Towards a Unified View of Parameter-Efficient
    Transfer Learning* (ICLR 2022).
    """

    def __init__(self, inner: nn.Module, adapter: Adapter):
        super().__init__()
        self.inner = inner
        self.adapter = adapter

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.inner.dense(hidden_states)
        hidden_states = self.inner.dropout(hidden_states)
        hidden_states = hidden_states + self.adapter(input_tensor)
        return self.inner.LayerNorm(hidden_states + input_tensor)


def _encoder_layers(model: nn.Module):
    """Find the transformer layer list on a RoBERTa/BERT-style model."""
    for attr in ("roberta", "bert", "base_model"):
        base = getattr(model, attr, None)
        if base is not None and hasattr(base, "encoder"):
            return base.encoder.layer
    raise AttributeError(
        f"could not locate encoder layers on {type(model).__name__}; "
        "parallel_adapter supports RoBERTa/BERT-style encoders"
    )


def apply_parallel_adapter(model: nn.Module, cfg: dict) -> nn.Module:
    _freeze_all(model)
    bottleneck = int(cfg.get("adapter_bottleneck", 16))
    scale = float(cfg.get("adapter_scale", 2.0))
    hidden_size = model.config.hidden_size

    for layer in _encoder_layers(model):
        adapter = Adapter(hidden_size, bottleneck, scale).to(
            device=layer.output.dense.weight.device,
            dtype=layer.output.dense.weight.dtype,
        )
        layer.output = ParallelAdapterOutput(layer.output, adapter)

    for name, param in model.named_parameters():
        if ".adapter." in name:
            param.requires_grad_(True)
    _unfreeze_classifier(model)
    return model


METHODS = {
    "full": apply_full,
    "bitfit": apply_bitfit,
    "lora": apply_lora,
    "parallel_adapter": apply_parallel_adapter,
}


def apply_method(model: nn.Module, method: str, cfg: dict | None = None) -> nn.Module:
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {sorted(METHODS)}")
    model = METHODS[method](model, cfg or {})
    if not any(p.requires_grad for p in model.parameters()):
        raise RuntimeError(f"method {method!r} left every parameter frozen")
    return model
