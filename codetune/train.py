"""One fine-tuning run: train, evaluate, and write a self-describing result file.

Every method gets the **same** budget — same epochs, same data, same seed, same
sequence length. That is deliberate: the 2024 runs gave LoRA 15 epochs, Parallel
Adapter 6 and BitFit 5, then compared the results as though the contest had been
fair. Early stopping is off for the same reason.
"""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

from codetune.cost import CostTracker, cost_summary
from codetune.data import build_split
from codetune.methods import apply_method

MODEL_NAME = "microsoft/codebert-base"


@dataclass
class RunConfig:
    task: str = "defect"
    method: str = "full"
    seed: int = 42
    epochs: int = 3
    lr: float = 5e-5
    peft_lr: float = 1e-4
    batch_size: int = 8
    grad_accum: int = 4
    max_length: int = 320
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    fp16: bool = True
    device: str = "auto"
    limit_train: int | None = None
    limit_eval: int | None = None
    model_name: str = MODEL_NAME
    output_dir: str = "results"
    method_kwargs: dict = field(default_factory=dict)

    def resolved_device(self) -> torch.device:
        if self.device != "auto":
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def learning_rate(self) -> float:
        """PEFT methods need a larger LR than full fine-tuning (as in the paper)."""
        return self.lr if self.method == "full" else self.peft_lr

    def run_id(self) -> str:
        return f"{self.task}__{self.method}__seed{self.seed}"

    def result_path(self) -> Path:
        """Where this run's result lands. Owned here so the writer and the
        grid's already-done check can never disagree about the filename."""
        return Path(self.output_dir) / self.task / f"{self.method}__seed{self.seed}.json"


def compute_metrics(labels: np.ndarray, preds: np.ndarray) -> dict:
    """Macro *and* positive-class scores.

    Reporting both is not padding. A classifier that predicts one class for every
    input scores exactly 0.50 macro recall and 0.00 positive-class F1 — the
    combination immediately identifies a collapsed run, which is precisely the
    failure the 2024 LoRA defect numbers exhibited without it being noticed.
    """
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_precision": float(precision_score(labels, preds, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(labels, preds, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "positive_f1": float(f1_score(labels, preds, pos_label=1, average="binary", zero_division=0)),
        "predicted_positive_rate": float(np.mean(preds == 1)),
        # Near-collapse is the same failure as collapse and just as misleading:
        # a run predicting one class 99% of the time is not a result about the
        # method. Threshold rather than exact uniformity so it is actually caught.
        "majority_class_rate": float(np.bincount(preds, minlength=2).max() / len(preds)),
        "collapsed": bool(np.bincount(preds, minlength=2).max() / len(preds) >= 0.99),
    }


@torch.no_grad()
def evaluate(model, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        labels = batch.pop("labels")
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        logits = model(**batch).logits
        all_preds.append(logits.argmax(dim=-1).cpu().numpy())
        all_labels.append(labels.numpy())
    return compute_metrics(np.concatenate(all_labels), np.concatenate(all_preds))


def _build_model(cfg: RunConfig, device: torch.device) -> tuple[torch.nn.Module, int]:
    """Build the model and return it with the pristine parameter count.

    The count is taken *before* the method runs, so every method is measured
    against the same denominator even when it restructures the module tree.
    """
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(cfg.model_name, num_labels=2)
    base_total = sum(p.numel() for p in model.parameters())
    model = apply_method(model, cfg.method, cfg.method_kwargs)
    return model.to(device), base_total


def train_one_run(cfg: RunConfig, verbose: bool = True, _max_oom_retries: int = 3) -> dict:
    """Run once; on CUDA OOM, halve batch size and double grad accumulation, then retry.

    The effective batch size (batch_size * grad_accum) is preserved across retries so
    the equal-budget comparison across methods still holds — only memory footprint per
    step shrinks.
    """
    for attempt in range(_max_oom_retries + 1):
        try:
            return _train_one_run_attempt(cfg, verbose)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if attempt == _max_oom_retries or cfg.batch_size <= 1:
                raise
            new_batch = max(1, cfg.batch_size // 2)
            new_accum = cfg.grad_accum * (cfg.batch_size // new_batch)
            if verbose:
                print(
                    f"[{cfg.run_id()}] OOM at batch_size={cfg.batch_size}; retrying with "
                    f"batch_size={new_batch}, grad_accum={new_accum}"
                )
            cfg = replace(cfg, batch_size=new_batch, grad_accum=new_accum)
    raise RuntimeError("unreachable")  # pragma: no cover


def _train_one_run_attempt(cfg: RunConfig, verbose: bool) -> dict:
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup, set_seed

    set_seed(cfg.seed)
    device = cfg.resolved_device()
    use_amp = cfg.fp16 and device.type == "cuda"

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    train_ds, train_stats = build_split(
        cfg.task, "train", tokenizer, cfg.max_length, cfg.limit_train, cfg.seed
    )
    eval_ds, eval_stats = build_split(
        cfg.task, "test", tokenizer, cfg.max_length, cfg.limit_eval, cfg.seed
    )
    # non_blocking transfers are silently synchronous unless the host buffer is
    # pinned, so without this the async syntax below buys nothing.
    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False, pin_memory=pin
    )
    eval_loader = DataLoader(eval_ds, batch_size=cfg.batch_size * 2, shuffle=False, pin_memory=pin)

    model, base_total = _build_model(cfg, device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg.learning_rate(), weight_decay=cfg.weight_decay)

    # Ceiling, not floor: the loop also steps on the partial group left at the
    # end of each epoch. Sizing the schedule with floor would leave the last
    # optimizer steps of the run clamped at a learning rate of zero.
    n_batches = len(train_loader)
    steps_per_epoch = max(1, -(-n_batches // cfg.grad_accum))
    total_steps = steps_per_epoch * cfg.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * cfg.warmup_ratio), total_steps
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    if verbose:
        print(
            f"[{cfg.run_id()}] device={device.type} train={train_stats['n_used']} "
            f"eval={eval_stats['n_used']} steps={total_steps} lr={cfg.learning_rate():g}"
        )

    def optimizer_step() -> None:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

    history = []
    with CostTracker(device) as tracker:
        for epoch in range(cfg.epochs):
            model.train()
            # Accumulated on device: calling .item() per micro-batch forces a
            # host-GPU sync every step, which inflates the wall-clock this
            # benchmark reports - and inflates it most for the cheapest methods,
            # biasing the very comparison the repo exists to make.
            running = torch.zeros((), device=device)
            optimizer.zero_grad(set_to_none=True)
            for step, batch in enumerate(train_loader):
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    loss = model(**batch).loss / cfg.grad_accum
                scaler.scale(loss).backward()

                # The second clause flushes the partial group at the end of an
                # epoch through the same path as a normal step.
                if (step + 1) % cfg.grad_accum == 0 or step + 1 == n_batches:
                    optimizer_step()

                running += loss.detach() * cfg.grad_accum
            history.append(
                {"epoch": epoch + 1, "train_loss": round(running.item() / max(n_batches, 1), 5)}
            )
            if verbose:
                print(f"  epoch {epoch + 1}/{cfg.epochs} loss={history[-1]['train_loss']:.4f}")

    metrics = evaluate(model, eval_loader, device)
    result = {
        "run_id": cfg.run_id(),
        "config": asdict(cfg),
        "effective_learning_rate": cfg.learning_rate(),
        "effective_batch_size": cfg.batch_size * cfg.grad_accum,
        "data": {"train": train_stats, "test": eval_stats},
        "metrics": metrics,
        "cost": cost_summary(model, tracker, base_total),
        "history": history,
        "environment": {
            "device": device.type,
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "torch": torch.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "fp16": use_amp,
        },
    }

    out_path = cfg.result_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if verbose:
        cost = result["cost"]
        print(
            f"  -> acc={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f} "
            f"| {cost['trainable_pct']}% params, {cost['delta_checkpoint_mb']} MB delta, "
            f"{cost['seconds']}s\n  -> {out_path}"
        )
    return result
