"""Figures: accuracy against each cost axis, plus per-method bars with error bars."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from codetune.aggregate import load_runs, summarize  # noqa: E402

from codetune.methods import METHODS  # noqa: E402

#: One colour per registered method, assigned by registry order, so a new method
#: gets a distinct colour instead of silently rendering grey.
_PALETTE = ["#264653", "#2a9d8f", "#e9c46a", "#e76f51", "#8d5a97", "#4f7cac"]
COLORS = {name: _PALETTE[i % len(_PALETTE)] for i, name in enumerate(METHODS)}


def _scatter(ax, rows, x_key, xlabel, log_x=False):
    for row in rows:
        x, y = row.get(x_key), row.get("macro_f1_mean")
        if x is None or y is None:
            continue
        ax.scatter(x, y * 100, s=110, color=COLORS.get(row["method"], "#888"),
                   edgecolor="white", linewidth=1.5, zorder=3)
        ax.annotate(row["method"], (x, y * 100), textcoords="offset points",
                    xytext=(8, 4), fontsize=9)
    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Macro F1 (%)")
    ax.grid(alpha=0.25, zorder=0)


def make_plots(results_dir: str | Path = "results") -> list[Path]:
    results_dir = Path(results_dir)
    rows = summarize(load_runs(results_dir))
    if not rows:
        raise SystemExit(f"no results under {results_dir}/")

    fig_dir = results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    written = []

    for task in sorted({r["task"] for r in rows}):
        task_rows = [r for r in rows if r["task"] == task]

        # 1. Accuracy against the two cost axes that matter when choosing a method.
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        _scatter(axes[0], task_rows, "trainable_params", "Trainable parameters (log)", log_x=True)
        _scatter(axes[1], task_rows, "peak_memory_mb", "Peak GPU memory allocated (MB)")
        fig.suptitle(f"{task}: quality vs. cost", fontsize=13)
        fig.tight_layout()
        path = fig_dir / f"{task}_quality_vs_cost.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)

        # 2. Per-method quality with seed spread.
        fig, ax = plt.subplots(figsize=(7, 4.2))
        names = [r["method"] for r in task_rows]
        means = [(r["macro_f1_mean"] or 0) * 100 for r in task_rows]
        errs = [(r["macro_f1_std"] or 0) * 100 for r in task_rows]
        ax.bar(names, means, yerr=errs, capsize=5,
               color=[COLORS.get(n, "#888") for n in names], zorder=3)
        ax.set_ylabel("Macro F1 (%)")
        ax.set_title(f"{task}: macro F1 over {task_rows[0]['n_seeds']} seeds "
                     f"(equal {task_rows[0]['epochs']}-epoch budget)")
        ax.grid(axis="y", alpha=0.25, zorder=0)
        fig.tight_layout()
        path = fig_dir / f"{task}_macro_f1.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)

    print(f"[plots] wrote {len(written)} figures to {fig_dir}/")
    return written
