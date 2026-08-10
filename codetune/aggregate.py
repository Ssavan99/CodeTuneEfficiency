"""Collapse per-run JSONs into a seed-averaged summary.

Every reported number carries a standard deviation over seeds. The 2024 paper
averaged two repeats without recording seeds or spread, which on Devign hides
run-to-run noise larger than several of the gaps it drew conclusions from.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

METRIC_KEYS = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "positive_f1"]
COST_KEYS = ["trainable_params", "trainable_pct", "peak_memory_mb", "seconds", "delta_checkpoint_mb"]
METHOD_ORDER = ["full", "bitfit", "lora", "parallel_adapter"]


def load_runs(results_dir: Path) -> list[dict]:
    runs = []
    for path in sorted(Path(results_dir).rglob("*.json")):
        if path.name == "summary.json":
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"[aggregate] skipping unreadable {path}")
            continue
        if "metrics" in data and "config" in data:
            runs.append(data)
    return runs


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    clean = [v for v in values if v is not None]
    if not clean:
        return None, None
    mean = statistics.fmean(clean)
    # stdev needs n>=2; a single seed has no measurable spread, not zero spread.
    return mean, statistics.stdev(clean) if len(clean) > 1 else None


def summarize(runs: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for run in runs:
        key = (run["config"]["task"], run["config"]["method"])
        grouped.setdefault(key, []).append(run)

    rows = []
    for (task, method), group in grouped.items():
        row = {
            "task": task,
            "method": method,
            "n_seeds": len(group),
            "seeds": sorted(r["config"]["seed"] for r in group),
            "train_n": group[0]["data"]["train"]["n_used"],
            "test_n": group[0]["data"]["test"]["n_used"],
            "epochs": group[0]["config"]["epochs"],
            "collapsed_runs": sum(1 for r in group if r["metrics"].get("collapsed")),
        }
        for key in METRIC_KEYS:
            mean, std = _mean_std([r["metrics"].get(key) for r in group])
            row[f"{key}_mean"] = round(mean, 4) if mean is not None else None
            row[f"{key}_std"] = round(std, 4) if std is not None else None
        for key in COST_KEYS:
            mean, _ = _mean_std([r["cost"].get(key) for r in group])
            row[key] = round(mean, 2) if mean is not None else None
        rows.append(row)

    rows.sort(key=lambda r: (r["task"], METHOD_ORDER.index(r["method"]) if r["method"] in METHOD_ORDER else 99))
    return rows


def _fmt(mean, std, scale=100.0, digits=2) -> str:
    if mean is None:
        return "—"
    text = f"{mean * scale:.{digits}f}"
    return f"{text} ± {std * scale:.{digits}f}" if std is not None else text


def to_markdown(rows: list[dict]) -> str:
    out = []
    for task in sorted({r["task"] for r in rows}):
        task_rows = [r for r in rows if r["task"] == task]
        head = task_rows[0]
        out.append(f"### {task} — {head['train_n']:,} train / {head['test_n']:,} test, "
                   f"{head['epochs']} epochs, {head['n_seeds']} seeds\n")
        out.append("| Method | Accuracy | Macro F1 | Positive F1 | Trainable | Delta ckpt | Peak VRAM | Train time |")
        out.append("|---|---|---|---|---|---|---|---|")
        for r in task_rows:
            vram = f"{r['peak_memory_mb']:.0f} MB" if r["peak_memory_mb"] else "—"
            secs = f"{r['seconds'] / 60:.1f} min" if r["seconds"] else "—"
            flag = " ⚠️" if r["collapsed_runs"] else ""
            out.append(
                f"| `{r['method']}`{flag} "
                f"| {_fmt(r['accuracy_mean'], r['accuracy_std'])} "
                f"| {_fmt(r['macro_f1_mean'], r['macro_f1_std'])} "
                f"| {_fmt(r['positive_f1_mean'], r['positive_f1_std'])} "
                f"| {r['trainable_pct']}% ({r['trainable_params']:,.0f}) "
                f"| {r['delta_checkpoint_mb']} MB "
                f"| {vram} | {secs} |"
            )
        if any(r["collapsed_runs"] for r in task_rows):
            out.append("\n⚠️ = at least one seed collapsed to a single predicted class.")
        out.append("")
    return "\n".join(out)


def write_summary(results_dir: str | Path = "results") -> list[dict]:
    results_dir = Path(results_dir)
    runs = load_runs(results_dir)
    if not runs:
        raise SystemExit(f"no result JSONs found under {results_dir}/ — run some experiments first")
    rows = summarize(runs)

    import pandas as pd

    pd.DataFrame(rows).to_csv(results_dir / "summary.csv", index=False)
    (results_dir / "summary.md").write_text(to_markdown(rows), encoding="utf-8")
    print(f"[aggregate] {len(runs)} runs -> {results_dir}/summary.csv, {results_dir}/summary.md")
    print(to_markdown(rows))
    return rows
