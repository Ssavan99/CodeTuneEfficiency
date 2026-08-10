"""Collapse per-run JSONs into a seed-averaged summary.

Every reported number carries a standard deviation over seeds. The 2024 paper
averaged two repeats without recording seeds or spread, which on Devign hides
run-to-run noise larger than several of the gaps it drew conclusions from.
"""

from __future__ import annotations

import json
import sys
import statistics
from pathlib import Path

from codetune.methods import METHODS

METRIC_KEYS = [
    "accuracy", "macro_precision", "macro_recall", "macro_f1",
    "positive_f1", "majority_class_rate",
]
COST_KEYS = [
    "trainable_params", "trainable_pct", "peak_memory_mb",
    "peak_reserved_mb", "seconds", "delta_checkpoint_mb",
]
#: Derived from the method registry so a new method needs one edit, not four.
METHOD_ORDER = list(METHODS)


def load_runs(results_dir: Path) -> list[dict]:
    runs = []
    for path in sorted(Path(results_dir).rglob("*.json")):
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
            # Cost gets a spread too. Wall-clock is the noisiest quantity here -
            # thermal throttling moves it 10-20% between seeds - and reporting it
            # as a bare point estimate is the omission this module criticises.
            mean, std = _mean_std([r["cost"].get(key) for r in group])
            row[key] = round(mean, 2) if mean is not None else None
            row[f"{key}_std"] = round(std, 2) if std is not None else None
        rows.append(row)

    rows.sort(key=lambda r: (r["task"], METHOD_ORDER.index(r["method"]) if r["method"] in METHOD_ORDER else 99))
    return rows


def _pct(mean, std) -> str:
    """A 0-1 metric as a percentage, with its seed spread when there is one."""
    if mean is None:
        return "—"
    text = f"{mean * 100:.2f}"
    return f"{text} ± {std * 100:.2f}" if std is not None else text


def _cell(value, suffix="", digits=0) -> str:
    """Any cost number, or an em dash. Never raises on a missing key."""
    if value is None:
        return "—"
    return f"{value:,.{digits}f}{suffix}"


def to_markdown(rows: list[dict]) -> str:
    out = []
    for task in sorted({r["task"] for r in rows}):
        task_rows = [r for r in rows if r["task"] == task]
        head = task_rows[0]
        out.append(f"### {task} — {head['train_n']:,} train / {head['test_n']:,} test, "
                   f"{head['epochs']} epochs\n")
        out.append(
            "| Method | Seeds | Accuracy | Macro F1 | Positive F1 "
            "| Trainable | Delta ckpt | Peak VRAM | Train time |"
        )
        out.append("|---|---|---|---|---|---|---|---|---|")
        for r in task_rows:
            secs = f"{r['seconds'] / 60:.1f} min" if r.get("seconds") else "—"
            flag = " ⚠️" if r["collapsed_runs"] else ""
            out.append(
                f"| `{r['method']}`{flag} "
                f"| {r['n_seeds']} "
                f"| {_pct(r['accuracy_mean'], r['accuracy_std'])} "
                f"| {_pct(r['macro_f1_mean'], r['macro_f1_std'])} "
                f"| {_pct(r['positive_f1_mean'], r['positive_f1_std'])} "
                f"| {_cell(r['trainable_pct'], '%', 3)} ({_cell(r['trainable_params'])}) "
                f"| {_cell(r['delta_checkpoint_mb'], ' MB', 2)} "
                f"| {_cell(r.get('peak_reserved_mb') or r.get('peak_memory_mb'), ' MB')} "
                f"| {secs} |"
            )
        if any(r["collapsed_runs"] for r in task_rows):
            out.append(
                "\n⚠️ = at least one seed predicted a single class for ≥99% of inputs, "
                "which is a training failure rather than a result about the method."
            )
        out.append("")
    return "\n".join(out)


def _safe_print(text: str) -> None:
    """Print to a console that may not speak UTF-8.

    The Windows console defaults to cp1252, which cannot encode the ± and ⚠️ in
    these tables. Letting that raise would abort *after* the files are written -
    and a non-zero exit stops any `aggregate && plot` chain, so the figures never
    get made for a failure that is purely cosmetic.
    """
    try:
        print(text)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or "utf-8"
        print(text.encode(encoding, errors="replace").decode(encoding))


README_START = "<!-- RESULTS:START -->"
README_END = "<!-- RESULTS:END -->"


def update_readme(markdown: str, readme: Path) -> bool:
    """Splice the results tables into README.md between its marker comments.

    Keeps the headline numbers in the README generated rather than hand-copied,
    so they cannot drift away from what is actually in results/.
    """
    if not readme.exists():
        return False
    text = readme.read_text(encoding="utf-8")
    start, end = text.find(README_START), text.find(README_END)
    if start == -1 or end == -1:
        return False
    block = f"{README_START}\n\n## Results\n\n{markdown}\n{README_END}"
    readme.write_text(text[:start] + block + text[end + len(README_END):], encoding="utf-8")
    return True


def write_summary(results_dir: str | Path = "results") -> list[dict]:
    results_dir = Path(results_dir)
    runs = load_runs(results_dir)
    if not runs:
        raise SystemExit(f"no result JSONs found under {results_dir}/ — run some experiments first")
    rows = summarize(runs)
    markdown = to_markdown(rows)

    import pandas as pd

    pd.DataFrame(rows).to_csv(results_dir / "summary.csv", index=False)
    (results_dir / "summary.md").write_text(markdown, encoding="utf-8")
    print(f"[aggregate] {len(runs)} runs -> {results_dir}/summary.csv, {results_dir}/summary.md")

    readme = Path(__file__).resolve().parent.parent / "README.md"
    if update_readme(markdown, readme):
        print(f"[aggregate] refreshed the results section of {readme.name}")

    _safe_print(markdown)
    return rows
