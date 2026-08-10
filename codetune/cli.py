"""Command line entry point: ``python -m codetune <command>``."""

from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
import sys
import zipfile
from pathlib import Path

import yaml

from codetune.data import DATA_ROOT, LOADERS
from codetune.methods import METHODS
from codetune.train import RunConfig, train_one_run

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ZIP = REPO_ROOT.parent / "CodeTuneEfficiency-model" / "defect.zip"


def _config_from(path: str | None, overrides: dict) -> RunConfig:
    values: dict = {}
    if path:
        loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        values.update(loaded)
    values.update({k: v for k, v in overrides.items() if v is not None})

    known = {f.name for f in dataclasses.fields(RunConfig)}
    unknown = set(values) - known
    if unknown:
        raise SystemExit(f"unknown config keys: {sorted(unknown)}")
    return RunConfig(**values)


def cmd_run(args: argparse.Namespace) -> int:
    cfg = _config_from(
        args.config,
        {
            "task": args.task,
            "method": args.method,
            "seed": args.seed,
            "epochs": args.epochs,
            "device": args.device,
            "limit_train": args.limit_train,
            "limit_eval": args.limit_eval,
            "output_dir": args.output_dir,
        },
    )
    train_one_run(cfg)
    return 0


def cmd_grid(args: argparse.Namespace) -> int:
    """Run every (method, seed) combination for one config, skipping finished runs."""
    methods = args.methods.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    failures = []
    for method in methods:
        for seed in seeds:
            cfg = _config_from(
                args.config,
                {"method": method, "seed": seed, "device": args.device, "output_dir": args.output_dir},
            )
            if cfg.result_path().exists() and not args.force:
                print(f"[grid] skip {cfg.run_id()} (already done)")
                continue
            try:
                train_one_run(cfg)
            except Exception as exc:  # keep the grid moving; report at the end
                print(f"[grid] FAILED {cfg.run_id()}: {type(exc).__name__}: {exc}", file=sys.stderr)
                failures.append((cfg.run_id(), f"{type(exc).__name__}: {exc}"))
    if failures:
        print(f"\n[grid] {len(failures)} run(s) failed:", file=sys.stderr)
        for run_id, err in failures:
            print(f"  {run_id}: {err}", file=sys.stderr)
        return 1
    return 0


def cmd_aggregate(args: argparse.Namespace) -> int:
    from codetune.aggregate import write_summary

    write_summary(args.output_dir)
    return 0


def cmd_plot(args: argparse.Namespace) -> int:
    from codetune.plots import make_plots

    make_plots(args.output_dir)
    return 0


def _prepare_clone() -> None:
    """Stage BigCloneBench under ``data/clone/``.

    Prefers a local copy from the upstream layout; that data is gitignored, so a
    fresh clone of this repo (Colab, Kaggle, anyone else's machine) will not have
    it and falls back to the public CodeXGLUE copy on the Hugging Face Hub. Free,
    public, no account or token needed.
    """
    names = ("data.jsonl", "train.txt", "valid.txt", "test.txt")
    if all((DATA_ROOT / "clone" / n).exists() for n in names):
        print("[prepare] clone: already staged")
        return

    local = REPO_ROOT / "clone" / "dataset"
    if all((local / n).exists() for n in names):
        for name in names:
            shutil.copy2(local / name, DATA_ROOT / "clone" / name)
            print(f"[prepare] clone/{name} (local)")
        return

    print("[prepare] clone: downloading BigCloneBench from the Hugging Face Hub")
    from datasets import load_dataset

    ds = load_dataset(
        "google/code_x_glue_cc_clone_detection_big_clone_bench", trust_remote_code=True
    )
    funcs: dict[str, str] = {}
    for split, out_name in (("train", "train.txt"), ("validation", "valid.txt"), ("test", "test.txt")):
        rows = 0
        with (DATA_ROOT / "clone" / out_name).open("w", encoding="utf-8") as fo:
            # Batched so the 900k-row train split is never materialised at once.
            for batch in ds[split].iter(batch_size=5000):
                for id1, id2, f1, f2, label in zip(
                    batch["id1"], batch["id2"], batch["func1"], batch["func2"], batch["label"]
                ):
                    funcs.setdefault(str(id1), f1)
                    funcs.setdefault(str(id2), f2)
                    fo.write(f"{id1}\t{id2}\t{int(label)}\n")
                    rows += 1
        print(f"[prepare] clone/{out_name} ({rows:,} pairs)")

    with (DATA_ROOT / "clone" / "data.jsonl").open("w", encoding="utf-8") as fo:
        for idx, func in funcs.items():
            fo.write(json.dumps({"idx": idx, "func": func}) + "\n")
    print(f"[prepare] clone/data.jsonl ({len(funcs):,} unique functions)")


def cmd_prepare(args: argparse.Namespace) -> int:
    """Stage both datasets under ``data/``.

    Clone detection ships with the upstream repo. Devign is pulled out of the
    5.4 GB archive if it is present locally; otherwise we fall back to the public
    CodeXGLUE copy on the Hugging Face Hub, which is free and needs no account.
    """
    (DATA_ROOT / "defect").mkdir(parents=True, exist_ok=True)
    (DATA_ROOT / "clone").mkdir(parents=True, exist_ok=True)

    _prepare_clone()

    wanted = {"train.jsonl", "valid.jsonl", "test.jsonl"}
    missing = {n for n in wanted if not (DATA_ROOT / "defect" / n).exists()}
    if not missing:
        print("[prepare] defect: already staged")
        return 0

    archive = Path(args.defect_zip) if args.defect_zip else DEFAULT_ZIP
    if archive.exists():
        with zipfile.ZipFile(archive) as zf:
            for name in sorted(missing):
                members = [m for m in zf.namelist() if m.endswith(f"dataset/{name}")]
                if not members:
                    print(f"[prepare] {name} not in {archive.name}")
                    continue
                with zf.open(members[0]) as fi, (DATA_ROOT / "defect" / name).open("wb") as fo:
                    shutil.copyfileobj(fi, fo, length=1 << 20)
                print(f"[prepare] defect/{name} from {archive.name}")
        return 0

    print(f"[prepare] {archive} not found; downloading Devign from the Hugging Face Hub")
    from datasets import load_dataset

    ds = load_dataset("google/code_x_glue_cc_defect_detection", trust_remote_code=True)
    for split, name in (("train", "train.jsonl"), ("validation", "valid.jsonl"), ("test", "test.jsonl")):
        with (DATA_ROOT / "defect" / name).open("w", encoding="utf-8") as fo:
            for rec in ds[split]:
                fo.write(json.dumps({"func": rec["func"], "target": int(rec["target"])}) + "\n")
        print(f"[prepare] defect/{name} ({len(ds[split])} rows)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="codetune", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare", help="stage datasets under data/")
    p.add_argument("--defect-zip", default=None, help="path to defect.zip (optional)")
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("run", help="one fine-tuning run")
    p.add_argument("--config")
    p.add_argument("--task", choices=sorted(LOADERS))
    p.add_argument("--method", choices=sorted(METHODS))
    p.add_argument("--seed", type=int)
    p.add_argument("--epochs", type=int)
    p.add_argument("--device")
    p.add_argument("--limit-train", type=int, dest="limit_train")
    p.add_argument("--limit-eval", type=int, dest="limit_eval")
    p.add_argument("--output-dir", dest="output_dir", default=None)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("grid", help="all methods x seeds for one config")
    p.add_argument("--config", required=True)
    p.add_argument("--methods", default=",".join(METHODS))
    p.add_argument("--seeds", default="42,1337,2024")
    p.add_argument("--device")
    p.add_argument("--output-dir", dest="output_dir", default=None)
    p.add_argument("--force", action="store_true", help="re-run even if a result exists")
    p.set_defaults(func=cmd_grid)

    p = sub.add_parser("aggregate", help="per-run JSONs -> summary.csv / summary.md")
    p.add_argument("--output-dir", dest="output_dir", default="results")
    p.set_defaults(func=cmd_aggregate)

    p = sub.add_parser("plot", help="render figures from the summary")
    p.add_argument("--output-dir", dest="output_dir", default="results")
    p.set_defaults(func=cmd_plot)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
