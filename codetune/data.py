"""Dataset loading for the two CodeXGLUE code-understanding tasks.

Both tasks are read from plain JSONL/TXT on disk under ``data/`` (see
``codetune prepare``). Everything is local, so no network access and no dataset
loading scripts are involved at training time.

Subsampling is deterministic given a seed, and the resulting subset size is
recorded in every result file so a reduced-scale run can never be mistaken for a
full one.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"

#: Filename of each split, per task. Clone detection is pair-shaped and keeps its
#: functions in one shared ``data.jsonl`` indexed by id.
_DEFECT_FILES = {"train": "train.jsonl", "validation": "valid.jsonl", "test": "test.jsonl"}
_CLONE_FILES = {"train": "train.txt", "validation": "valid.txt", "test": "test.txt"}


@dataclass
class Example:
    """One classification example. ``text_b`` is None for single-sequence tasks."""

    text_a: str
    text_b: str | None
    label: int


def _read_jsonl(path: Path):
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_defect(split: str, root: Path = DATA_ROOT) -> list[Example]:
    """Devign defect detection (C). One function in, binary vulnerable label out."""
    path = root / "defect" / _DEFECT_FILES[split]
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m codetune prepare` first (see README)."
        )
    return [Example(rec["func"], None, int(rec["target"])) for rec in _read_jsonl(path)]


def load_clone(
    split: str, root: Path = DATA_ROOT, limit: int | None = None, seed: int = 42
) -> tuple[list[Example], int]:
    """BigCloneBench clone detection (Java). A function pair in, binary clone label out.

    The index files are large — the train split alone lists ~900 k pairs — while a
    run typically uses a few thousand. Subsampling operates on the index lines,
    whose third column is the label, so only the surviving ids are ever resolved
    to function text. Materialising all 900 k pairs first and discarding 99.7% of
    them costs seconds and a few hundred MB per run, repeated for every cell of
    the grid.
    """
    base = root / "clone"
    index_path, funcs_path = base / _CLONE_FILES[split], base / "data.jsonl"
    for path in (index_path, funcs_path):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run `python -m codetune prepare` first (see README)."
            )

    pairs: list[tuple[str, str, int]] = []
    with index_path.open(encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) == 3:
                pairs.append((parts[0], parts[1], int(parts[2])))

    n_available = len(pairs)
    if limit is not None and limit < n_available:
        by_label: dict[int, list[tuple[str, str, int]]] = {}
        for pair in pairs:
            by_label.setdefault(pair[2], []).append(pair)
        rng = random.Random(seed)
        picked: list[tuple[str, str, int]] = []
        spare: list[tuple[str, str, int]] = []
        for label in sorted(by_label):
            pool = rng.sample(by_label[label], len(by_label[label]))
            take = limit * len(pool) // n_available
            picked.extend(pool[:take])
            spare.extend(pool[take:])
        rng.shuffle(spare)
        picked.extend(spare[: limit - len(picked)])
        rng.shuffle(picked)
        pairs = picked

    funcs = {str(rec["idx"]): rec["func"] for rec in _read_jsonl(funcs_path)}
    examples, missing = [], 0
    for id1, id2, label in pairs:
        # A handful of ids in the index files have no entry in data.jsonl.
        if id1 not in funcs or id2 not in funcs:
            missing += 1
            continue
        examples.append(Example(funcs[id1], funcs[id2], label))
    if missing:
        print(f"[data] clone/{split}: skipped {missing} pairs with unresolved function ids")
    return examples, n_available


def _load_defect_counted(
    split: str, root: Path = DATA_ROOT, limit: int | None = None, seed: int = 42
) -> tuple[list[Example], int]:
    examples = load_defect(split, root=root)
    n_available = len(examples)
    return subsample(examples, limit, seed), n_available


LOADERS = {"defect": _load_defect_counted, "clone": load_clone}


def subsample(examples: list[Example], limit: int | None, seed: int) -> list[Example]:
    """Take a deterministic, label-stratified subset.

    Stratifying matters for clone detection, whose splits are not balanced; a
    naive random slice can shift the positive rate enough to move F1 on its own.
    """
    if limit is None or limit >= len(examples):
        return examples

    by_label: dict[int, list[Example]] = {}
    for ex in examples:
        by_label.setdefault(ex.label, []).append(ex)

    # Flooring the per-label quota can never overshoot, so the only correction
    # needed is a top-up of fewer than `len(by_label)` items, drawn from the
    # proportional leftovers. Rounding instead would require a symmetric
    # trim-or-top-up fix-up whose top-up ignores labels and drifts the rate.
    rng = random.Random(seed)
    picked: list[Example] = []
    spare: list[Example] = []
    for label in sorted(by_label):
        pool = rng.sample(by_label[label], len(by_label[label]))
        take = limit * len(pool) // len(examples)
        picked.extend(pool[:take])
        spare.extend(pool[take:])

    rng.shuffle(spare)
    picked.extend(spare[: limit - len(picked)])
    rng.shuffle(picked)
    return picked


class TokenizedDataset(Dataset):
    """Examples tokenized up front to a fixed length.

    Pre-tokenizing costs one pass and a few hundred MB, and keeps the GPU fed on
    Windows, where DataLoader worker processes are expensive to spawn.
    """

    def __init__(self, examples: list[Example], tokenizer, max_length: int):
        self.labels = torch.tensor([ex.label for ex in examples], dtype=torch.long)
        text_a = [ex.text_a for ex in examples]
        text_b = [ex.text_b for ex in examples] if examples and examples[0].text_b is not None else None
        # Encoded in chunks: the fast tokenizer builds an Encoding object per
        # example (offsets, word ids, masks) before the tensor conversion, and
        # none of that survives. In one shot over ~22 k examples that transient
        # peaked at ~2.3 GB of host RAM — 22x the size of the output it produces.
        id_chunks, mask_chunks = [], []
        for start in range(0, len(text_a), 1000):
            stop = start + 1000
            encoded = tokenizer(
                text_a[start:stop],
                text_b[start:stop] if text_b else None,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            id_chunks.append(encoded["input_ids"].to(torch.int32))
            mask_chunks.append(encoded["attention_mask"].to(torch.bool))
        self.input_ids = torch.cat(id_chunks) if id_chunks else torch.empty(0, max_length, dtype=torch.int32)
        self.attention_mask = (
            torch.cat(mask_chunks) if mask_chunks else torch.empty(0, max_length, dtype=torch.bool)
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # Stored narrow to halve resident memory; the model wants int64.
        return {
            "input_ids": self.input_ids[idx].long(),
            "attention_mask": self.attention_mask[idx].long(),
            "labels": self.labels[idx],
        }


def build_split(
    task: str,
    split: str,
    tokenizer,
    max_length: int,
    limit: int | None = None,
    seed: int = 42,
    root: Path = DATA_ROOT,
) -> tuple[TokenizedDataset, dict]:
    """Load, subsample and tokenize one split. Returns the dataset and its stats."""
    if task not in LOADERS:
        raise ValueError(f"unknown task {task!r}; expected one of {sorted(LOADERS)}")

    examples, n_available = LOADERS[task](split, root=root, limit=limit, seed=seed)

    positives = sum(ex.label for ex in examples)
    stats = {
        "split": split,
        "n_available": n_available,
        "n_used": len(examples),
        "subsampled": len(examples) < n_available,
        "positive_rate": round(positives / len(examples), 4) if examples else 0.0,
    }
    return TokenizedDataset(examples, tokenizer, max_length), stats
