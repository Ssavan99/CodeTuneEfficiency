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


def load_clone(split: str, root: Path = DATA_ROOT) -> list[Example]:
    """BigCloneBench clone detection (Java). A function pair in, binary clone label out."""
    base = root / "clone"
    index_path, funcs_path = base / _CLONE_FILES[split], base / "data.jsonl"
    for path in (index_path, funcs_path):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run `python -m codetune prepare` first (see README)."
            )

    funcs = {str(rec["idx"]): rec["func"] for rec in _read_jsonl(funcs_path)}

    examples, missing = [], 0
    with index_path.open(encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) != 3:
                continue
            id1, id2, label = parts
            # A handful of ids in the index files have no entry in data.jsonl.
            if id1 not in funcs or id2 not in funcs:
                missing += 1
                continue
            examples.append(Example(funcs[id1], funcs[id2], int(label)))
    if missing:
        print(f"[data] clone/{split}: skipped {missing} pairs with unresolved function ids")
    return examples


LOADERS = {"defect": load_defect, "clone": load_clone}


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

    rng = random.Random(seed)
    picked: list[Example] = []
    for label in sorted(by_label):
        pool = by_label[label]
        take = round(limit * len(pool) / len(examples))
        picked.extend(rng.sample(pool, min(take, len(pool))))

    # Rounding can leave us a couple short or long of the requested size.
    rng.shuffle(picked)
    if len(picked) > limit:
        picked = picked[:limit]
    elif len(picked) < limit:
        chosen = {id(ex) for ex in picked}
        remainder = [ex for ex in examples if id(ex) not in chosen]
        picked.extend(rng.sample(remainder, min(limit - len(picked), len(remainder))))
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
        encoded = tokenizer(
            text_a,
            text_b,
            padding="max_length",
            truncation="longest_first" if text_b else True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.input_ids = encoded["input_ids"]
        self.attention_mask = encoded["attention_mask"]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
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

    examples = LOADERS[task](split, root=root)
    full_size = len(examples)
    examples = subsample(examples, limit, seed)

    positives = sum(ex.label for ex in examples)
    stats = {
        "split": split,
        "n_available": full_size,
        "n_used": len(examples),
        "subsampled": len(examples) < full_size,
        "positive_rate": round(positives / len(examples), 4) if examples else 0.0,
    }
    return TokenizedDataset(examples, tokenizer, max_length), stats
