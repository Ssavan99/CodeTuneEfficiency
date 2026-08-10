"""Subsampling must be deterministic and label-preserving — a shifted positive
rate would move F1 on its own and be mistaken for a method effect."""

import pytest

from codetune.data import Example, build_split, subsample


def make_examples(n_pos=200, n_neg=800):
    return [Example(f"pos_{i}", None, 1) for i in range(n_pos)] + [
        Example(f"neg_{i}", None, 0) for i in range(n_neg)
    ]


def test_subsample_is_deterministic_for_a_seed():
    examples = make_examples()
    a = subsample(examples, 100, seed=42)
    b = subsample(examples, 100, seed=42)
    assert [e.text_a for e in a] == [e.text_a for e in b]


def test_different_seeds_give_different_subsets():
    examples = make_examples()
    a = subsample(examples, 100, seed=42)
    b = subsample(examples, 100, seed=1337)
    assert [e.text_a for e in a] != [e.text_a for e in b]


def test_subsample_returns_exactly_the_requested_size():
    examples = make_examples()
    for limit in (1, 37, 100, 999):
        assert len(subsample(examples, limit, seed=0)) == limit


def test_subsample_preserves_the_positive_rate():
    examples = make_examples(n_pos=200, n_neg=800)  # 20% positive
    picked = subsample(examples, 200, seed=7)
    rate = sum(e.label for e in picked) / len(picked)
    assert rate == pytest.approx(0.20, abs=0.02)


def test_no_limit_returns_everything_unchanged():
    examples = make_examples()
    assert subsample(examples, None, seed=0) == examples
    assert subsample(examples, 10_000, seed=0) == examples


def test_unknown_task_is_rejected():
    with pytest.raises(ValueError, match="unknown task"):
        build_split("not_a_task", "train", tokenizer=None, max_length=8)
