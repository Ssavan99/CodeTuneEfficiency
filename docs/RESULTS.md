# Results

The tables and figures themselves are generated, not hand-written:

- [`results/summary.md`](../results/summary.md) — seed-averaged tables, mean ± std
- [`results/summary.csv`](../results/summary.csv) — the same data, machine-readable
- [`results/figures/`](../results/figures/) — quality-vs-cost scatters and per-method bars
- `results/<task>/<method>__seed<n>.json` — one self-describing file per run

Regenerate everything with:

```bash
python -m codetune aggregate && python -m codetune plot
```

This document is the interpretation that sits on top of them.

---

## How to read these numbers

**The scale is small and deliberately so.** 2,500 training examples, 3 epochs,
sequence length 128, on a 6 GB GTX 1660 Ti. That is what a $0 compute budget buys.
Absolute scores are therefore well below what the same methods reach on the full
splits with a 32 GB card — the 2024 study's own defect accuracy was ~65% against
Devign's ~55% majority-class baseline, and less training data moves that down, not
up.

What survives reduced scale is the **comparison**, because every method receives an
identical budget: same subset, same seeds, same epochs, same sequence length, same
optimizer. What does not survive is any claim about a method's ceiling. Read the
gaps between methods and the cost columns; do not read the absolute quality numbers
as state-of-the-art anything.

**Check the ⚠️ flag before quoting a row.** It marks a run where one seed predicted
a single class for ≥99% of inputs. On Devign, whose classes are close to balanced,
that is a training failure, not a finding about the method — and it is exactly the
failure that went unnoticed in the 2024 results (see below).

**Standard deviations are over three seeds** (42, 1337, 2024). Where a gap between
two methods is smaller than the spread on either, there is no result there. This is
the main reason to distrust small differences at this scale.

---

## What to look for

1. **The cost columns are the point.** Trainable parameters, delta checkpoint size,
   peak VRAM and wall-clock are reported for every method. The interesting question
   is not "does full fine-tuning win on quality" — it does, essentially always — but
   how much quality a method gives up per unit of cost saved.

2. **Delta checkpoint size is the honest version of PEFT's storage claim.** It counts
   only the tensors a deployment would have to ship for this task. Full fine-tuning
   ships the whole model (~476 MB at fp32); BitFit ships a few MB. This is the axis
   where PEFT's advantage is largest and least ambiguous, and it is the one the 2024
   runs never realised — every method there saved a complete ~500 MB `model.safetensors`.

3. **Peak VRAM separates less than parameter count suggests.** PEFT reduces optimizer
   state, but activations dominate memory at these batch sizes and are largely
   unchanged by freezing weights. Expect the memory column to compress far less than
   the trainable-parameter column.

4. **Cross-task behaviour.** Clone detection is a pairwise similarity judgement;
   defect detection asks a single function whether it is vulnerable. PEFT methods
   generally hold up better on the former. At sequence length 128 a clone pair is
   truncated hard — two functions sharing one window — so the clone numbers carry an
   extra caveat the defect numbers do not.

---

## Relationship to the 2024 study

The original CSCE 962 project asked the same question on the same model and datasets
at much larger scale (full splits, 32 GB V100). Its numbers and raw artifacts are
preserved in [`provenance/`](../provenance/), and
[`provenance/README.md`](../provenance/README.md) documents them in detail.

Direct comparison of absolute values is **not** meaningful — different training set
sizes, epoch counts and sequence lengths. What is comparable is experimental design,
and this rebuild changes it in five ways:

| | 2024 | This rebuild |
|---|---|---|
| Budget per method | varied (5, 5, 15, 6 epochs) | identical for all methods |
| Seeds | unrecorded, 2 repeats averaged | 42 / 1337 / 2024, spread reported |
| Training set | stated as full; some runs used 100 examples | recorded per run in the result JSON |
| Failed runs | averaged in silently | flagged via majority-class rate |
| Cost | not measured | params, VRAM, wall-clock, delta checkpoint |

The last two matter most. A LoRA row in the 2024 defect table shows precision 0.2828
with recall exactly 0.5000 — the arithmetic signature of a classifier emitting one
class for every input. Nothing in that pipeline flagged it, so it was reported as a
result about LoRA. Here that condition is computed, stored, and surfaced in the
table.

**Where the rebuild disagrees with the 2024 conclusions, it says so.** If a method
ranking here differs from the paper's, the reduced scale is the first explanation to
consider, not evidence that the paper was wrong — but the paper's own ranking rested
partly on rows trained on 100 examples, so neither should be treated as settled.

## Honest limitations

- **Reduced scale.** ~11% of Devign's training set; a fraction of BigCloneBench.
  Conclusions about absolute quality do not transfer.
- **Sequence length 128** truncates aggressively, and asymmetrically across the two
  tasks (a clone pair loses more than a single function).
- **One base model.** CodeBERT-base only. Nothing here speaks to CodeT5, PLBART, or
  decoder-only code models.
- **Three seeds** is enough to expose instability, not enough for a confidence
  interval anyone should lean on.
- **One GPU, one machine.** Wall-clock and VRAM are specific to a GTX 1660 Ti with
  no tensor cores; on a card with them, fp16 changes the time column substantially
  and would compress the gaps between methods.
- **`peak_memory_mb` is allocator-tracked**, so it excludes the CUDA context
  (several hundred MB). `peak_reserved_mb` is the closer proxy for "will this fit."

To run it at a scale where the absolute numbers mean more, use
[`notebooks/run_on_free_gpu.ipynb`](../notebooks/run_on_free_gpu.ipynb) on a free
Colab or Kaggle T4 — 16 GB and tensor cores, still $0.
