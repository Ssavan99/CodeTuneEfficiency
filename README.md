# CodeTuneEfficiency

**What does parameter-efficient fine-tuning actually cost you, and what does it buy you, on code models?**

A reproducible benchmark of four fine-tuning strategies for [CodeBERT](https://huggingface.co/microsoft/codebert-base)
on two CodeXGLUE code-understanding tasks. Every method gets an identical budget, every
number is averaged over three seeds with its spread reported, and — unlike most write-ups
of this comparison — **the cost side of the trade-off is measured, not asserted**.

It runs end to end on a single 6 GB consumer GPU. There is also a
[notebook](notebooks/run_on_free_gpu.ipynb) for Colab's free T4 or Kaggle's free weekly
GPU hours, so anyone can reproduce it without paying for compute.

<!-- RESULTS:START -->

## Results

### clone — 600 train / 1,000 test, 2 epochs

| Method | Seeds | Accuracy | Macro F1 | Positive F1 | Trainable | Delta ckpt | Peak VRAM | Train time |
|---|---|---|---|---|---|---|---|---|
| `full` | 3 | 62.37 ± 15.58 | 53.21 ± 9.81 | 34.09 ± 3.78 | 100.000% (124,647,170) | 475.49 MB | 3,432 MB | 2.9 min |
| `bitfit` ⚠️ | 3 | 19.37 ± 5.80 | 18.68 ± 6.50 | 24.89 ± 1.04 | 0.560% (694,274) | 2.65 MB | 3,432 MB | 1.9 min |
| `lora` ⚠️ | 3 | 64.27 ± 38.77 | 38.48 ± 16.91 | 10.69 ± 12.87 | 0.710% (887,042) | 3.38 MB | 3,458 MB | 2.0 min |
| `parallel_adapter` ⚠️ | 3 | 72.27 ± 12.59 | 52.75 ± 4.98 | 23.72 ± 18.32 | 0.720% (896,450) | 3.42 MB | 3,458 MB | 1.9 min |

⚠️ = at least one seed predicted a single class for ≥99% of inputs, which is a training failure rather than a result about the method.

### defect — 600 train / 1,000 test, 2 epochs

| Method | Seeds | Accuracy | Macro F1 | Positive F1 | Trainable | Delta ckpt | Peak VRAM | Train time |
|---|---|---|---|---|---|---|---|---|
| `full` ⚠️ | 3 | 54.57 ± 0.61 | 37.32 ± 1.51 | 4.44 ± 2.91 | 100.000% (124,647,170) | 475.49 MB | 3,432 MB | 2.9 min |
| `bitfit` ⚠️ | 3 | 54.10 ± 0.00 | 35.49 ± 0.67 | 0.85 ± 1.47 | 0.560% (694,274) | 2.65 MB | 3,432 MB | 2.0 min |
| `lora` ⚠️ | 3 | 54.13 ± 0.06 | 35.82 ± 1.03 | 1.54 ± 2.30 | 0.710% (887,042) | 3.38 MB | 3,458 MB | 2.0 min |
| `parallel_adapter` ⚠️ | 3 | 54.10 ± 0.10 | 35.17 ± 0.15 | 0.14 ± 0.25 | 0.720% (896,450) | 3.42 MB | 3,458 MB | 1.9 min |

⚠️ = at least one seed predicted a single class for ≥99% of inputs, which is a training failure rather than a result about the method.

<!-- RESULTS:END -->

---

## Why this exists

This repository began life as a copy of the artifact release for Liu, Sha & Peng,
*An Empirical Study of Parameter-Efficient Fine-Tuning Methods for Pre-Trained Code Models*
(ASE 2023) — see [THIRD_PARTY.md](THIRD_PARTY.md). That upstream code is still here,
unmodified, under `clone/`, `defect/`, `petl/` and friends.

On top of it, a 2024 Master's course project (CSCE 962, University of Nebraska–Lincoln)
used that framework to compare full fine-tuning, LoRA, BitFit and Parallel Adapters on
Devign and BigCloneBench, adding BitFit — which the upstream study does not cover — to the
framework itself. That work is preserved in [`provenance/`](provenance/), including the
original Slurm logs and metric dumps.

**This repository is the 2026 rebuild of that project**: a clean, self-contained
implementation under [`codetune/`](codetune/) that anyone can run, with the experimental
design tightened and the efficiency measurements added. The original study was a
scoped-down replication of Liu et al. run on a 32 GB V100; this is a smaller, sharper,
fully reproducible version of the same question that fits on hardware people actually own.
[`provenance/README.md`](provenance/README.md) documents what changed and why.

## The four methods

| Method | What trains | Idea |
|---|---|---|
| `full` | everything (100%) | the baseline |
| `bitfit` | bias vectors + classifier | [Ben Zaken et al., ACL 2022](https://aclanthology.org/2022.acl-short.1/) |
| `lora` | rank-8 updates on attention query/value | [Hu et al., ICLR 2022](https://arxiv.org/abs/2106.09685) |
| `parallel_adapter` | a bottleneck MLP alongside each FFN block | [He et al., ICLR 2022](https://arxiv.org/abs/2110.04366) |

`bitfit` and `parallel_adapter` are implemented directly in [`codetune/methods.py`](codetune/methods.py);
`lora` uses Hugging Face `peft`. The parallel adapter's up-projection is zero-initialised,
so it is exactly the identity at step 0 and the pre-trained function is preserved — a
property [asserted in the tests](tests/test_methods.py).

## What is measured

Four cost numbers accompany every accuracy number:

- **trainable parameters** — what the optimizer updates
- **peak GPU memory** — whether it fits on the card you own
- **wall-clock training time** — what a run costs
- **delta checkpoint size** — bytes you must store and ship *per task*, counting only the
  trainable tensors

That last one is the honest form of PEFT's storage claim. Saving a full model per method,
which is the default behaviour of most training scripts, throws the saving away entirely.

Reported quality metrics are accuracy, macro precision/recall/F1, and positive-class F1.
Both macro **and** positive-class F1 appear because the pair identifies a collapsed run at
a glance: a classifier that predicts one class for every input scores exactly 0.50 macro
recall and 0.00 positive F1. Runs that collapse are flagged rather than quietly averaged in.

## Quickstart

```bash
git clone https://github.com/Ssavan99/CodeTuneEfficiency.git && cd CodeTuneEfficiency
python -m venv .venv && .venv/Scripts/activate   # Linux/macOS: source .venv/bin/activate
pip install -r requirements-dev.txt
```

```bash
python -m codetune prepare
```

```bash
python -m codetune run --config configs/smoke.yaml
```

The smoke config is a CPU-only sanity check that finishes in a couple of minutes. Its
numbers are meaningless at that size — it exists to prove the pipeline works before you
spend GPU time.

Then the real thing:

```bash
python -m codetune grid --config configs/defect.yaml
```

```bash
python -m codetune grid --config configs/clone.yaml
```

```bash
python -m codetune aggregate && python -m codetune plot
```

`grid` skips any run that already has a result file, so an interrupted session resumes
where it left off rather than starting over.

### Tests

```bash
pytest
```

The unit tests build a tiny randomly-initialised RoBERTa in-process, so they run offline
in seconds and never download a checkpoint. The end-to-end test skips cleanly if the
datasets or the base model are unavailable.

## Experimental design

| | |
|---|---|
| Base model | `microsoft/codebert-base` (125 M) |
| Tasks | Devign defect detection (C) · BigCloneBench clone detection (Java) |
| Budget | identical for every method — same epochs, data, sequence length, seed |
| Seeds | 42, 1337, 2024 · mean ± std reported |
| Learning rate | 5e-5 full, 1e-4 PEFT |
| Precision | fp16 (Turing — no bf16) |
| Early stopping | off, deliberately |

**Equal budgets are the point.** Comparing methods that received different epoch counts
measures the schedule, not the method. Early stopping is disabled for the same reason: it
silently hands more optimisation to whichever method happens to keep improving.

**Scale is reduced and stated, not hidden.** The clone-detection split is subsampled to fit
a $0 compute budget; the exact sizes are in `configs/clone.yaml` and recorded in every
result JSON. A result anyone can verify is worth more than a larger number nobody can
re-run.

## Layout

```
codetune/       the benchmark: data, methods, cost accounting, training, reporting
configs/        smoke (CPU) · defect · clone
tests/          offline unit tests + an end-to-end smoke test
notebooks/      free-GPU runner for Colab / Kaggle
results/        per-run JSONs, summary.csv, summary.md, figures/
provenance/     the 2024 Master's project preserved as evidence
docs/           results discussion
clone/ defect/ petl/ …   upstream code, unmodified (see THIRD_PARTY.md)
```

## Reading the results

Full discussion, including where this rebuild disagrees with the 2024 numbers, is in
[docs/RESULTS.md](docs/RESULTS.md).

## Licence

MIT — see [LICENSE](LICENSE). Upstream code retains its own MIT terms; attribution is in
[THIRD_PARTY.md](THIRD_PARTY.md).
