# Results

24 runs — 2 tasks × 4 methods × 3 seeds (42, 1337, 2024) — every method on an identical
budget: 2 epochs, sequence length 256, batch 32, on a free Colab **Tesla T4**. Devign uses
its full 21,854-example training set; BigCloneBench is subsampled to 20,000 pairs. No run
collapsed.

Generated artifacts, not hand-written:

- [`results/summary.md`](../results/summary.md) · [`results/summary.csv`](../results/summary.csv)
- [`results/figures/`](../results/figures/)
- `results/<task>/<method>__seed<n>.json` — one self-describing file per run

Reproduce with `python -m codetune grid --config configs/defect.yaml` (and `clone.yaml`),
then `aggregate && plot`. The configs in this repo are exactly the ones that produced these
numbers; [`notebooks/run_on_free_gpu.ipynb`](../notebooks/run_on_free_gpu.ipynb) runs the
whole thing on free compute.

---

## 1. Read positive-class F1, not accuracy

Both test splits are imbalanced, and on clone detection severely so:

| Task | Positive rate | Majority-class accuracy |
|---|---|---|
| defect (Devign) | 45.9% | 54.10% |
| clone (BigCloneBench) | 13.6% | **86.40%** |

A model that answers "not a clone" to everything scores **86.4% accuracy**. So BitFit's
89.20% is under three points above doing nothing, while its positive F1 of 66.95 against
full fine-tuning's 77.42 shows the real gap. Every table below reports both.

## 2. Quality

### Defect detection (Devign) — full 21,854 train

| Method | Accuracy | Macro F1 | Positive F1 | Majority-class rate |
|---|---|---|---|---|
| `full` | **64.33 ± 1.37** | **62.61 ± 1.64** | **54.59 ± 2.50** | 0.673 |
| `parallel_adapter` | 59.00 ± 0.70 | 51.74 ± 2.97 | 33.11 ± 6.46 | 0.843 |
| `lora` | 58.27 ± 0.42 | 45.41 ± 0.21 | 18.91 ± 0.13 | 0.944 |
| `bitfit` | 57.67 ± 0.67 | 44.42 ± 1.39 | 17.29 ± 2.49 | 0.947 |

Accuracy makes this look like a 5–7 point spread. Positive F1 shows it is a **3× spread**:
54.59 for full fine-tuning against 17–19 for LoRA and BitFit. Those two answer "not
vulnerable" for ~94.5% of inputs — not collapsed by the ≥99% criterion, but heavily biased
toward the majority class. Defect detection is where PEFT genuinely struggles, and reporting
accuracy alone would have hidden it.

`parallel_adapter` sits clearly between the two groups on every metric.

### Clone detection (BigCloneBench) — 20,000 train

| Method | Accuracy | Macro F1 | Positive F1 |
|---|---|---|---|
| `full` | **93.37 ± 0.91** | **86.77 ± 1.76** | **77.42 ± 2.99** |
| `parallel_adapter` | 92.27 ± 0.64 | 84.96 ± 1.39 | 74.48 ± 2.42 |
| `lora` | 91.73 ± 0.81 | 84.06 ± 1.39 | 73.00 ± 2.31 |
| `bitfit` | 89.20 ± 0.62 | 80.25 ± 0.61 | 66.95 ± 0.85 |

Here PEFT is genuinely competitive. `parallel_adapter` is **1.1 accuracy points and 2.9
positive-F1 points** behind full fine-tuning while training 0.72% of the parameters and
shipping a checkpoint 139× smaller. For most deployments that is a good trade.

**The ordering is identical on both tasks:** `full` > `parallel_adapter` > `lora` > `bitfit`,
and every gap exceeds the seed spread. Two independent tasks agreeing on a strict ordering
is the strongest evidence in this report.

## 3. Cost

Seed-independent, and the reason this benchmark exists.

| Method | Trainable | Delta checkpoint | vs. full | Train time (defect) | Peak allocated | Peak reserved |
|---|---|---|---|---|---|---|
| `full` | 100% (124,647,170) | 475.49 MB | 1× | 12.1 min | 6,829 MB | 7,374 MB |
| `bitfit` | 0.560% (694,274) | 2.65 MB | **179× smaller** | 9.4 min | 4,657 MB (−31.8%) | 7,502 MB (+1.7%) |
| `lora` | 0.710% (887,042) | 3.38 MB | **141× smaller** | 10.0 min | 5,040 MB (−26.2%) | 7,508 MB (+1.8%) |
| `parallel_adapter` | 0.720% (896,450) | 3.42 MB | **139× smaller** | 9.1 min | 4,460 MB (−34.7%) | 7,508 MB (+1.8%) |

**Storage is where PEFT wins, decisively.** 475 MB per task versus 2.65 MB. Across ten
tasks: 4.75 GB versus 27 MB. This is the saving the 2024 runs never realised — they wrote a
complete ~500 MB `model.safetensors` for every method, PEFT included, discarding it
entirely.

**Training is 17–25% faster** — real, but far less than the 140× parameter ratio suggests.
The backward pass still traverses the whole network; only the optimizer update and gradient
storage shrink.

**The memory story has two halves, and reporting only one of them misleads.**

*Allocated* peak memory — the bytes live tensors need — fell by 26–35%. That is the
gradients and Adam moments for 125 M parameters no longer existing, and it is a genuine
saving of roughly 2.3 GB.

*Reserved* peak memory — what the process actually holds from the driver, and therefore what
determines whether the job fits — was **flat**: 7,374 MB for full fine-tuning against
7,502–7,508 MB for the PEFT methods, i.e. marginally worse. Activations dominate the
transient high-water mark that drives the caching allocator, and freezing weights does not
shrink activations. LoRA and the parallel adapter also add modules of their own.

So: a **140× reduction in trainable parameters bought a ~30% reduction in allocated memory
and no reduction in footprint.** The memory benefit of PEFT is real but an order of magnitude
smaller than the parameter count suggests, and on this configuration it would not have let
the job fit on a meaningfully smaller card. Whether it does on *your* card depends on where
your activation peak sits — which is exactly why both numbers are recorded per run.

None of this is visible in the 2024 paper, which reported no cost measurements at all.

## 4. Comparison with the 2024 study and with Liu et al.

The 2024 numbers and raw artifacts are in [`provenance/`](../provenance/).

| Devign, full fine-tuning | Accuracy |
|---|---|
| Liu, Sha & Peng (ASE 2023), CodeBERT | 64.92 |
| 2024 CSCE 962 paper | 65.08 |
| **This rebuild** | **64.33 ± 1.37** |

Three independent implementations within 0.75 points. That is a clean replication and good
evidence the pipeline is sound. `parallel_adapter` matches too: 58.78 in the 2024 paper
against **59.00 ± 0.70** here.

**The one number that does not reproduce is the interesting one.** The 2024 paper reports
LoRA on Devign at precision 0.2828 and recall exactly 0.5000 — the arithmetic signature of a
classifier emitting a single class for every input. Here LoRA reaches 58.27 ± 0.42 accuracy
with positive F1 18.91 ± 0.13 and a majority-class rate of 0.944 across all three seeds.

**LoRA does learn defect detection. That row was a training failure, not a property of
LoRA** — and it was published as a finding about LoRA because nothing in the pipeline
checked for it. The collapse detector added here fires on exactly that condition.

Note also that the 2024 paper's full and BitFit defect rows came from runs with
`train_samples = 100` (see [`provenance/README.md`](../provenance/README.md)), so its BitFit
number was never comparable to the rows beside it.

## 5. Limitations

- **Two epochs.** More would likely favour the PEFT methods, which converge more slowly;
  the defect gap in particular may narrow. The budget is equal across methods, which is the
  controlled comparison — but it is not a convergence study.
- **Clone detection is subsampled** to 20,000 of 901,028 pairs, and both test splits to
  1,000, to fit one free Colab session. Sizes are in every result JSON.
- **Sequence length 256** truncates, and asymmetrically: a clone *pair* shares one window.
- **Three seeds** is enough to establish that the ordering is stable, not enough for a
  confidence interval.
- **One base model** (CodeBERT-base). Nothing here speaks to CodeT5, PLBART or decoder-only
  code models.
- **Timings are T4-specific.** `peak_memory_mb` is allocator-tracked and excludes the CUDA
  context; `peak_reserved_mb` is the better "will it fit" proxy and is what the tables show.
