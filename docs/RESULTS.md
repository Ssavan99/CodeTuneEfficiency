# Results

24 runs: 2 tasks × 4 methods × 3 seeds (42, 1337, 2024), every method on an identical
budget — 600 training examples, 2 epochs, sequence length 128, on a GTX 1660 Ti (6 GB).

Generated artifacts, not hand-written:

- [`results/summary.md`](../results/summary.md) · [`results/summary.csv`](../results/summary.csv)
- [`results/figures/`](../results/figures/)
- `results/<task>/<method>__seed<n>.json` — one self-describing file per run

Regenerate with `python -m codetune aggregate && python -m codetune plot`.

---

## The short version

**Cost separates the methods decisively. Quality does not separate them at all at this
budget.** Both halves of that sentence are results, and the second one is the more useful
finding for anyone planning a PEFT experiment.

---

## 1. Cost: large, exact, reproducible

These numbers are identical across seeds and independent of how much data you train on, so
the small scale costs them nothing.

| Method | Trainable | Delta checkpoint | vs. full | Train time | Peak VRAM |
|---|---|---|---|---|---|
| `full` | 100% (124,647,170) | 475.49 MB | 1× | 2.9 min | 3,432 MB |
| `bitfit` | 0.560% (694,274) | 2.65 MB | **179× smaller** | 2.0 min | 3,432 MB |
| `lora` | 0.710% (887,042) | 3.38 MB | **141× smaller** | 2.0 min | 3,458 MB |
| `parallel_adapter` | 0.720% (896,450) | 3.42 MB | **139× smaller** | 1.9 min | 3,458 MB |

**Storage is where PEFT wins, and it wins enormously.** Shipping a per-task model costs
475 MB with full fine-tuning and 2.65 MB with BitFit. For ten tasks that is 4.75 GB versus
27 MB. This is the claim the 2024 runs never actually realised — they saved a complete
~500 MB `model.safetensors` for every method, including the PEFT ones, discarding the
entire advantage.

**Training is ~33% faster.** Real, but far smaller than the parameter ratio suggests: the
backward pass still propagates through the whole network, and only the optimizer update
and gradient storage shrink.

**Peak VRAM does not improve — and for LoRA and parallel adapters it gets slightly worse.**
This is the most counterintuitive number here and it deserves emphasis: freezing 99.4% of
the parameters saved **0 MB**. Activations dominate memory at these batch sizes, and
freezing weights does not shrink activations. LoRA and the parallel adapter add 26 MB
because they add modules.

> If you reached for PEFT to make a model fit on a smaller card, this is the wrong tool.
> Reach for it to avoid storing and shipping N copies of a 500 MB model.

That distinction is not visible anywhere in the 2024 paper, because it reported no cost
measurements at all.

## 2. Quality: not resolvable at this budget

### Defect detection (Devign) — nothing learned

| Method | Accuracy | Macro F1 | Positive F1 | Collapsed |
|---|---|---|---|---|
| `full` | 54.57 ± 0.61 | 37.32 ± 1.51 | 4.44 ± 2.91 | 1/3 |
| `bitfit` | 54.10 ± 0.00 | 35.49 ± 0.67 | 0.85 ± 1.47 | 2/3 |
| `lora` | 54.13 ± 0.06 | 35.82 ± 1.03 | 1.54 ± 2.30 | 2/3 |
| `parallel_adapter` | 54.10 ± 0.10 | 35.17 ± 0.15 | 0.14 ± 0.25 | 3/3 |

**The majority-class baseline on this test split is 54.1%.** Every method lands on it.
Positive-class F1 is near zero throughout, and 8 of 12 runs predicted a single class for
≥99% of inputs. BitFit's accuracy standard deviation of exactly 0.00 is the giveaway — three
different seeds produced identical predictions, which happens when all three collapse to
the same constant output.

The honest reading: **600 examples and 2 epochs are not enough for any of these methods to
learn Devign at all.** No ranking should be drawn from this table. It is a statement about
the budget, not about the methods.

This is worth putting beside the 2024 result, where a LoRA run at the *full* 21,854-example
scale produced precision 0.2828 and recall exactly 0.5000 — the same collapse — and was
reported as a finding about LoRA. Devign is simply a hard, noisy task, and collapse is its
characteristic failure mode. The difference is that here it is detected and labelled
automatically rather than averaged into a results table.

### Clone detection (BigCloneBench) — signal, but unstable

| Method | Accuracy | Macro F1 | Positive F1 | Collapsed |
|---|---|---|---|---|
| `full` | 62.37 ± 15.58 | 53.21 ± 9.81 | 34.09 ± 3.78 | **0/3** |
| `bitfit` | 19.37 ± 5.80 | 18.68 ± 6.50 | 24.89 ± 1.04 | 2/3 |
| `lora` | 64.27 ± 38.77 | 38.48 ± 16.91 | 10.69 ± 12.87 | 2/3 |
| `parallel_adapter` | 72.27 ± 12.59 | 52.75 ± 4.98 | 23.72 ± 18.32 | 1/3 |

Clone detection is the easier task and something does get learned here. Two observations
survive the noise:

1. **Full fine-tuning is the only method that never collapsed.** 0 of 3 seeds, against 2, 2
   and 1 for the PEFT methods. At a small budget, full fine-tuning is the robust choice —
   which is consistent with the 2024 paper's conclusion, reached here by a different route.
2. **Everything else is too unstable to rank.** LoRA's accuracy spread is ±38.77 — wider
   than the gap between any two methods in the table. `parallel_adapter` has the highest
   mean accuracy (72.27%) but also a collapsed seed and a ±18.32 spread on positive F1.
   Reporting "parallel adapter beats full fine-tuning" from this would be exactly the error
   this project set out to avoid.

BitFit's 19.37% accuracy is *below chance* for a binary task. That is collapse onto the
minority class: predicting "clone" for nearly everything when only about a fifth of the
pairs are clones.

## 3. What this run actually demonstrates

The deliverable here is not a leaderboard. It is:

- **An exact, reproducible cost accounting** for four fine-tuning strategies, including the
  storage figure the original study asserted but never measured, and the memory result that
  contradicts a common assumption about PEFT.
- **Instrumentation that makes failure visible.** Collapse is detected, flagged in the
  table, and counted per method. The 2024 study had no such check, and published a
  collapsed run as a result.
- **A negative result stated plainly**: at 600 examples, none of these methods learns
  Devign, and only full fine-tuning reliably learns BigCloneBench. Knowing where the floor
  is has real value when planning experiments on a budget.

## 4. Limitations

- **Scale.** 600 of 21,854 Devign training examples; 600 BigCloneBench pairs. The quality
  numbers do not transfer to full-scale training, and no absolute claim should be drawn
  from them.
- **Sequence length 128** truncates aggressively, and asymmetrically: a clone *pair* shares
  one window, so it loses more than a single defect function does.
- **Three seeds** is enough to expose instability — and it did — but not enough for a
  confidence interval.
- **One base model.** CodeBERT-base only; nothing here speaks to CodeT5, PLBART or
  decoder-only code models.
- **One machine.** Wall-clock and VRAM are specific to a GTX 1660 Ti, a Turing GTX part with
  no tensor cores that throttles to ~88 °C under sustained load. On a card with tensor cores
  the time column would compress.
- **`peak_memory_mb` is allocator-tracked** and excludes the CUDA context (several hundred
  MB); `peak_reserved_mb` is the better proxy for "will this fit".

## 5. The obvious next step

Run the identical grid at a scale where the quality columns mean something —
[`notebooks/run_on_free_gpu.ipynb`](../notebooks/run_on_free_gpu.ipynb) on a free Colab or
Kaggle T4 uses the full Devign training set, batch 32 and sequence length 256, and still
costs nothing. The cost conclusions in §1 will not change; the quality conclusions in §2
should be expected to.
