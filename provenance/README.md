# Provenance — the original 2024 Master's work

This folder preserves the CSCE 962 course project (University of Nebraska–Lincoln, spring
2024) that gave this repository its name, so that the 2026 rebuild in [`codetune/`](../codetune)
can be compared against it honestly.

## What the 2024 project was

**Paper:** *Exploring Efficient Fine-Tuning Strategies for Pre-trained Code Models* — Savan
Patel. A scoped-down replication of Liu, Sha & Peng (ASE 2023), run with that paper's own
released code (see [`../THIRD_PARTY.md`](../THIRD_PARTY.md)).

CodeBERT-base (125 M) fine-tuned four ways — full, LoRA, BitFit, Parallel Adapter — on
Devign (defect detection, C) and BigCloneBench (clone detection, Java). Trained on a
32 GB NVIDIA V100 on UNL's Slurm cluster.

## What is here

| File | What it is |
|---|---|
| `original-2024-edits.patch` | The complete working-tree diff of the 2024 modifications to the upstream code, plus the two new `run-bitfit.sh` Slurm scripts. Never committed at the time. HF token redacted. |
| `original-runs/*__all_results.json`, `*__test_results.json`, `*__train_results.json` | Final metrics per run, as written by HuggingFace `Trainer`. |
| `original-runs/*__trainer_state.json` | Full per-epoch training curves. |
| `original-runs/*__config.json` | Model config per run, showing the PETL settings (`attn_mode`, `ffn_mode`, bottleneck dims). |
| `original-runs/slurm-*.out` | Raw cluster job logs. |

Extracted from a 5.4 GB `defect.zip` archive. The six ~500 MB model checkpoints it also
contained are **not** included — only the kilobyte-scale evidence.

**Only the defect task was archived.** No clone-detection checkpoints or logs survive; those
results exist solely as numbers in the paper.

## What the 2024 work actually contributed

The upstream framework shipped Adapter, LoRA, Prefix, Parallel-Adapter and MHM.
**BitFit was added by this project** — `petl/petl_enc_model.py` widened to accept
`BertConfig`, plus new `run-bitfit.sh` harnesses for both tasks. BitFit is not studied in
Liu et al., so it is the one genuinely additive element.

The defect-detection data path was also rewritten: upstream's `defect/run.py` tokenized a
*pair* of code snippets (clone-detection shaped) and wrote the label under `label` rather
than the `labels` key HuggingFace `Trainer` actually reads. Both were real bugs, and both
were fixed. See the patch.

## Reading the artifacts alongside the paper

The archived JSONs reproduce the paper's Table 3 exactly:

| method | epochs | `train_samples` | test acc | test F1 | test prec | test rec |
|---|---|---|---|---|---|---|
| full | 5 | **100** | 0.6508 | 0.6368 | 0.6435 | 0.6360 |
| bitfit | 5 | **100** | 0.5659 | 0.3735 | 0.5413 | 0.5019 |
| lora | 15 | 21 854 | 0.5655 | 0.3612 | 0.2828 | 0.5000 |
| parallel-adapter | 6 | 21 854 | 0.5878 | 0.5229 | 0.5761 | 0.5489 |

Several things in this table are worth knowing before the numbers are quoted:

1. **`train_samples = 100` for the `full` and `bitfit` rows.** Those two runs — including
   the full fine-tuning baseline every other method is compared against — were trained on
   100 examples, not the 21 854 the paper states. They are not comparable to the LoRA and
   Parallel-Adapter rows beside them.
   *Evidence:* `original-runs/…am_none.ao_none.fm_none…__all_results.json` and
   `…am_bitfit…__all_results.json`.
2. **Epoch budgets differ per method** — 5, 5, 15, 6 — while the paper describes a uniform
   "10 epochs with early stopping". A method given 15 epochs and one given 5 are not on
   equal footing. *Evidence:* the `ne<N>` field in each run directory name, corroborated by
   `__trainer_state.json`.
3. **LoRA's precision of 0.2828 with recall of exactly 0.5000** is the arithmetic signature
   of a classifier that emits a single class for every input: macro-averaged recall over
   two classes is exactly 0.5 for a constant predictor. That run did not converge.
4. **No seed variance.** All archived runs are `seed42`; the paper reports an average over
   two repeats, without seeds or spread. On Devign, run-to-run noise exceeds several of the
   gaps the paper interprets.
5. **No efficiency measurements**, despite efficiency being the subject — no trainable
   parameter counts, peak memory, wall-clock, or checkpoint sizes appear anywhere.
6. **Each checkpoint is a full ~500 MB `model.safetensors`.** The PETL layers are baked into
   a complete state dict rather than saved as a small adapter, so the storage advantage PEFT
   is supposed to provide was never actually realised.

None of this makes the project worthless — it makes it unfinished. The rebuild addresses
each point directly; see the root [`README.md`](../README.md).

## Security note

The 2024 working tree contained a hardcoded Hugging Face access token in four `run.py`
files. It was **never committed** (verified with `git log --all -S`) and therefore never
reached GitHub. It is redacted in `original-2024-edits.patch`, and the token has been
revoked.
