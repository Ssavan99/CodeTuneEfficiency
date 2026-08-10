# FINDINGS — CodeTuneEfficiency

**Date:** 2026-08-10 · **Branch:** `feat/peft-code-benchmark` · **Status:** Phase 3 complete (24/24 runs)

This document records what actually survives of the CSCE 962 Master's project
"Exploring Efficient Fine-Tuning Strategies for Pre-trained Code Models" (Savan Patel,
UNL, spring 2024), the evidence behind that conclusion, and the verdict that drives
[PLAN.md](PLAN.md).

---

## VERDICT: B — partial, but the ideas hold value. Implement them properly, at honest scale.

The paper's *empirical work was really done* — there are checkpoints, Slurm logs and
metric dumps to prove it. But **none of it is on GitHub, none of it is in a commit, and
the repo that carries the project's name is a byte-for-byte copy of somebody else's
research artifact.** The public repo currently claims nothing about Savan's work because
it contains nothing of Savan's work.

The underlying question — *what does parameter-efficient fine-tuning actually cost you,
and what does it buy you, on code models* — is worth building. Verdict B, with a
deliberate scope change explained in §5.

---

## 1. What exists

### 1.1 The public repo (`Ssavan99/CodeTuneEfficiency`)

| Check | Result |
|---|---|
| Commits total | 28 |
| Commits by Savan | **0** — `git log --author` for `savan`, `Ssavan99`, `savan.cars99@gmail.com` all return empty |
| Authors | `anonymous-ase23` (26), `anonymous-pikachu` (2), 2023-04-24 → 2024-01-02 |
| Local branches | `main` only, exactly in sync with `origin/main` (`e683b57`, 0 ahead / 0 behind) |
| Stashes | none |
| Dangling / unreachable objects | none — `git fsck --lost-found --unreachable --dangling` produced no output |
| Reflog | four entries, all from the single `clone` operation |
| `.git` size | 1.5 MB |

**What it is:** a clone of `anonymous-ase23/CodeModelParameterEfficientFinetuning`, the
official artifact repo for Liu, Sha & Peng, *"An Empirical Study of Parameter-Efficient
Fine-Tuning Methods for Pre-Trained Code Models"* (ASE 2023) — the exact paper Savan's
paper replicates. It was cloned and pushed to Savan's account under a new name. It is not
a GitHub fork (no fork metadata, single `origin` remote), which is why GitHub shows it as
a repo with no contribution from the account owner.

Nothing was lost. There was never anything to lose: **no commit was ever made.**

### 1.2 Savan's real code work — uncommitted, working tree only

`git status` is dirty, and the diff is the Master's work:

```
 clone/run.py                   | 58 +++++++++++---     defect/run.py   | 34 ++++++--
 clone/run-lora.sh              | 30 ++++---            defect/run-lora.sh              | 20 ++++--
 clone/run-none.sh              | 30 ++++---            defect/run-none.sh             | 24 ++++--
 clone/run-parallel-adapter.sh  | 30 ++++---            defect/run-parallel-adapter.sh | 20 ++++--
 clone/modeling_roberta.py      |  2 +-                 defect/utils.py                |  3 +
 petl/petl_enc_model.py         |  4 +-
 + untracked: clone/run-bitfit.sh, defect/run-bitfit.sh
 − deleted:   run-MHM.sh, run-adapter.sh, run-prefix.sh   (both tasks — unused methods)
```

Substantively, this is:

- **BitFit added to the PETL framework.** `petl/petl_enc_model.py` widened from
  `RobertaConfig` to also accept `BertConfig`; new `run-bitfit.sh` Slurm scripts for both
  tasks with `attn_mode="bitfit"`. Upstream shipped Adapter / LoRA / Prefix / Parallel-Adapter /
  MHM — **BitFit is Savan's addition**, and it is the one method in the paper that is not
  in the upstream study.
- **Defect-detection data path rewritten.** Upstream `defect/run.py` tokenized a *pair*
  (`code1`, `code2`) — clone-detection shaped. Savan rewrote `preprocess_function` for
  single-sequence defect input and fixed the label key (`label` → `labels`, which is what
  HF `Trainer` actually reads). This is a genuine bug fix, not a cosmetic edit.
- **Slurm/HPC harness** for UNL's cluster (`--gres=gpu:1`, `gpu_32gb&gpu_v100`,
  `conda activate fine-tune-bert`).

**Also present, and a problem:** a hardcoded Hugging Face token
`hf_fsYP…REDACTED` in `clone/run.py`, `defect/run.py`,
`clone-POJ-104/run.py`, `NL_code_search_adv/run.py`. Verified **not** in git history
(`git log --all -S` empty) and therefore **not on GitHub** — but it is almost certainly
inside `defect.zip`, and it must be revoked. It is scrubbed before anything is committed
(Phase 1).

### 1.3 `CodeTuneEfficiency-model/` — the experiment output

Not a git repo. One file: `defect.zip`, **5,406,058,829 bytes (5.4 GB)**, 157 entries,
inspected via `zipfile` without extracting. Contents:

- **6 checkpoint directories**, one per run, each a *full* `model.safetensors` (~500 MB):
  `am_none/fm_none` (full FT), `am_bitfit`, `am_lora`, `fo_parallel` ×2 variants, plus a
  cached `microsoft/codebert-base`. Note these are complete model state dicts, not
  adapter-only files — the PETL layers are baked in, so the *storage* saving that PEFT is
  supposed to deliver was never actually realised here.
- **Slurm logs** — `slurm-6269087-bitfit.out`, `slurm-6269090-lora.out`,
  `slurm-6269091-parallel-adapter.out`, `slurm-6264801-none.out`. Real cluster runs.
- **`all_results.json` / `test_results.json` / `trainer_state.json`** per run — the
  numbers behind the paper.
- The Devign dataset and a copy of `petl/`.

Only the defect task is archived. **No clone-detection checkpoints survive anywhere** —
those results exist only as numbers in `clone metrics.xlsx` and the paper.

### 1.4 The paper and its folder

`C:\Users\savan\Desktop\UNL\sem 2 MS\CSCE 962\final project\` holds the final paper (PDF +
DOCX), the presentation, an earlier draft, the raw metric spreadsheets, the figures, and
the primary reference. Also present but unrelated: a proposal and four references for an
*abandoned* first topic (text-to-2D-game-asset generation with Stable Diffusion) — the
project pivoted between March and April 2024.

**Paper in one line:** CodeBERT-base (125 M) fine-tuned four ways — full, LoRA, BitFit,
Parallel Adapter — on Devign (defect detection, C, 21 854 train) and BigCloneBench (clone
detection, Java, 90 102 train), on a 32 GB V100, lr 1e-4 for PEFT / 5e-5 for full, batch
40, seq len 512, 10 epochs w/ early stopping, 2 repeats averaged.

Reported results:

| Clone (BigCloneBench) | Acc | P | R | F1 | | Defect (Devign) | Acc | P | R | F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| Full | 97.54 | 90.10 | 91.04 | 90.57 | | Full | 65.08 | 64.35 | 63.60 | 63.68 |
| LoRA | 86.96 | 87.96 | 88.08 | 86.95 | | LoRA | 56.55 | 28.28 | 50.00 | 36.12 |
| BitFit | 86.98 | 87.81 | 88.19 | 86.57 | | BitFit | 56.59 | 54.13 | 50.19 | 37.35 |
| Parallel-Adapter | 87.04 | 89.02 | 89.23 | 88.21 | | Parallel-Adapter | 58.78 | 57.61 | 54.89 | 52.29 |

Conclusion: full fine-tuning always wins on accuracy; PEFT is a viable cheaper
alternative; PEFT holds up better on clone detection than on defect detection.

---

## 2. Problems found in the original results

These come from reading the raw artifacts, not from second-guessing the paper.

1. **The BitFit defect row is from a 100-sample run.** `all_results.json` for the BitFit
   defect checkpoint records `train_samples = 100`; LoRA records `21854` and
   Parallel-Adapter `21854`. The full-FT sheet in `defect metrics.xlsx` also shows `100`.
   Two of the four rows in Table 3 — including the baseline everything is compared
   against — were not trained on the stated dataset. The 56.59 / 37.35 BitFit row and the
   65.08 full row are not comparable to the LoRA and Parallel-Adapter rows beside them.
2. **Unequal epoch budgets, presented as a controlled comparison.** From
   `trainer_state.json`: BitFit 5 epochs, Parallel-Adapter 6, LoRA 15. The paper states
   "10 epochs with early stopping" uniformly. A method that got 15 epochs and one that
   got 5 are not on equal footing.
3. **LoRA's defect precision of 28.28 with recall exactly 50.00** is the signature of a
   classifier that predicts a single class — macro-averaged, a constant predictor scores
   exactly 0.5 recall. That is a training failure being reported as a result, not a
   finding about LoRA.
4. **No seeds, no variance.** "Repeated twice and averaged" with no seed record and no
   spread reported. On Devign, run-to-run noise is comfortably larger than several of the
   gaps the paper draws conclusions from.
5. **A paper about efficiency reports no efficiency numbers.** Every table is
   accuracy/P/R/F1. There is no trainable-parameter count, no peak memory, no wall-clock,
   no checkpoint size — the entire premise is a cost/benefit trade-off and the cost column
   is missing. (And per §1.3, the saved artifacts show the storage saving was never
   actually obtained.)
6. **Numbers diverge from the study being replicated.** Liu et al. report CodeBERT +
   Adapter at F1 94.70 on BigCloneBench; this paper's best PEFT is 88.21. Full FT: 94.05
   vs 90.57. A 6-point gap on the same model, data and protocol is unexplained.

None of this makes the project worthless. It makes it *unfinished* — and every one of
these six is a concrete, cheap thing to fix.

---

## 3. Is the contribution novel?

**No — and the paper does not claim otherwise.** It is a scoped-down replication of Liu
et al. (ASE 2023), run with Liu et al.'s own code, using hyperparameters the paper
explicitly says it copied from them ("This paper followed the experimental setup of Liu
et al. [3]"). Liu et al. cover 4 base models × 4 tasks × 5 PEFT methods plus low-resource,
cross-language and cross-project studies; this paper covers 1 model × 2 tasks × 3 PEFT
methods.

The one genuinely additive element is **BitFit**, which Liu et al. do not study and which
Savan implemented into their framework himself. That is a real contribution, and it is
also the row §2.1 shows was trained on 100 samples.

Calling this novel research in a portfolio would not survive contact with an interviewer
who opens the repo. Calling it *reproduction engineering* — and doing it better than the
original — will.

---

## 4. What is missing

- Any commit, branch, tag or PR authored by Savan. Anywhere.
- Any clone-detection artifact beyond the spreadsheet numbers.
- Any dependency manifest — no `requirements.txt`, no environment file, no lockfile.
- Any test, CI, or entry point that a stranger could run.
- Any statement of provenance. The README describes Liu et al.'s work in the first person
  ("This repository contains source code and data for the paper…"), on a repo named after
  Savan's project. Left as-is, that reads as passing off someone else's artifact.
- A LICENSE file, despite the README asserting MIT.

---

## 5. Why the plan changes scope

Reproducing the paper as written needs a 32 GB V100, 90 k clone pairs at sequence length
512, and 10-epoch budgets. The available hardware is a **GTX 1660 Ti with 6 GB of VRAM**,
and the budget is **$0**. Chasing the original scale would mean either spending money or
faking it.

So the rebuild targets a different and, for a portfolio, better claim:

> A reproducible PEFT benchmark for code models that **measures the cost axis the original
> study omitted**, runs end to end on one 6 GB consumer GPU in about two hours, and can be
> re-run by anyone who clones it.

Concretely, versus the paper: honest fixed-budget comparison (same epochs, same data, same
seeds for every method), 3 seeds with reported spread, the full cost column (trainable
params, peak VRAM, wall-clock, on-disk adapter size), a real adapter-only checkpoint so
the storage saving is actually demonstrated, and a CPU smoke path so `pytest` passes on
any machine with no GPU at all. Reduced scale is stated up front rather than hidden — a
result anyone can verify beats a bigger number nobody can.

The 2024 work is not thrown away. It is preserved as evidence: the working-tree diff is
committed as a patch, and the Slurm logs and metric JSONs are extracted from `defect.zip`
into `provenance/` (kilobytes, not the 500 MB checkpoints), so the repo can show exactly
what was done in 2024 and exactly what the rebuild changes.

---

## 6. Evidence index

| Claim | Command / file |
|---|---|
| 0 commits by Savan | `git log --all --author=avan` → empty |
| Repo is upstream's artifact | `README.md` ¶1; `git shortlog -sne --all` |
| Nothing recoverable is hidden | `git fsck --lost-found --unreachable --dangling` → no output; `git stash list` → empty |
| Savan's edits are uncommitted | `git diff --stat` (§1.2) |
| Token not on GitHub | `git log --all -S 'hf_fsYPFq…'` → empty |
| BitFit trained on 100 samples | `defect.zip` → `checkpoints/…am_bitfit…/all_results.json` |
| Uneven epochs | `defect.zip` → `…/trainer_state.json` |
| Paper's tables | `final project/Exploring Efficient Fine-Tuning Strategies….pdf`, Tables 2–3 |
| Local GPU | `nvidia-smi` → GeForce GTX 1660 Ti, 6144 MiB |

---

## 7. Outcome of the rebuild

The plan in [PLAN.md](PLAN.md) was executed in full. 24 runs completed on a free Colab T4 —
2 tasks x 4 methods x 3 seeds, identical budget for every method, no collapsed runs. Numbers
are in [docs/RESULTS.md](docs/RESULTS.md) and generated into [README.md](README.md).

**What the rebuild established:**

1. **A clean replication.** Full fine-tuning on Devign: 64.33 +/- 1.37 here, against 65.08 in
   the 2024 paper and 64.92 in Liu et al. (ASE 2023). Parallel adapter: 59.00 +/- 0.70 here
   against 58.78 in the paper. Three independent implementations inside a point.
2. **One number that does not reproduce, and it matters.** The 2024 paper's LoRA defect row
   (precision 0.2828, recall exactly 0.5000) is the signature of a single-class predictor.
   Here LoRA reaches 58.27 +/- 0.42 accuracy across three seeds. LoRA does learn the task;
   that row was a training failure published as a property of the method, because nothing in
   the original pipeline checked for it.
3. **The cost axis the 2024 study omitted.** A per-task BitFit checkpoint is 2.65 MB against
   full fine-tuning's 475.49 MB - 179x smaller - and PEFT trains 17-25% faster.
4. **A result that qualifies a common assumption about PEFT and memory.** Peak *allocated*
   memory fell 26-35% (6,829 MB -> 4,460-5,040 MB) as gradients and Adam state disappeared -
   a real ~2.3 GB saving. But peak *reserved* memory, which is what determines whether a job
   fits, was flat: 7,374 MB against 7,502-7,508 MB. Activations dominate the high-water mark
   and freezing weights does not shrink them. A 140x cut in trainable parameters bought a
   ~30% cut in allocated memory and no reduction in footprint.
5. **A stable ordering across two independent tasks:** full > parallel_adapter > lora >
   bitfit, with every gap exceeding the seed spread.

**The honesty ledger, updated.** Section 3 called the 2024 contribution replication rather
than novel research, and that stands - this rebuild does not make it novel. What it makes it
is *verifiable*, and it turns up a concrete error in the original results. The portfolio
claim is reproduction engineering and honest instrumentation: the collapse detector, the
cost accounting, and equal budgets are what turned an unnoticed bug into a documented
finding.

**Carried forward.** Two epochs is an equal-budget comparison, not a convergence study; more
epochs would likely narrow the defect gap, which favours PEFT. Clone detection is subsampled.
One base model.
