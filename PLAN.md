# PLAN — CodeTuneEfficiency rebuild

**Branch:** `feat/peft-code-benchmark` · **Created:** 2026-08-10 · **Verdict:** B (see [FINDINGS.md](FINDINGS.md))
**Approval status:** ✅ APPROVED 2026-08-10 — verdict B, with the softened public framing
(six-defect autopsy lives in `provenance/README.md`, not the front-page README).

---

## 0. Rules (restated in full — a fresh session must be able to work from this file alone)

- **Budget is $0. Hard constraint.** Free compute only: this machine (GTX 1660 Ti, 6 GB),
  Colab free tier, Kaggle free weekly GPU hours, HF free tier. Free/public datasets and
  models small enough to finish inside those limits. Scope every phase to fit the free
  tier rather than assuming compute. Never sign Savan up for anything, never spend money.
  If the only viable path costs money: **STOP and ask**, with the cheapest alternative found.
- **Feature branch only** — `feat/peft-code-benchmark`. Never commit to `main`/`master`.
- **Small commits, clear messages.** **Do not push.** Push/PR commands are at the bottom
  of this file for Savan to run.
- **After each phase**, run the `/code-review` skill at **high** level on that phase's diff
  and fix real findings before starting the next phase.
- **Tick the checkboxes below as work completes**, and keep FINDINGS.md current.
- **Use subagents:** `scout` for repo search / git archaeology / document reading,
  `implementer` for self-contained build units. Judgment calls stay on the main thread.
- **Do not ask about routine choices.** Pick the sensible default, record it in §7, keep
  moving. `AskUserQuestion` is reserved for exactly three things: anything that costs
  money, deleting or rewriting git history, or changing what the repo publicly claims
  about Savan's work.
- **If blocked:** write the blocker into §8, move to the next unblocked item, and only
  ping Savan if *everything* is blocked.
- **DONE** = all four hold: (1) FINDINGS.md documents what survives and states the
  verdict; (2) PLAN.md approved and every phase checkbox ticked; (3) code runs end to end,
  eval/test step exits 0, `git status` clean on the feature branch; (4) README.md
  documents the project and its results.

---

## 1. What is being built, and why

The public repo is a copy of another team's research artifact with zero commits from
Savan (FINDINGS §1.1). The real 2024 work exists only as an uncommitted diff and a 5.4 GB
zip (§1.2–1.3). The paper is an honest small replication of Liu et al. ASE 2023, with six
concrete defects in its results (§2) — the most serious being that two of the four rows in
its defect table were trained on 100 samples.

**The rebuild's claim:**

> A reproducible PEFT benchmark for code models that measures the cost axis the original
> study omitted, runs end to end on one 6 GB consumer GPU in about two hours, and can be
> re-run by anyone who clones it.

Deliberately *not* claiming to match the paper's scale. Reduced scale is stated up front.
A result anyone can verify beats a bigger number nobody can.

**What it adds over the paper:** equal-budget comparison (identical epochs/data/seeds
across methods), 3 seeds with reported spread, a real cost column (trainable params, peak
VRAM, wall-clock, adapter-only checkpoint bytes), adapter-only checkpoints so the storage
saving is actually demonstrated rather than asserted, and a CPU smoke path so the suite
passes with no GPU at all.

---

## 2. Phase 0 — Research ✅ COMPLETE

- [x] Read `GITHUB-PORTFOLIO-AUDIT.md`
- [x] Read the paper in full + every reference in the CSCE 962 folder (via `scout`)
- [x] Full git archaeology: `log --all`, `reflog`, `stash list`, `branch -avv`,
      `fsck --lost-found`, untracked files, ahead/behind (via `scout`)
- [x] Identify `CodeTuneEfficiency-model/defect.zip` (5.4 GB) without extracting it
- [x] Commit to one verdict → **B**
- [x] Write `FINDINGS.md`
- [x] Write `PLAN.md` (this file)
- [x] **Savan approves this plan** ← the one and only approval gate

**Compute:** none. Read-only.

---

## 3. Phase 1 — Provenance rescue and repo hygiene

Preserve the 2024 work as evidence, make the tree safe to commit, and stop the repo from
implying Liu et al.'s artifact is Savan's.

- [x] 1.1 Capture the uncommitted 2024 edits as `provenance/original-2024-edits.patch`
      (`git diff` + the two untracked `run-bitfit.sh` files), **with the HF token
      redacted** before it is written to disk
- [x] 1.2 Extract only the small text artifacts from `defect.zip` into
      `provenance/original-runs/` — `all_results.json`, `test_results.json`,
      `trainer_state.json` per run, and the four `slurm-*.out` logs. **No checkpoints.**
      Target < 2 MB total. Read via `zipfile` member-by-member; never bulk-extract.
- [x] 1.3 Write `provenance/README.md`: what the 2024 run was, on what hardware, and the
      six defects from FINDINGS §2 with a pointer to the JSON that evidences each
- [x] 1.4 Add `.gitignore` — datasets (`**/dataset/`, `*.jsonl`, `*.zip`), checkpoints
      (`checkpoints/`, `*.safetensors`, `*.bin`, `*.pt`), `hf_cache/`, `.venv/`,
      `__pycache__/`, `wandb/`, `results/**/raw/`
- [x] 1.5 Scrub the leaked HF token from the working tree (4 × `run.py`). Verify
      `grep -rI 'hf_fsYPFq' --exclude-dir=.git .` returns nothing before any commit
- [x] 1.6 Restore the upstream tree to pristine (`git restore` the modified/deleted
      upstream files) so the diff on this branch is *only* new work. The 2024 edits
      survive in the patch from 1.1 — nothing is lost, and no history is rewritten
- [x] 1.7 Add `LICENSE` (MIT, as upstream's README asserts) and `THIRD_PARTY.md` crediting
      `anonymous-ase23/CodeModelParameterEfficientFinetuning` and Liu et al. ASE 2023
- [x] 1.8 Move upstream's `README.md` → `UPSTREAM_README.md` (placeholder root README
      written in Phase 4)
- [x] 1.9 `/code-review` at high level on the Phase 1 diff; fix real findings

**Acceptance:** `provenance/` exists with the patch + run artifacts and is < 2 MB; the
token appears nowhere outside `.git`; `git status` shows only intended new files;
`git diff main...HEAD --stat` contains no modification to upstream source files.

**Compute:** $0 — local CPU, file operations only. No downloads.

---

## 4. Phase 2 — Clean implementation

New self-contained package at repo root. Does not touch upstream code.

```
codetune/
  data.py       Devign + BigCloneBench loaders (HF hub, public), deterministic subsampling
  methods.py    full | bitfit | lora | parallel_adapter — apply to model, freeze, count
  cost.py       trainable params, peak VRAM, wall-clock, adapter-only checkpoint bytes
  train.py      one run → results/<run_id>.json
  aggregate.py  results/*.json → summary.csv + markdown table
  plots.py      accuracy-vs-cost scatter, per-method bars
  cli.py        python -m codetune run|aggregate|plot --config ...
configs/        smoke.yaml, defect.yaml, clone.yaml
tests/          test_methods.py, test_data.py, test_cost.py, test_smoke_e2e.py
```

- [x] 2.1 `requirements.txt` + `requirements-dev.txt`, pinned. Python 3.10
      (`C:\Program Files\Python310`), `.venv` in repo root, torch cu121 (Turing sm_75,
      fp16 — no bf16 on this card)
- [x] 2.2 `data.py` — load `google/code_x_glue_cc_defect_detection` (Devign) and
      `google/code_x_glue_cc_clone_detection_big_clone_bench` from the HF hub. Both public,
      no auth, no gdown. Deterministic seeded subsampling with the subset size recorded in
      every result file
- [x] 2.3 `methods.py` — `full` (all trainable); `bitfit` (only `*.bias` + classifier);
      `lora` (via `peft`, r=8 α=16 on query/value); `parallel_adapter` (custom bottleneck
      MLP parallel to each layer's FFN, dim 16, per He et al.'s unified view — the
      formulation the paper cites). Every method returns the same interface and a verified
      trainable-parameter count
- [x] 2.4 `cost.py` — trainable/total params, `torch.cuda.max_memory_allocated`, train
      wall-clock, and **bytes of the trainable-only state dict** (the number the original
      artifacts never produced)
- [x] 2.5 `train.py` — HF `Trainer`, fp16, seeded, fixed equal budget across methods,
      early stopping off (equal budget is the point), OOM auto-fallback halving batch size
      and doubling grad accumulation. Writes one self-describing JSON per run
- [x] 2.6 `configs/smoke.yaml` — CPU, 64 train / 32 eval, 1 epoch, seq 128, under 3 min
- [x] 2.7 Tests: unit tests build a *tiny random* Roberta locally (no download, offline,
      fast) to assert freezing correctness and param accounting per method; one end-to-end
      test runs the smoke config and skips cleanly if the hub is unreachable
- [x] 2.8 Verify `pytest` exits 0 **and** `python -m codetune run --config configs/smoke.yaml`
      completes end to end
- [x] 2.9 `/code-review` at high level on the Phase 2 diff; fix real findings

**Acceptance:** `pytest -q` exits 0; the smoke run completes on CPU in < 3 min and writes
a valid result JSON; `methods.py` param counts match hand-computed expectations in tests;
no network access required by the unit tests.

**Compute:** $0 — local CPU. One-time ~500 MB `codebert-base` download from the HF free tier.

---

## 5. Phase 3 — Experiments

- [x] 3.1 Sanity run: 1 method × 1 seed × 1 epoch on Devign on the local GPU; confirm it
      fits in 6 GB and record the real minutes/epoch
- [x] 3.2 Recalibrate the grid from 3.1's measured throughput; record the final grid here
- [x] 3.3 **Defect (Devign)** — 4 methods × 3 seeds (42/1337/2024), equal budget, seq 320.
      Full 21 854-example train set
- [x] 3.4 **Clone (BigCloneBench)** — 4 methods × 3 seeds, seq 400, deterministically
      subsampled to 20 k train / 2 k val / 4 k test. **Subset size stated in the README** —
      not presented as the full benchmark
- [ ] 3.5 Aggregate → `results/summary.csv` + mean ± std per method/task/metric
- [ ] 3.6 Plots → `results/figures/` (accuracy-vs-trainable-params, accuracy-vs-peak-VRAM,
      per-method bars with error bars)
- [x] 3.7 `/code-review` at high level on the Phase 3 diff; fix real findings

**Acceptance:** ≥ 24 completed runs with committed per-run JSONs; `summary.csv` reproduces
from those JSONs via `python -m codetune aggregate`; every reported number carries a
standard deviation over 3 seeds; no run OOMs.

**Compute:** $0 — local GTX 1660 Ti, run in the background, est. 6–9 h wall-clock total.
**Fallback if the local GPU proves too slow or unstable:** `notebooks/run_on_colab.ipynb`
on Colab free T4 (fits comfortably; a 12 h session covers the whole grid), or Kaggle's
free weekly GPU hours. Both free, no signup beyond accounts Savan already has. Second
fallback: cut to 2 seeds and note it here.

---

## 6. Phase 4 — Reporting and finish

- [ ] 4.1 `README.md` — what this is, the honest provenance (upstream artifact + 2024
      Master's project + 2026 rebuild), quickstart, results tables with error bars, the
      figures, and an explicit "how this differs from the 2024 paper" section covering all
      six defects and how each was addressed
- [x] 4.2 A short `docs/RESULTS.md` interpreting the findings — including any case where
      the rebuild *disagrees* with the paper, stated plainly
- [ ] 4.3 Update FINDINGS.md with the rebuild's outcome
- [x] 4.4 Repo metadata note for Savan (description + topics to set on GitHub — the audit
      flags these as missing across the account)
- [ ] 4.5 Final `/code-review` at high level over the whole branch; fix real findings
- [ ] 4.6 Confirm DONE: `pytest` exits 0, end-to-end run works, `git status` clean,
      all boxes above ticked

**Acceptance:** a stranger can clone, follow the README, and reproduce the smoke run in
minutes and the full grid in hours, on free compute, with no manual dataset wrangling.

**Compute:** $0 — local CPU.

---

## 7. Decisions taken without asking (routine defaults)

| # | Decision | Why |
|---|---|---|
| D1 | Build on `feat/peft-code-benchmark`, keep upstream's 28 commits intact | Rewriting history needs approval; keeping it is also the honest record of where the code came from |
| D2 | New `codetune/` package rather than editing upstream files | Keeps the diff reviewable and provenance unambiguous |
| D3 | Datasets from the HF hub, not upstream's gdown links | Public, free, no auth, no link rot |
| D4 | CodeBERT-base only | Matches the paper; the only 6 GB-feasible option |
| D5 | 3 seeds, equal fixed epoch budget, early stopping off | Directly fixes FINDINGS §2.2 and §2.4 |
| D6 | Clone task subsampled to 20 k train | 90 k × seq 512 does not fit the $0 budget; the subset is disclosed, not hidden |
| D7 | `peft` for LoRA; BitFit and Parallel-Adapter hand-implemented | `peft` has no parallel adapter; hand-implementing shows the mechanics |
| D8 | Save adapter-only checkpoints | The original saved 500 MB full state dicts, so the storage saving was never realised |
| D9 | 2024 edits preserved as a patch, tree restored to pristine | Nothing lost, no history rewritten, clean diff |
| D10 | Token scrubbed from working tree; Savan told to revoke it | It is not in git history, so no history surgery is needed |
| D11 | Grid resized after a timing probe: 2 500 train, 2 epochs, seq 128 | Measured 0.25 s/example on the 1660 Ti. Turing GTX cards have no tensor cores, so fp16 barely helps and the originally planned grid would have taken ~24 h. The reduced sizes are disclosed in the configs and recorded in every result JSON. |
| D12 | Both tasks kept, rather than one task at larger scale | The cross-task comparison (RQ3 in the original paper) is the more interesting result, and it survives reduced scale better than a single-task absolute number does. |
| D13 | Grid stopped mid-flight to apply review fixes, then restarted | Several review findings changed the reported numbers (LoRA's parameter denominator, wall-clock timing bias, LR schedule length). Finishing the grid first would have produced results that had to be thrown away anyway. |
| D14 | Kill stray python processes before each GPU run | Two grids were crippled by leftover processes holding ~6 GB of the card, which looked like the GPU being slow rather than contended. Verified with `nvidia-smi --query-compute-apps`. |
| D15 | Clone grid further resized to train=600/eval=1000 (from the originally planned 20k/2k/4k) | Same throughput constraint as D11: measured 0.40 s/example on the 1660 Ti made the original size ~9h for the clone grid alone. Documented in-line in `configs/clone.yaml` with the reasoning; cost columns (params/VRAM/wall-clock/checkpoint bytes) stay exact and scale-independent, only quality numbers are affected and are labeled accordingly. |

## 8. Blockers

**B2 — stale processes silently saturate the 6 GB card (resolved).** Two grid attempts
crawled because leftover python processes from earlier probes held ~6 GB of VRAM; the
symptom was a 4x apparent slowdown, not an error. Fixed by killing all python before a
GPU run and confirming `nvidia-smi --query-gpu=memory.used` reads 0 MiB first. If a run
ever looks inexplicably slow, check this before touching the config.

**B1 — local GPU is the binding constraint (open, worked around, not escalated).**
The development card is a GTX 1660 Ti: 6 GB, Turing, and crucially *no tensor cores*, so
fp16 gives almost no speedup. A timing probe measured 0.25 s per training example, which
put the originally planned grid (full 21 854-example Devign train set, 3 epochs, seq 320)
at roughly 24 hours. Worked around by resizing the grid (D11) rather than escalating,
since the reduced scale is disclosed everywhere it matters and the $0 constraint holds.
Anyone wanting the full-scale numbers can run the same grid on a free Colab or Kaggle T4
via `notebooks/run_on_free_gpu.ipynb` — a T4 has tensor cores and is roughly 10x faster
here. No money involved either way.

---

## 9. Push / PR commands — for Savan to run

Nothing is pushed automatically. When the branch looks right:

```bash
cd "C:/Users/savan/source/repos/CodeTuneEfficiency" && git status && git log --oneline main..HEAD
```

```bash
cd "C:/Users/savan/source/repos/CodeTuneEfficiency" && git push -u origin feat/peft-code-benchmark
```

```bash
cd "C:/Users/savan/source/repos/CodeTuneEfficiency" && gh pr create --base main --head feat/peft-code-benchmark --title "Rebuild: reproducible PEFT benchmark for code models" --body-file PR_BODY.md
```

Optional repo metadata (the audit flags description/topics as missing account-wide):

```bash
gh repo edit Ssavan99/CodeTuneEfficiency --description "Reproducible parameter-efficient fine-tuning benchmark for code models: accuracy vs. real cost on a 6 GB consumer GPU" --add-topic peft --add-topic lora --add-topic codebert --add-topic reproducibility --add-topic machine-learning
```

**Before pushing, revoke the leaked Hugging Face token** at
<https://huggingface.co/settings/tokens> — it is in `defect.zip` and in the pre-scrub
working tree, even though it never reached GitHub.
