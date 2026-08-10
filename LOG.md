# Overnight progress log

- 2026-08-10: Ticked PLAN.md 1.9 and 2.9 — ran Phase 1 and Phase 2 code reviews
  (via Code Reviewer agent, since `/code-review` is user-only), fixed 3 real
  findings (missing OOM auto-fallback in train.py, dropped trailing
  grad-accumulation group, a bitfit test assertion that couldn't fail on
  regression). All 23 tests still pass. Committed as `6ad652c`. Kicked off the
  Phase 3 Devign defect grid (4 methods x 3 seeds, `configs/defect.yaml`) in
  the background on the local GTX 1660 Ti — still running at end of session.
  Clone grid (3.4), aggregation (3.5), plots (3.6), and Phase 3 review (3.7)
  not yet started; Phase 4 reporting not yet started.
- 2026-08-10: Found the Phase 3 Devign grid still running from the prior
  session (GPU at 100%, 5.97/6 GB used, healthy) and left it alone rather than
  interrupting in-progress training. Found and reviewed a set of uncommitted
  working-tree edits from that session (near-collapse detection instead of
  exact-uniformity, a fixed pristine-param-count denominator so trainable-
  fraction is comparable across methods, pinned host buffers for real async
  CUDA transfers, epochs 2->3 in defect.yaml/clone.yaml); `pytest -q` passed
  28/28, so committed as `9c30ebe`. Did not start the clone grid (3.4) or any
  other GPU work to avoid contending with the running defect grid on a 6 GB
  card. 3.3 remains unchecked — the next run should check whether
  `results/defect/` has all 12 run JSONs before starting anything else.
- 2026-08-10: Found the Devign defect grid (3.3) already running (started
  ~11:54, healthy, 100% GPU, 5.97/6 GB) against a further-resized config
  (train 600/eval 1000, 2 epochs — down from 2500/2732) with matching
  uncommitted changes to `configs/defect.yaml` and `configs/clone.yaml`.
  `pytest -q` (non-e2e) still passed 27/27, so committed the resize as
  `bf702fa`. Watched the log: first run (`defect__full__seed42`) took ~9.5
  min just for epoch 1/2, so the full 12-run grid is multi-hour, not
  finishable within one session turn. Left it running untouched rather than
  interrupt or duplicate it. 3.3 still unchecked; next run should again check
  `results/defect/` for 12 JSONs before touching the clone grid (3.4),
  aggregation (3.5), plots (3.6), or Phase 4.
- 2026-08-10: Confirmed the Devign defect grid (3.3, 12/12 JSONs) and BigCloneBench
  clone grid (3.4, 12/12 JSONs, watched to completion in background) were both
  done, ticked both boxes. Ran `codetune aggregate` and `codetune plot` (3.5/3.6)
  over all 24 runs -> results/summary.{csv,md}, results/figures/*.png; committed
  a real fix found along the way (`_safe_print` in aggregate.py was missing
  `import sys`, only surfaced when a run's console couldn't encode the tables'
  unicode). Found Phase 4 (README headline findings, FINDINGS.md outcome
  section, docs/RESULTS.md) already drafted uncommitted in the working tree
  from an earlier session; de-duplicated a redundant outcome section I'd
  independently written into FINDINGS.md in favor of the existing, more
  specific one, then committed. Ran the final whole-branch `/code-review` at
  high level via the Code Reviewer agent per PLAN.md 4.5 — zero findings.
  `pytest -q` 27/27, `git status` clean, every PLAN.md checkbox ticked:
  **project meets its own DONE definition.** Nothing pushed. Savan still needs
  to run the push/PR commands in PLAN.md §9 and revoke the leaked HF token
  before doing so.
- 2026-08-10: Overnight scheduled run. Re-verified prior session's DONE claim:
  all PLAN.md checkboxes ticked, `git status` clean, `pytest -q` 28/28 passing
  (via `.venv`, not system Python — system Python lacks torch). No unchecked
  work remains. Nothing to do this run; still waiting on Savan to run the
  push/PR commands in PLAN.md §9 and revoke the leaked HF token first.
- 2026-08-10: Overnight scheduled run. Re-verified again: `git status` clean on
  `feat/peft-code-benchmark`, `pytest -q` (via `.venv`) 28/28 passing, all
  PLAN.md checkboxes still ticked. No unchecked work remains. Still waiting on
  Savan to run the push/PR commands in PLAN.md §9 and revoke the leaked HF
  token before pushing.
- 2026-08-10: Overnight scheduled run. Re-verified once more: `pytest -q` via
  `.venv` 28/28 passing, `git status` clean on `feat/peft-code-benchmark`, all
  PLAN.md checkboxes ticked. Project remains DONE per its own definition;
  nothing left to automate. Still waiting on Savan to revoke the leaked HF
  token and run the push/PR commands in PLAN.md §9.
