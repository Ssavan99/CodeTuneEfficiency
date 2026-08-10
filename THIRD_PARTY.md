# Third-party code and attribution

## Upstream repository

The git history of this repository (28 commits, April 2023 – January 2024, authored by
`anonymous-ase23` and `anonymous-pikachu`) is **not our work**. It is the artifact
repository released with:

> Liu, Sha & Peng. *An Empirical Study of Parameter-Efficient Fine-Tuning Methods for
> Pre-Trained Code Models.* ASE 2023.
> <https://github.com/anonymous-ase23/CodeModelParameterEfficientFinetuning>

That code lives under `clone/`, `defect/`, `petl/`, `summarization/`, `translation/`,
`clone-POJ-104/` and `NL_code_search_adv/`, and is preserved unmodified. Its original
README is kept as [`UPSTREAM_README.md`](UPSTREAM_README.md). It is MIT-licensed per that
README.

**Our own work in this repository is confined to** `codetune/`, `configs/`, `tests/`,
`notebooks/`, `provenance/`, `results/`, and the root `README.md`, `FINDINGS.md`,
`PLAN.md`, `THIRD_PARTY.md` and `LICENSE`.

## Models and datasets

| Asset | Source | License |
|---|---|---|
| `microsoft/codebert-base` | <https://huggingface.co/microsoft/codebert-base> | MIT |
| Devign (defect detection) | `google/code_x_glue_cc_defect_detection` via CodeXGLUE | see dataset card |
| BigCloneBench (clone detection) | `google/code_x_glue_cc_clone_detection_big_clone_bench` via CodeXGLUE | see dataset card |

## Papers referenced

- Feng et al. *CodeBERT: A Pre-Trained Model for Programming and Natural Languages.* EMNLP Findings 2020.
- Hu et al. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022.
- Ben Zaken, Ravfogel & Goldberg. *BitFit: Simple Parameter-Efficient Fine-tuning for Transformer-based Masked Language-models.* ACL 2022.
- He et al. *Towards a Unified View of Parameter-Efficient Transfer Learning.* ICLR 2022 — the parallel-adapter formulation used here.
- Houlsby et al. *Parameter-Efficient Transfer Learning for NLP.* ICML 2019.
- Lu et al. *CodeXGLUE: A Machine Learning Benchmark Dataset for Code Understanding and Generation.* NeurIPS Datasets & Benchmarks 2021.
