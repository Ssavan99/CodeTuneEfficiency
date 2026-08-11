# Third-party code and attribution

## Vendored third-party code

`clone/`, `defect/` and `petl/` contain code taken **unmodified** from the artifact release
accompanying:

> Liu, Sha & Peng. *An Empirical Study of Parameter-Efficient Fine-Tuning Methods for
> Pre-Trained Code Models.* ASE 2023.
> <https://github.com/anonymous-ase23/CodeModelParameterEfficientFinetuning>

**Copyright (c) 2023 the authors of that work, MIT licensed.** It is vendored here for
reference only, and each of those directories carries its own `THIRD-PARTY-NOTICE.md`
repeating this.

**Nothing in `codetune/` imports or depends on it.** The benchmark is a separate,
self-contained implementation; the vendored code can be deleted without affecting it.

Everything else in this repository — `codetune/`, `configs/`, `tests/`, `notebooks/`,
`results/`, `docs/` and the root documentation — is original work by Savan Patel.

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
