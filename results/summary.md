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
