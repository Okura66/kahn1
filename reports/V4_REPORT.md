# Kahn1 v4 (Qwen3.5-4B + LoRA) — local evaluation

v4: Qwen3.5-4B, LoRA r=16 on attention and linear-attention projections, native chat
template, trained on data/train_v4.jsonl (the v3 mixture subsampled + DocNLI, ShARC, WANLI,
QuALITY), checkpoint selected on data/dev_v4.jsonl. v3: Qwen2.5-3B. Local, not published.

## Held-out 14,663 items, like for like

Choice over the same 8 options for every system; JEV's Noul asked whether the text supports
the statement. k = 3 and temperature calibration for v4 and v3.

| Dataset | Items | v4 | v3 | JEV 1.13.0 | v4 ECE | v3 ECE | JEV ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| banking77 | 3076 | 91.1 % | 91.1 % | **94.8 %** | 0.009 | 0.026 | 0.024 |
| massive | 2974 | 92.3 % | 90.8 % | **94.1 %** | 0.027 | 0.011 | 0.017 |
| rte | 277 | 87.4 % | 82.7 % | **89.9 %** | 0.036 | 0.105 | 0.034 |
| scitail | 2126 | 71.0 % | 65.0 % | **72.4 %** | 0.147 | 0.248 | 0.077 |
| sst5 | 2210 | 51.9 % | 52.4 % | **57.7 %** | 0.102 | 0.098 | 0.178 |
| app_reviews | 4000 | 48.4 % | **50.9 %** | 48.9 % | 0.172 | 0.096 | 0.252 |
| all choice | 6050 | 91.7 % | 90.9 % | **94.5 %** | 0.015 | 0.017 | 0.019 |
| all score | 6210 | 49.7 % | 51.4 % | **52.0 %** | 0.147 | 0.097 | 0.226 |
| all noul | 2403 | 72.9 % | 67.0 % | **74.4 %** | 0.131 | 0.232 | 0.071 |
| all | 14663 | 70.8 % | 70.3 % | **73.2 %** | 0.076 | 0.082 | 0.113 |

## Choice over every intent

| Condition | v4 | v3 | JEV 1.13.0 |
|---|---:|---:|---:|
| every intent (77 / 60), 1184 items | 70.1 % | 68.4 % | **79.1 %** |

## JevBench, 231 public items

| Tier | Items | v4 | v3 | JevK5 v0.2 | Jev 1.13.0 |
|---|---:|---:|---:|---:|---:|
| easy | 48 | **100.0 %** | **100.0 %** | **100.0 %** | **100.0 %** |
| original | 72 | 97.2 % | 84.7 % | 95.8 % | **98.6 %** |
| hard | 111 | 55.9 % | 42.3 % | **73.9 %** | 73.0 % |
| all | 231 | 77.9 % | 67.5 % | 86.1 % | **86.6 %** |

## Iteration: interpolating base and v4

Every LoRA delta multiplied by a coefficient before the merge (0 = the untrained base, 1 = v4).
k = 1, uncalibrated. The coefficient is chosen on the dev split only; the held-out sample and
JevBench are measured for this report, not used to choose.

| Coefficient | Dev split (selection) | Held-out sample, balanced | JevBench | JevBench hard |
|---|---:|---:|---:|---:|
| 0 (Qwen3.5-4B untrained) | — | 61.2 % | 79.2 % | 63.1 % |
| 0.5 | 76.2 % | 68.1 % | 80.1 % | 64.0 % |
| 0.75 | 78.5 % | 69.6 % | 78.4 % | 57.7 % |
| 1.0 (v4) | 80.6 % | 71.0 % | 77.1 % | 55.0 % |

Training improves what it trains (dev, held-out) and steadily erodes the base model's hard
judgment (JevBench hard tier). Interpolation moves along that trade-off without improving both;
on the dev split v4 (1.0) stays the choice. The lever is data that trains hard judgment.
