# Kahn1 vs JevK5 (and JEV)

JevK5 v0.2 (github.com/allebee/jevk5): Qwen3.5-4B + a LoRA distilled from Qwen3.6-27B,
Apache-2.0. Kahn1: v3 checkpoint, k = 3, temperature calibration.

## JevBench, 231 public items

JevK5: its own run through JevBench's runner (per-item file in its repository). Jev: the
per-task outcomes JevBench publishes. Kahn1: `reports/jevbench_v3_preds.json`.

| Tier | Items | Kahn1 | JevK5 v0.2 | Jev 1.13.0 |
|---|---:|---:|---:|---:|
| easy | 48 | **100.0 %** | **100.0 %** | **100.0 %** |
| original | 72 | 84.7 % | 95.8 % | **98.6 %** |
| hard | 111 | 42.3 % | **73.9 %** | 73.0 % |
| all | 231 | 67.5 % | 86.1 % | **86.6 %** |

JevBench v1.4.0 re-measured both on its own pods: public accuracy JevK5
85.3 %, Jev 86.6 %; on the 308 fresh sealed items
JevK5 33.1 %, Jev 36.7 %. Kahn1 cannot run the sealed items.

## BANKING77 test split (77 intents)

| | Kahn1 v3 | JevK5 v0.2 | JEV 1.13.0 |
|---|---:|---:|---:|
| Accuracy, 1335 paired items | **91.0 %** | 68.6 % | 78.1 % |
| ECE (15 bins, p_max) | 0.029 | 0.035 | 0.101 |

- Coverage: 1337 of 3080 test items were run, which covers 34 of 77
  intents (PolyAI's test file is ordered by intent). All three systems are scored on the same items.
- JevK5: its own `bench/many_options.py`, knockout over ceil(77/16)+1 = 6 passes, in its own
  request shape; 1337 test items, 68.5 % on all of them; median
  389 ms per item on one RTX 5070 Ti without its optional
  flash-linear-attention kernels (JevK5 reports 116 ms on an H100 with them).
- Kahn1 and JEV: Kahn1's prompt and option order (the held-out run), JEV through its API.
- Paired by normalized text; 3076 BANKING77 items in Kahn1's eval set.
- JevK5 v0.2 trained on 1,500 BANKING77 train texts; Kahn1 never saw BANKING77.
