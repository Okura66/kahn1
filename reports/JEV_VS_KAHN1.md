# Kahn1 vs JEV — paired, on the Kahn1 held-out set

- Items: 14663 held-out items of `data/eval.jsonl`; JEV answered 14663.
- JEV: TypeSafe API, model jev-1.13.0, one question per request, run with `scripts/jev_holdout.py`.
- Kahn1: v3 checkpoint, k = 3, temperature calibration (reports/QWEN_FULL_EVAL_v3_temponly.md).
- Same state, prompt and options on both sides, scored against the same labels. ECE: 15 bins on p_max.
- JEV round trip over the internet: p50 248 ms, p95 308 ms.

| Dataset | Paired items | Kahn1 | JEV | Kahn1 ECE | JEV ECE | Only Kahn1 right | Only JEV right |
|---|---:|---:|---:|---:|---:|---:|---:|
| banking77 | 3076 | **91.09 %** | **80.17 %** | 0.0256 | 0.0876 | 414 | 78 |
| massive | 2974 | **90.79 %** | **80.13 %** | 0.0112 | 0.0844 | 377 | 60 |
| rte | 277 | **82.67 %** | **82.31 %** | 0.1046 | 0.0764 | 18 | 17 |
| scitail | 2126 | **65.00 %** | **44.36 %** | 0.2481 | 0.4136 | 490 | 51 |
| sst5 | 2210 | **52.35 %** | **57.74 %** | 0.0979 | 0.1777 | 216 | 335 |
| app_reviews | 4000 | **50.95 %** | **48.85 %** | 0.0961 | 0.2521 | 463 | 379 |
| all choice | 6050 | **90.94 %** | **80.15 %** | 0.0169 | 0.0853 | 791 | 138 |
| all noul | 2403 | **67.04 %** | **48.73 %** | 0.2316 | 0.3696 | 508 | 68 |
| all score | 6210 | **51.45 %** | **52.01 %** | 0.0967 | 0.2256 | 679 | 714 |
| all | 14663 | **70.30 %** | **63.08 %** | 0.0819 | 0.1913 | 1978 | 920 |

## Noul wording

JEV's native Noul form is the bare statement as `instructions` (also what JevBench sends). On SciTail,
whose negatives are hypotheses the text does not support, JEV then answers yes to most items: it seems to
judge whether the statement is true in general. Kahn1's prompt asks whether it is "true for this state".
The Noul items were re-run with `The text supports this statement: {statement}` to give JEV the same question:

| Dataset | Items | Kahn1 | JEV, bare statement | JEV, "supported" wording | JEV ECE (supported) |
|---|---:|---:|---:|---:|---:|
| rte | 277 | **82.67 %** | 82.31 % | **89.89 %** | 0.0344 |
| scitail | 2126 | **65.00 %** | 44.36 % | **72.44 %** | 0.0769 |
| all noul | 2403 | **67.04 %** | 48.73 % | **74.45 %** | 0.0705 |
| all | 14663 | **70.30 %** | 63.08 % | **67.30 %** | 0.1409 |

Kahn1's latency (p50 36.4 ms, k = 3) is local vLLM on one RTX 5070 Ti; JEV's is a hosted
API measured from the client, network included. They are not the same measurement.
