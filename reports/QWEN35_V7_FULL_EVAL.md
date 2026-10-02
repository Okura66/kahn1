# Comprehensive Evaluation Report — 14,663 Holdout Instances

Evaluated Model: `Okura66/Kahn1-Qwen2.5-3B` (run from a local copy of the same weights)
Timestamp: 2026-10-02 16:23:56
Mode: Full Pipeline (Merged LoRA + debiasing $k=3$ + post-hoc calibration)

---

## 1. Overall Metrics (All 14,663 Unseen Instances)

| Metric | Value | Description |
|---|:---:|---|
| **Total Instances** | **14663** | every item of the held-out set |
| **Global Accuracy** | **72.45 %** | Mean top-1 accuracy in pure zero-shot task evaluation |
| **NLL (Negative Log-Likelihood)** | **0.6860** | Probabilistic cross-entropy over correct answer tokens |
| **Brier Score** | **0.3673** | Multi-class quadratic score (lower is better, 0 = perfect) |
| **ECE (Expected Calibration Error)** | **0.0719** | Calibration gap across 15 equal-width bins |
| **ACE (Adaptive Calibration Error)** | **0.0719** | Calibration gap across 15 equal-mass quantile bins |
| **AURC (Area Under Risk-Coverage)** | **0.0933** | Error rejection capability via selective thresholding |
| **Latency p50** | **88.7 ms** | Median response latency |
| **Latency p95** | **279.3 ms** | Tail latency percentile |
| **Throughput** | **8.5 q/s** | Real throughput (queries/second) |
| **Total Runtime** | **28.7 min** | (1723.3 seconds) |

---

## 2. Breakdown by Source Dataset

| Dataset | Primitive Type | Instances | Accuracy | ECE | Brier |
|---|---|:---:|:---:|:---:|:---:|
| **app_reviews_eval** | Score (5 levels) | 4000 | **50.18 %** | 0.1455 | 0.6431 |
| **banking77** | Choice (77 options) | 3076 | **91.81 %** | 0.0183 | 0.1220 |
| **massive** | Choice (60 options) | 2974 | **92.91 %** | 0.0130 | 0.1033 |
| **sst5_eval** | Score (5 levels) | 2210 | **54.98 %** | 0.0522 | 0.5880 |
| **scitail_eval** | Noul (binary) | 2126 | **74.13 %** | 0.1411 | 0.3651 |
| **rte_eval** | Noul (binary) | 277 | **85.92 %** | 0.0483 | 0.1970 |

---

## 2b. Breakdown by Primitive

| Primitive | Instances | Accuracy | ECE | Brier | NLL |
|---|:---:|:---:|:---:|:---:|:---:|
| **choice** | 6050 | **92.35 %** | 0.0151 | 0.1128 | 0.2360 |
| **noul** | 2403 | **75.49 %** | 0.1275 | 0.3457 | 0.5247 |
| **score** | 6210 | **51.88 %** | 0.1123 | 0.6235 | 1.1868 |

---

## 3. Ordinal Regression Analysis ($\mathbb{E}[	ext{Score}]$)

| Dataset | Exact Match | Off-by-one ($\pm 1$) | Discrete MAE | Continuous MAE $\mathbb{E}[S]$ | Spearman $\rho$ | Midpoint Bias Rate |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **app_reviews_eval** | **50.18 %** | **84.42 %** | 0.711 | 0.673 | **0.783** | 14.35 % |
| **sst5_eval** | **54.98 %** | **94.48 %** | 0.515 | 0.572 | **0.833** | 6.24 % |

---

## 4. Key Takeaways

- **Scaling Performance**: Accuracy reached **72.45%** on the full held-out test split when trained on the balanced mixture.
- **API Determinism & Schema Conformance**: Out-of-schema error rate = **0.0%** (zero risk of JSON malformation or syntax hallucination).
- **Inference Speed**: median 88.7 ms per request (k = 3, one RTX 5070 Ti, vLLM).
