# Comprehensive Evaluation Report — 14,663 Holdout Instances

Evaluated Model: `Okura66/Kahn1-Qwen3.5-4B` (local merged copy)
Timestamp: 2026-09-25 15:03:53
Mode: Full Pipeline (Merged LoRA + debiasing $k=3$ + post-hoc calibration)

---

## 1. Overall Metrics (All 14,663 Unseen Instances)

| Metric | Value | Description |
|---|:---:|---|
| **Total Instances** | **14663** | 100% of the reserved holdout evaluation partition |
| **Global Accuracy** | **70.33 %** | Mean top-1 accuracy in pure zero-shot task evaluation |
| **NLL (Negative Log-Likelihood)** | **0.7098** | Probabilistic cross-entropy over correct answer tokens |
| **Brier Score** | **0.3803** | Multi-class quadratic accuracy (0 = perfect calibration) |
| **ECE (Expected Calibration Error)** | **0.0642** | Calibration gap across 15 equal-width bins |
| **ACE (Adaptive Calibration Error)** | **0.0652** | Calibration gap across 15 equal-mass quantile bins |
| **AURC (Area Under Risk-Coverage)** | **0.1040** | Error rejection capability via selective thresholding |
| **Latency p50** | **87.5 ms** | Median response latency |
| **Latency p95** | **204.8 ms** | Tail latency percentile |
| **Throughput** | **9.3 q/s** | Real throughput (queries/second) |
| **Total Runtime** | **26.4 min** | (1584.1 seconds) |

---

## 2. Breakdown by Source Dataset

| Dataset | Primitive Type | Instances | Accuracy | ECE | Brier |
|---|---|:---:|:---:|:---:|:---:|
| **app_reviews_eval** | Score (5 levels) | 4000 | **48.18 %** | 0.1279 | 0.6451 |
| **banking77** | Choice (77 options) | 3076 | **91.48 %** | 0.0171 | 0.1269 |
| **massive** | Choice (60 options) | 2974 | **92.40 %** | 0.0343 | 0.1110 |
| **sst5_eval** | Score (5 levels) | 2210 | **49.23 %** | 0.1245 | 0.6280 |
| **scitail_eval** | Noul (binary) | 2126 | **70.37 %** | 0.1485 | 0.3926 |
| **rte_eval** | Noul (binary) | 277 | **86.64 %** | 0.0263 | 0.1893 |

---

## 2b. Breakdown by Primitive

| Primitive | Instances | Accuracy | ECE | Brier | NLL |
|---|:---:|:---:|:---:|:---:|:---:|
| **choice** | 6050 | **91.93 %** | 0.0225 | 0.1191 | 0.2541 |
| **noul** | 2403 | **72.24 %** | 0.1317 | 0.3691 | 0.5463 |
| **score** | 6210 | **48.55 %** | 0.1264 | 0.6390 | 1.2172 |

---

## 3. Ordinal Regression Analysis ($\mathbb{E}[	ext{Score}]$)

| Dataset | Exact Match | Off-by-one ($\pm 1$) | Discrete MAE | Continuous MAE $\mathbb{E}[S]$ | Spearman $\rho$ | Midpoint Bias Rate |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **app_reviews_eval** | **48.18 %** | **81.45 %** | 0.789 | 0.693 | **0.768** | 10.12 % |
| **sst5_eval** | **49.23 %** | **93.57 %** | 0.582 | 0.593 | **0.823** | 3.89 % |

---

## 4. Key Takeaways

- **Scaling Performance**: Accuracy reached **70.33%** on the full held-out test split when trained on the balanced mixture.
- **API Determinism & Schema Conformance**: Out-of-schema error rate = **0.0%** (zero risk of JSON malformation or syntax hallucination).
- **Inference Speed**: 87.5 ms vs ~300 ms for standard autoregressive JSON decoding (**~3.4x faster**).
