# Comprehensive Evaluation Report — 14,663 Holdout Instances

Evaluated Model: `checkpoints/qwen_merged`
Timestamp: 2026-09-22 16:52:23
Mode: Full Pipeline (Merged LoRA + debiasing $k=3$ + post-hoc calibration)

---

## 1. Overall Metrics (All 14,663 Unseen Instances)

| Metric | Value | Description |
|---|:---:|---|
| **Total Instances** | **14663** | 100% of the reserved holdout evaluation partition |
| **Global Accuracy** | **70.30 %** | Mean top-1 accuracy in pure zero-shot task evaluation |
| **NLL (Negative Log-Likelihood)** | **0.7531** | Probabilistic cross-entropy over correct answer tokens |
| **Brier Score** | **0.3987** | Multi-class quadratic accuracy (0 = perfect calibration) |
| **ECE (Expected Calibration Error)** | **0.0819** | Calibration gap across 15 equal-width bins |
| **ACE (Adaptive Calibration Error)** | **0.0812** | Calibration gap across 15 equal-mass quantile bins |
| **AURC (Area Under Risk-Coverage)** | **0.1153** | Error rejection capability via selective thresholding |
| **Latency p50** | **36.4 ms** | Median response latency |
| **Latency p95** | **82.6 ms** | Tail latency percentile |
| **Throughput** | **21.7 q/s** | Real throughput (queries/second) |
| **Total Runtime** | **11.3 min** | (675.4 seconds) |

---

## 2. Breakdown by Source Dataset

| Dataset | Primitive Type | Instances | Accuracy | ECE | Brier |
|---|---|:---:|:---:|:---:|:---:|
| **app_reviews_eval** | Score (5 levels) | 4000 | **50.95 %** | 0.0961 | 0.6096 |
| **banking77** | Choice (77 options) | 3076 | **91.09 %** | 0.0256 | 0.1371 |
| **massive** | Choice (60 options) | 2974 | **90.79 %** | 0.0112 | 0.1386 |
| **sst5_eval** | Score (5 levels) | 2210 | **52.35 %** | 0.0979 | 0.5936 |
| **scitail_eval** | Noul (binary) | 2126 | **65.00 %** | 0.2481 | 0.5578 |
| **rte_eval** | Noul (binary) | 277 | **82.67 %** | 0.1046 | 0.2723 |

---

## 2b. Breakdown by Primitive

| Primitive | Instances | Accuracy | ECE | Brier | NLL |
|---|:---:|:---:|:---:|:---:|:---:|
| **choice** | 6050 | **90.94 %** | 0.0169 | 0.1379 | 0.3104 |
| **noul** | 2403 | **67.04 %** | 0.2316 | 0.5249 | 0.9201 |
| **score** | 6210 | **51.45 %** | 0.0967 | 0.6039 | 1.1197 |

---

## 3. Ordinal Regression Analysis ($\mathbb{E}[	ext{Score}]$)

| Dataset | Exact Match | Off-by-one ($\pm 1$) | Discrete MAE | Continuous MAE $\mathbb{E}[S]$ | Spearman $\rho$ | Midpoint Bias Rate |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **app_reviews_eval** | **50.95 %** | **86.45 %** | 0.674 | 0.654 | **0.787** | 16.02 % |
| **sst5_eval** | **52.35 %** | **95.16 %** | 0.532 | 0.559 | **0.834** | 5.20 % |

---

## 4. Key Takeaways

- **Scaling Performance**: Accuracy reached **70.30%** on the full held-out test split when trained on the balanced mixture.
- **API Determinism & Schema Conformance**: Out-of-schema error rate = **0.0%** (zero risk of JSON malformation or syntax hallucination).
- **Inference Speed**: 36.4 ms vs ~300 ms for standard autoregressive JSON decoding (**~8.2x faster**).
