# Comprehensive Evaluation Report — 14,663 Holdout Instances

Evaluated Model: `<merged model>`
Timestamp: 2026-09-25 09:51:00
Mode: Full Pipeline (Merged LoRA + debiasing $k=3$ + post-hoc calibration)

---

## 1. Overall Metrics (All 14,663 Unseen Instances)

| Metric | Value | Description |
|---|:---:|---|
| **Total Instances** | **14663** | 100% of the reserved holdout evaluation partition |
| **Global Accuracy** | **70.82 %** | Mean top-1 accuracy in pure zero-shot task evaluation |
| **NLL (Negative Log-Likelihood)** | **0.7187** | Probabilistic cross-entropy over correct answer tokens |
| **Brier Score** | **0.3832** | Multi-class quadratic accuracy (0 = perfect calibration) |
| **ECE (Expected Calibration Error)** | **0.0756** | Calibration gap across 15 equal-width bins |
| **ACE (Adaptive Calibration Error)** | **0.0781** | Calibration gap across 15 equal-mass quantile bins |
| **AURC (Area Under Risk-Coverage)** | **0.1034** | Error rejection capability via selective thresholding |
| **Latency p50** | **91.9 ms** | Median response latency |
| **Latency p95** | **225.5 ms** | Tail latency percentile |
| **Throughput** | **8.7 q/s** | Real throughput (queries/second) |
| **Total Runtime** | **27.9 min** | (1676.8 seconds) |

---

## 2. Breakdown by Source Dataset

| Dataset | Primitive Type | Instances | Accuracy | ECE | Brier |
|---|---|:---:|:---:|:---:|:---:|
| **app_reviews_eval** | Score (5 levels) | 4000 | **48.45 %** | 0.1720 | 0.6668 |
| **banking77** | Choice (77 options) | 3076 | **91.06 %** | 0.0089 | 0.1294 |
| **massive** | Choice (60 options) | 2974 | **92.30 %** | 0.0269 | 0.1169 |
| **sst5_eval** | Score (5 levels) | 2210 | **51.95 %** | 0.1021 | 0.6071 |
| **scitail_eval** | Noul (binary) | 2126 | **71.03 %** | 0.1470 | 0.3824 |
| **rte_eval** | Noul (binary) | 277 | **87.36 %** | 0.0356 | 0.1830 |

---

## 2b. Breakdown by Primitive

| Primitive | Instances | Accuracy | ECE | Brier | NLL |
|---|:---:|:---:|:---:|:---:|:---:|
| **choice** | 6050 | **91.67 %** | 0.0154 | 0.1233 | 0.2636 |
| **noul** | 2403 | **72.91 %** | 0.1308 | 0.3594 | 0.5254 |
| **score** | 6210 | **49.69 %** | 0.1470 | 0.6456 | 1.2368 |

---

## 3. Ordinal Regression Analysis ($\mathbb{E}[	ext{Score}]$)

| Dataset | Exact Match | Off-by-one ($\pm 1$) | Discrete MAE | Continuous MAE $\mathbb{E}[S]$ | Spearman $\rho$ | Midpoint Bias Rate |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **app_reviews_eval** | **48.45 %** | **82.70 %** | 0.761 | 0.700 | **0.774** | 9.05 % |
| **sst5_eval** | **51.95 %** | **94.25 %** | 0.547 | 0.569 | **0.835** | 3.26 % |

---

## 4. Key Takeaways

- **Scaling Performance**: Accuracy reached **70.82%** on the full held-out test split when trained on the balanced mixture.
- **API Determinism & Schema Conformance**: Out-of-schema error rate = **0.0%** (zero risk of JSON malformation or syntax hallucination).
- **Inference Speed**: 91.9 ms vs ~300 ms for standard autoregressive JSON decoding (**~3.3x faster**).
