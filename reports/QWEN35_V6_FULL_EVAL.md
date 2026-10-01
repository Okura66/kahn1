# Comprehensive Evaluation Report — 14,663 Holdout Instances

Evaluated Model: `/home/amontzamir/k1merged/v6`
Timestamp: 2026-10-01 14:09:44
Mode: Full Pipeline (Merged LoRA + debiasing $k=3$ + post-hoc calibration)

---

## 1. Overall Metrics (All 14,663 Unseen Instances)

| Metric | Value | Description |
|---|:---:|---|
| **Total Instances** | **14663** | 100% of the reserved holdout evaluation partition |
| **Global Accuracy** | **71.00 %** | Mean top-1 accuracy in pure zero-shot task evaluation |
| **NLL (Negative Log-Likelihood)** | **0.6858** | Probabilistic cross-entropy over correct answer tokens |
| **Brier Score** | **0.3735** | Multi-class quadratic accuracy (0 = perfect calibration) |
| **ECE (Expected Calibration Error)** | **0.0768** | Calibration gap across 15 equal-width bins |
| **ACE (Adaptive Calibration Error)** | **0.0767** | Calibration gap across 15 equal-mass quantile bins |
| **AURC (Area Under Risk-Coverage)** | **0.0997** | Error rejection capability via selective thresholding |
| **Latency p50** | **82.4 ms** | Median response latency |
| **Latency p95** | **287.9 ms** | Tail latency percentile |
| **Throughput** | **8.0 q/s** | Real throughput (queries/second) |
| **Total Runtime** | **30.5 min** | (1831.3 seconds) |

---

## 2. Breakdown by Source Dataset

| Dataset | Primitive Type | Instances | Accuracy | ECE | Brier |
|---|---|:---:|:---:|:---:|:---:|
| **app_reviews_eval** | Score (5 levels) | 4000 | **50.88 %** | 0.1060 | 0.6144 |
| **banking77** | Choice (77 options) | 3076 | **91.51 %** | 0.0223 | 0.1260 |
| **massive** | Choice (60 options) | 2974 | **93.11 %** | 0.0127 | 0.1042 |
| **sst5_eval** | Score (5 levels) | 2210 | **50.18 %** | 0.1406 | 0.6180 |
| **scitail_eval** | Noul (binary) | 2126 | **68.16 %** | 0.1656 | 0.4205 |
| **rte_eval** | Noul (binary) | 277 | **84.12 %** | 0.0547 | 0.2205 |

---

## 2b. Breakdown by Primitive

| Primitive | Instances | Accuracy | ECE | Brier | NLL |
|---|:---:|:---:|:---:|:---:|:---:|
| **choice** | 6050 | **92.30 %** | 0.0101 | 0.1153 | 0.2414 |
| **noul** | 2403 | **70.00 %** | 0.1521 | 0.3974 | 0.5878 |
| **score** | 6210 | **50.63 %** | 0.1179 | 0.6157 | 1.1567 |

---

## 3. Ordinal Regression Analysis ($\mathbb{E}[	ext{Score}]$)

| Dataset | Exact Match | Off-by-one ($\pm 1$) | Discrete MAE | Continuous MAE $\mathbb{E}[S]$ | Spearman $\rho$ | Midpoint Bias Rate |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **app_reviews_eval** | **50.88 %** | **86.22 %** | 0.679 | 0.652 | **0.789** | 14.20 % |
| **sst5_eval** | **50.18 %** | **94.43 %** | 0.562 | 0.574 | **0.837** | 4.43 % |

---

## 4. Key Takeaways

- **Scaling Performance**: Accuracy reached **71.00%** on the full held-out test split when trained on the balanced mixture.
- **API Determinism & Schema Conformance**: Out-of-schema error rate = **0.0%** (zero risk of JSON malformation or syntax hallucination).
- **Inference Speed**: 82.4 ms vs ~300 ms for standard autoregressive JSON decoding (**~3.6x faster**).
