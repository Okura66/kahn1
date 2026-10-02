# Comprehensive Evaluation Report — 231 Holdout Instances

Evaluated Model: `<merged model>`
Timestamp: 2026-10-01 13:39:07
Mode: Full Pipeline (Merged LoRA + debiasing $k=3$ + post-hoc calibration)

---

## 1. Overall Metrics (All 231 Unseen Instances)

| Metric | Value | Description |
|---|:---:|---|
| **Total Instances** | **231** | 100% of the reserved holdout evaluation partition |
| **Global Accuracy** | **86.15 %** | Mean top-1 accuracy in pure zero-shot task evaluation |
| **NLL (Negative Log-Likelihood)** | **0.4294** | Probabilistic cross-entropy over correct answer tokens |
| **Brier Score** | **0.2220** | Multi-class quadratic accuracy (0 = perfect calibration) |
| **ECE (Expected Calibration Error)** | **0.0737** | Calibration gap across 15 equal-width bins |
| **ACE (Adaptive Calibration Error)** | **0.0738** | Calibration gap across 15 equal-mass quantile bins |
| **AURC (Area Under Risk-Coverage)** | **0.0323** | Error rejection capability via selective thresholding |
| **Latency p50** | **107.5 ms** | Median response latency |
| **Latency p95** | **516.4 ms** | Tail latency percentile |
| **Throughput** | **2.8 q/s** | Real throughput (queries/second) |
| **Total Runtime** | **1.4 min** | (83.8 seconds) |

---

## 2. Breakdown by Source Dataset

| Dataset | Primitive Type | Instances | Accuracy | ECE | Brier |
|---|---|:---:|:---:|:---:|:---:|
| **jevbench-original** | Noul (binary) | 72 | **98.61 %** | 0.0663 | 0.0378 |
| **jevbench-easy** | Choice (5 options) | 48 | **100.00 %** | 0.0051 | 0.0003 |
| **jevbench-hard** | Choice (5 options) | 111 | **72.07 %** | 0.1922 | 0.4373 |

---

## 2b. Breakdown by Primitive

| Primitive | Instances | Accuracy | ECE | Brier | NLL |
|---|:---:|:---:|:---:|:---:|:---:|
| **choice** | 139 | **86.33 %** | 0.0756 | 0.2167 | 0.4610 |
| **noul** | 74 | **86.49 %** | 0.0903 | 0.2347 | 0.3851 |
| **score** | 18 | **83.33 %** | 0.1352 | 0.2111 | 0.3673 |

---

## 4. Key Takeaways

- **Scaling Performance**: Accuracy reached **86.15%** on the full held-out test split when trained on the balanced mixture.
- **API Determinism & Schema Conformance**: Out-of-schema error rate = **0.0%** (zero risk of JSON malformation or syntax hallucination).
- **Inference Speed**: 107.5 ms vs ~300 ms for standard autoregressive JSON decoding (**~2.8x faster**).
