# Comprehensive Evaluation Report — 10,000 Holdout Instances

Evaluated Model: `Okura66/Kahn1-Qwen3.5-4B`
Timestamp: 2026-10-02 20:30:01
Mode: Full Pipeline (Merged LoRA + debiasing $k=3$ + post-hoc calibration)

---

## 1. Overall Metrics (All 10,000 Unseen Instances)

| Metric | Value | Description |
|---|:---:|---|
| **Total Instances** | **10000** | every item of the evaluation file |
| **Global Accuracy** | **79.92 %** | Mean top-1 accuracy in pure zero-shot task evaluation |
| **NLL (Negative Log-Likelihood)** | **0.4986** | Probabilistic cross-entropy over correct answer tokens |
| **Brier Score** | **0.2794** | Multi-class quadratic score (lower is better, 0 = perfect) |
| **ECE (Expected Calibration Error)** | **0.0485** | Calibration gap across 15 equal-width bins |
| **ACE (Adaptive Calibration Error)** | **0.0487** | Calibration gap across 15 equal-mass quantile bins |
| **AURC (Area Under Risk-Coverage)** | **0.0576** | Error rejection capability via selective thresholding |
| **Latency p50** | **86.8 ms** | Median response latency |
| **Latency p95** | **223.0 ms** | Tail latency percentile |
| **Throughput** | **9.2 q/s** | Real throughput (queries/second) |
| **Total Runtime** | **18.2 min** | (1090.8 seconds) |

---

## 2. Breakdown by Source Dataset

| Dataset | Primitive Type | Instances | Accuracy | ECE | Brier |
|---|---|:---:|:---:|:---:|:---:|
| **banking77:train** | Choice (8 of 77 options) | 2000 | **91.55 %** | 0.0167 | 0.1250 |
| **massive:train** | Choice (8 of 60 options) | 2000 | **93.40 %** | 0.0152 | 0.0985 |
| **sst5_eval:train** | Score (5 levels) | 2000 | **55.40 %** | 0.0534 | 0.5950 |
| **scitail_eval:train** | Noul (binary) | 2000 | **71.95 %** | 0.1507 | 0.3971 |
| **rte_eval:train** | Noul (binary) | 2000 | **87.30 %** | 0.0264 | 0.1813 |

---

## 2b. Breakdown by Primitive

| Primitive | Instances | Accuracy | ECE | Brier | NLL |
|---|:---:|:---:|:---:|:---:|:---:|
| **choice** | 4000 | **92.47 %** | 0.0123 | 0.1118 | 0.2300 |
| **noul** | 4000 | **79.62 %** | 0.0885 | 0.2892 | 0.4492 |
| **score** | 2000 | **55.40 %** | 0.0534 | 0.5950 | 1.1345 |

---

## 3. Ordinal Regression Analysis ($\mathbb{E}[	ext{Score}]$)

| Dataset | Exact Match | Off-by-one ($\pm 1$) | Discrete MAE | Continuous MAE $\mathbb{E}[S]$ | Spearman $\rho$ | Midpoint Bias Rate |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **sst5_eval:train** | **55.40 %** | **94.10 %** | 0.515 | 0.575 | **0.823** | 6.20 % |

---

## 4. Key Takeaways

- **Scaling Performance**: Accuracy reached **79.92%** on the full held-out test split when trained on the balanced mixture.
- **API Determinism & Schema Conformance**: Out-of-schema error rate = **0.0%** (zero risk of JSON malformation or syntax hallucination).
- **Inference Speed**: median 86.8 ms per request.
