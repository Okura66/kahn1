# Comprehensive Evaluation Report — 10,663 Holdout Instances

Evaluated Model: `checkpoints/qwen_merged`
Timestamp: 2026-09-22 05:55:14
Mode: Full Pipeline (Merged LoRA + debiasing $k=3$ + post-hoc calibration)

First evaluation covering all three primitives: the previous 8,260-instance
benchmark contained no Noul example at all. Calibration temperatures were fitted
on a split disjoint from training (the previous split was fully contained in it).

---

## 1. Overall Metrics (All 10,663 Unseen Instances)

| Metric | Value | Description |
|---|:---:|---|
| **Total Instances** | **10,663** | 100% of the reserved holdout evaluation partition |
| **Global Accuracy** | **77.60 %** | Mean top-1 accuracy in pure zero-shot task evaluation |
| **NLL (Negative Log-Likelihood)** | **0.5714** | Probabilistic cross-entropy over correct answer tokens |
| **Brier Score** | **0.3115** | Multi-class quadratic accuracy (0 = perfect calibration) |
| **ECE (Expected Calibration Error)** | **0.0537** | Calibration gap across 15 equal-width bins |
| **ACE (Adaptive Calibration Error)** | **0.0552** | Calibration gap across 15 equal-mass quantile bins |
| **AURC (Area Under Risk-Coverage)** | **0.0768** | Error rejection capability via selective thresholding |
| **Latency p50** | **35.2 ms** | Median response latency |
| **Latency p95** | **86.5 ms** | Tail latency percentile |
| **Throughput** | **21.5 q/s** | Real throughput (queries/second) |
| **Total Runtime** | **8.3 min** | (496.0 seconds) |

Note: the global accuracy is not comparable to the previous report's 80.12%,
which averaged over a different composition (choice and score only, 8,260 items).

---

## 2. Breakdown by Source Dataset

| Dataset | Primitive Type | Instances | Accuracy | ECE | Brier |
|---|---|:---:|:---:|:---:|:---:|
| **banking77** | Choice (77 options) | 3076 | **90.73 %** | 0.0230 | 0.1462 |
| **massive** | Choice (60 options) | 2974 | **90.85 %** | 0.0224 | 0.1378 |
| **sst5_eval** | Score (5 levels) | 2210 | **52.22 %** | 0.0576 | 0.5892 |
| **rte_eval** | Noul (binary) | 277 | **84.12 %** | 0.0771 | 0.2481 |
| **scitail_eval** | Noul (binary) | 2126 | **65.57 %** | 0.2204 | 0.5131 |

---

## 2b. Breakdown by Primitive

| Primitive | Instances | Accuracy | ECE | Brier | NLL |
|---|:---:|:---:|:---:|:---:|:---:|
| **choice** | 6050 | **90.79 %** | 0.0167 | 0.1421 | 0.3199 |
| **noul** | 2403 | **67.71 %** | 0.2037 | 0.4825 | 0.7767 |
| **score** | 2210 | **52.22 %** | 0.0576 | 0.5892 | 1.0366 |

---

## 3. Ordinal Regression Analysis ($\mathbb{E}[\text{Score}]$)

| Dataset | Exact Match | Off-by-one ($\pm 1$) | Discrete MAE | Continuous MAE $\mathbb{E}[S]$ | Spearman $\rho$ | Midpoint Bias Rate |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **sst5_eval** | **52.22 %** | **95.20 %** | 0.532 | 0.572 | **0.839** | 12.94 % |

---

## 4. Key Takeaways

- **Noul is measured for the first time**: 84.12% on RTE (in-form binary
  entailment) and 65.57% on SciTail. SciTail's entailment-vs-neutral framing sits
  outside the entailment-vs-contradiction training signal, and its ECE of 0.22
  shows the model overconfident there — the honest picture of the current
  out-of-domain behaviour.
- **Choice holds at ~91%** on both unseen intent taxonomies while the training
  mixture got strictly harder (25-option cardinality, balanced primitives).
- **SST-5 exact match improves to 52.22%** (from 49.00%) with MAE down to 0.572
  and off-by-one at 95.20% — mixed-cardinality Score sources pay off.
- **API Determinism & Schema Conformance**: Out-of-schema error rate = **0.0%**
  (zero risk of JSON malformation or syntax hallucination).
- **Inference Speed**: 35.2 ms median vs ~300 ms for standard autoregressive
  JSON decoding (**~8.5x faster**), 21.5 q/s sustained with $k=3$ debiasing.
