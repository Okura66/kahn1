"""Evaluation metrics module.

All metrics are benchmarked with unit tests over canonical reference cases (see tests/test_metrics.py).

  - NLL: -mean(log p_correct)
  - Multi-class Brier score: mean(Σ_i (p_i - y_i)²)
  - ECE: Expected Calibration Error with 15 equal-width bins
  - ACE: Adaptive Calibration Error with equal-mass quantile bins (robust to confidence skew)
  - Reliability diagram generation (matplotlib export)
  - Risk-coverage curve & AURC: samples sorted by descending confidence, plotting cumulative risk
    as a function of coverage. Primary evaluation metric.
  - Latency percentiles (p50/p95) and throughput (queries/second).
  - Ordinal evaluation metrics: MAE, off-by-one accuracy, Spearman rank correlation rho, midpoint bias rate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np


# ---------------------------------------------------------------------------
# NLL
# ---------------------------------------------------------------------------

def nll(probs_correct: Sequence[float]) -> float:
    """Compute Negative Log-Likelihood: NLL = -mean(log p_correct)."""
    if not probs_correct:
        return float("nan")
    total = 0.0
    for p in probs_correct:
        p = max(float(p), 1e-12)
        total += -math.log(p)
    return total / len(probs_correct)


# ---------------------------------------------------------------------------
# Multi-class Brier Score
# ---------------------------------------------------------------------------

def brier_multiclass(
    all_probs: Sequence[Sequence[float]],
    all_labels: Sequence[int],
) -> float:
    """Compute multi-class Brier score: mean(Σ_i (p_i - y_i)²), where y_i = 1 if i == label else 0."""
    if not all_probs:
        return float("nan")
    total = 0.0
    for probs, label in zip(all_probs, all_labels):
        for i, p in enumerate(probs):
            y = 1.0 if i == label else 0.0
            total += (float(p) - y) ** 2
    return total / len(all_probs)


# ---------------------------------------------------------------------------
# ECE — Equal-width bins (15 bins default)
# ---------------------------------------------------------------------------

def ece(
    confidences: Sequence[float],
    corrects: Sequence[int],  # 1 if correct, 0 otherwise
    n_bins: int = 15,
) -> float:
    """Expected Calibration Error with uniform equal-width bins.

    ECE = Σ_b (|bin_b| / N) × |acc(bin_b) - conf(bin_b)|
    """
    confidences = np.array(confidences, dtype=float)
    corrects = np.array(corrects, dtype=float)
    n = len(confidences)
    if n == 0:
        return float("nan")
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece_val = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() == 0:
            continue
        acc = corrects[mask].mean()
        conf = confidences[mask].mean()
        ece_val += (mask.sum() / n) * abs(acc - conf)
    return float(ece_val)


# ---------------------------------------------------------------------------
# ACE — Equal-mass quantile bins
# ---------------------------------------------------------------------------

def ace(
    confidences: Sequence[float],
    corrects: Sequence[int],
    n_bins: int = 15,
) -> float:
    """Adaptive Calibration Error with equal-mass quantile bins.

    Sorts predictions by ascending confidence and partitions them into n_bins
    equal-cardinality groups to compute |acc - conf| per bin. Provides increased
    statistical robustness when confidence distributions are highly skewed toward 1.0.
    """
    confidences = np.array(confidences, dtype=float)
    corrects = np.array(corrects, dtype=float)
    n = len(confidences)
    if n == 0:
        return float("nan")

    # If all confidence values are identical, treat as a single homogeneous bin
    if np.all(confidences == confidences[0]):
        return float(abs(corrects.mean() - confidences.mean()))

    order = np.argsort(confidences)
    conf_sorted = confidences[order]
    correct_sorted = corrects[order]
    bin_size = n // n_bins
    if bin_size == 0:
        n_bins = n
        bin_size = 1
    ace_val = 0.0
    for i in range(n_bins):
        start = i * bin_size
        end = (i + 1) * bin_size if i < n_bins - 1 else n
        if start >= end:
            continue
        acc = correct_sorted[start:end].mean()
        conf = conf_sorted[start:end].mean()
        ace_val += ((end - start) / n) * abs(acc - conf)
    return float(ace_val)


# ---------------------------------------------------------------------------
# AURC — Risk-Coverage Curve
# ---------------------------------------------------------------------------

@dataclass
class RiskCoverage:
    coverages: np.ndarray
    risks: np.ndarray
    aurc: float
    # Error rate at 100% full coverage
    error_at_full_coverage: float


def risk_coverage_curve(
    confidences: Sequence[float],
    corrects: Sequence[int],
) -> RiskCoverage:
    """Compute risk-coverage curve and Area Under Risk-Coverage curve (AURC).

    Sorts predictions in DESCENDING order of confidence. At each coverage level c in (0, 1],
    risk corresponds to the error rate evaluated over the top c*N highest-confidence samples.
    AURC represents the integral area under risk(c).

    Serves as the primary selective prediction metric, modelling thresholded decision making
    and escalation workflows.
    """
    confidences = np.array(confidences, dtype=float)
    corrects = np.array(corrects, dtype=float)
    n = len(confidences)
    if n == 0:
        return RiskCoverage(
            coverages=np.array([]), risks=np.array([]),
            aurc=float("nan"), error_at_full_coverage=float("nan"),
        )
    order = np.argsort(-confidences)  # Descending order
    correct_sorted = corrects[order]
    # Cumulative risk: error rate at each prefix step = (1 - cumulative accuracy)
    cum_correct = np.cumsum(correct_sorted)
    cum_count = np.arange(1, n + 1)
    acc_cum = cum_correct / cum_count
    risks = 1.0 - acc_cum
    coverages = cum_count / n

    # Conventionally, risk at zero coverage matches initial top-1 sample risk risks[0].
    # Prepended (0, risks[0]) coordinates permit closed integration over [0, 1].
    cov_ext = np.concatenate(([0.0], coverages))
    risk_ext = np.concatenate(([risks[0]], risks))

    # AURC = trapezoidal integration across [0, 1]
    try:
        from scipy.integrate import trapezoid
        aurc = float(trapezoid(risk_ext, cov_ext))
    except (ImportError, AttributeError):
        aurc = float(np.trapz(risk_ext, cov_ext))

    return RiskCoverage(
        coverages=coverages, risks=risks,
        aurc=aurc,
        error_at_full_coverage=float(risks[-1]),
    )


def aurc(confidences: Sequence[float], corrects: Sequence[int]) -> float:
    """Compute scalar Area Under Risk-Coverage curve (AURC)."""
    return risk_coverage_curve(confidences, corrects).aurc


# ---------------------------------------------------------------------------
# Accuracy at 100% Coverage
# ---------------------------------------------------------------------------

def accuracy(corrects: Sequence[int]) -> float:
    """Compute overall classification accuracy over binary correctness indicators."""
    if not corrects:
        return float("nan")
    return float(np.mean(corrects))


# ---------------------------------------------------------------------------
# Latency and Throughput Statistics
# ---------------------------------------------------------------------------

@dataclass
class LatencyStats:
    p50: float
    p95: float
    throughput_qps: float  # Queries per second


def latency_stats(latencies_ms: Sequence[float], n_questions: int = 1) -> LatencyStats:
    """Compute median, 95th percentile latency (ms), and throughput (QPS)."""
    arr = np.array(latencies_ms, dtype=float)
    if len(arr) == 0:
        return LatencyStats(0, 0, 0)
    p50 = float(np.percentile(arr, 50))
    p95 = float(np.percentile(arr, 95))
    mean_ms = float(arr.mean())
    qps = n_questions / (mean_ms / 1000.0) if mean_ms > 0 else 0.0
    return LatencyStats(p50=p50, p95=p95, throughput_qps=qps)


# ---------------------------------------------------------------------------
# Summary Bundle
# ---------------------------------------------------------------------------

@dataclass
class MetricsBundle:
    nll: float
    brier: float
    ece: float
    ace: float
    aurc: float
    accuracy: float
    n: int


def compute_all(
    all_probs: Sequence[Sequence[float]],
    all_labels: Sequence[int],
    confidences: Sequence[float] | None = None,
    corrects: Sequence[int] | None = None,
) -> MetricsBundle:
    """Compute full suite of calibration and selective prediction metrics.

    If confidences or corrects are None, they are derived automatically from
    all_probs and all_labels (confidence = max(p), correct = argmax == label).
    """
    if confidences is None:
        confidences = [max(probs) for probs in all_probs]
    if corrects is None:
        corrects = [
            1 if max(range(len(p)), key=lambda i: p[i]) == lab
            else 0
            for p, lab in zip(all_probs, all_labels)
        ]
    probs_correct = [
        float(probs[lab]) if lab < len(probs) and probs[lab] > 0 else 1e-12
        for probs, lab in zip(all_probs, all_labels)
    ]
    return MetricsBundle(
        nll=nll(probs_correct),
        brier=brier_multiclass(all_probs, all_labels),
        ece=ece(confidences, corrects),
        ace=ace(confidences, corrects),
        aurc=aurc(confidences, corrects),
        accuracy=accuracy(corrects),
        n=len(all_labels),
    )


# ---------------------------------------------------------------------------
# Ordinal Calibration and Ranking Metrics
# ---------------------------------------------------------------------------

def off_by_one_accuracy(
    predictions: Sequence[int | float],
    labels: Sequence[int],
) -> float:
    """Compute tolerance accuracy allowing an absolute distance error <= 1 level."""
    if not predictions or not labels:
        return float("nan")
    corrects = [1 if abs(round(p) - y) <= 1 else 0 for p, y in zip(predictions, labels)]
    return float(np.mean(corrects))


def mean_absolute_error(
    predictions: Sequence[int | float],
    labels: Sequence[int | float],
) -> float:
    """Compute Mean Absolute Error (MAE) between predictions and ground-truth ordinal levels."""
    if not predictions or not labels:
        return float("nan")
    diffs = [abs(float(p) - float(y)) for p, y in zip(predictions, labels)]
    return float(np.mean(diffs))


def spearman_correlation(
    predictions: Sequence[int | float],
    labels: Sequence[int | float],
) -> tuple[float, float]:
    """Compute Spearman rank correlation coefficient rho and two-tailed p-value."""
    if len(predictions) < 2 or len(labels) < 2:
        return float("nan"), float("nan")
    if len(set(predictions)) <= 1 or len(set(labels)) <= 1:
        return 0.0, 1.0
    from scipy.stats import spearmanr
    res = spearmanr(predictions, labels)
    return float(res.statistic), float(res.pvalue)


def midpoint_bias_rate(
    predictions: Sequence[int | float],
    n_classes: int = 5,
) -> float:
    """Compute empirical prediction frequency of the exact central median class."""
    if not predictions:
        return float("nan")
    mid = n_classes // 2
    count = sum(1 for p in predictions if round(p) == mid)
    return float(count / len(predictions))


@dataclass
class OrdinalMetricsBundle:
    accuracy: float
    off_by_one_acc: float
    mae_discrete: float
    mae_continuous: float
    spearman_rho: float
    spearman_pvalue: float
    midpoint_bias: float
    n: int


def compute_ordinal_metrics(
    all_probs: Sequence[Sequence[float]],
    all_labels: Sequence[int],
    expected_scores: Sequence[float] | None = None,
) -> OrdinalMetricsBundle:
    """Compute comprehensive ordinal evaluation metrics for Score primitive tasks."""
    n = len(all_labels)
    if n == 0:
        return OrdinalMetricsBundle(
            accuracy=float("nan"),
            off_by_one_acc=float("nan"),
            mae_discrete=float("nan"),
            mae_continuous=float("nan"),
            spearman_rho=float("nan"),
            spearman_pvalue=float("nan"),
            midpoint_bias=float("nan"),
            n=0,
        )
    preds = [int(np.argmax(p)) for p in all_probs]
    n_classes = len(all_probs[0]) if all_probs else 5

    if expected_scores is None:
        expected_scores = [
            sum(p[i] * i for i in range(len(p)))
            for p in all_probs
        ]

    acc = float(np.mean([1 if p == y else 0 for p, y in zip(preds, all_labels)]))
    obo = off_by_one_accuracy(preds, all_labels)
    mae_d = mean_absolute_error(preds, all_labels)
    mae_c = mean_absolute_error(expected_scores, all_labels)
    rho, pval = spearman_correlation(expected_scores, all_labels)
    mid_bias = midpoint_bias_rate(preds, n_classes)

    return OrdinalMetricsBundle(
        accuracy=acc,
        off_by_one_acc=obo,
        mae_discrete=mae_d,
        mae_continuous=mae_c,
        spearman_rho=rho,
        spearman_pvalue=pval,
        midpoint_bias=mid_bias,
        n=n,
    )
