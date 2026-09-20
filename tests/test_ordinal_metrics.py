"""Tests for ordinal evaluation metrics.

Verifies calculations on synthetic distributions and known vectors:
- off_by_one_accuracy
- mean_absolute_error (discrete and continuous expectation)
- spearman_correlation
- midpoint_bias_rate
- compute_ordinal_metrics
"""

import pytest

from eval.metrics import (
    off_by_one_accuracy,
    mean_absolute_error,
    spearman_correlation,
    midpoint_bias_rate,
    compute_ordinal_metrics,
)


def test_off_by_one_accuracy_perfect():
    """All predictions exact or within distance <= 1 -> 100%."""
    preds = [0, 1, 2, 4]
    labels = [0, 2, 1, 3]  # distances: 0, 1, 1, 1
    assert off_by_one_accuracy(preds, labels) == 1.0


def test_off_by_one_accuracy_mixed():
    """Distances: 0 (<=1), 2 (>1), 1 (<=1), 3 (>1) -> 2/4 = 50%."""
    preds = [0, 0, 2, 4]
    labels = [0, 2, 3, 1]
    assert off_by_one_accuracy(preds, labels) == 0.5


def test_mean_absolute_error():
    """MAE = mean(|p - y|)."""
    preds = [1.0, 2.5, 4.0]
    labels = [1.0, 2.0, 3.0]
    # |1-1|=0, |2.5-2|=0.5, |4-3|=1.0 -> mean = 1.5 / 3 = 0.5
    assert mean_absolute_error(preds, labels) == pytest.approx(0.5)


def test_spearman_correlation_monotonic():
    """Perfect rank correlation rho=1.0."""
    preds = [1.2, 2.1, 3.8, 4.5]
    labels = [0, 1, 2, 3]
    rho, pval = spearman_correlation(preds, labels)
    assert rho == pytest.approx(1.0)
    assert pval < 0.05


def test_spearman_correlation_inverted():
    """Perfect negative rank correlation rho=-1.0."""
    preds = [5.0, 4.0, 2.0, 1.0]
    labels = [0, 1, 2, 3]
    rho, _ = spearman_correlation(preds, labels)
    assert rho == pytest.approx(-1.0)


def test_spearman_correlation_constant():
    """Constant predictions return rho=0.0 gracefully without division by zero."""
    preds = [2.0, 2.0, 2.0, 2.0]
    labels = [0, 1, 2, 3]
    rho, _ = spearman_correlation(preds, labels)
    assert rho == 0.0


def test_midpoint_bias_rate():
    """For 5 classes (midpoint = 2), frequency of predicting index 2."""
    preds = [2, 2, 2, 1]  # 3 out of 4 = 75%
    assert midpoint_bias_rate(preds, n_classes=5) == 0.75


def test_compute_ordinal_metrics_bundle():
    """Verify full bundle of ordinal evaluation metrics."""
    # 3 examples over 5 classes
    probs = [
        [0.8, 0.1, 0.1, 0.0, 0.0],  # argmax = 0, E[S] = 0.3
        [0.0, 0.1, 0.8, 0.1, 0.0],  # argmax = 2, E[S] = 2.0
        [0.0, 0.0, 0.1, 0.2, 0.7],  # argmax = 4, E[S] = 3.6
    ]
    labels = [0, 2, 4]

    bundle = compute_ordinal_metrics(probs, labels)
    assert bundle.accuracy == 1.0
    assert bundle.off_by_one_acc == 1.0
    assert bundle.mae_discrete == 0.0
    assert bundle.spearman_rho == pytest.approx(1.0)
    assert bundle.midpoint_bias == pytest.approx(1.0 / 3.0)
    assert bundle.n == 3
