"""Tests for evaluation metrics.

Validates ECE, Brier, NLL, ACE, and AURC against known analytical distributions.
"""

import math

import pytest

from eval.metrics import (
    nll, brier_multiclass, ece, ace, risk_coverage_curve, aurc,
    accuracy, latency_stats, compute_all,
)


# ---------------------------------------------------------------------------
# NLL
# ---------------------------------------------------------------------------

def test_nll_perfect():
    """p_correct = 1.0 everywhere -> NLL = 0."""
    assert nll([1.0, 1.0, 1.0]) == pytest.approx(0.0, abs=1e-6)


def test_nll_uniform():
    """p_correct = 0.5 everywhere -> NLL = ln(2) ~= 0.693."""
    assert nll([0.5, 0.5]) == pytest.approx(math.log(2), abs=1e-4)


def test_nll_empty():
    import math
    assert math.isnan(nll([]))


# ---------------------------------------------------------------------------
# Brier
# ---------------------------------------------------------------------------

def test_brier_perfect():
    """Perfect predictions (1.0 on true class) -> Brier = 0."""
    probs = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    labels = [0, 1]
    assert brier_multiclass(probs, labels) == pytest.approx(0.0, abs=1e-6)


def test_brier_uniform_2class():
    """Uniform predictions [0.5, 0.5] over 2 classes -> Brier = 0.5.

    sum_i (p_i - y_i)^2 = (0.5-1)^2 + (0.5-0)^2 = 0.25 + 0.25 = 0.5.
    """
    probs = [[0.5, 0.5]]
    labels = [0]
    assert brier_multiclass(probs, labels) == pytest.approx(0.5, abs=1e-6)


def test_brier_known():
    """Analytical verification: 3 classes, p=[0.7, 0.2, 0.1], label=0.

    (0.7-1)^2 + (0.2-0)^2 + (0.1-0)^2 = 0.09 + 0.04 + 0.01 = 0.14.
    """
    probs = [[0.7, 0.2, 0.1]]
    labels = [0]
    assert brier_multiclass(probs, labels) == pytest.approx(0.14, abs=1e-6)


# ---------------------------------------------------------------------------
# ECE
# ---------------------------------------------------------------------------

def test_ece_perfect_calibration():
    """When confidence == accuracy across every bin -> ECE = 0.

    Constructed with 10 instances of confidence=0.9 and 9 correct predictions (acc=0.9).
    """
    conf = [0.9] * 10
    corr = [1]*9 + [0]*1
    assert ece(conf, corr, n_bins=10) == pytest.approx(0.0, abs=1e-6)


def test_ece_worst_calibration():
    """Completely uncalibrated predictions -> ECE = 1.0.

    confidence = 1.0, accuracy = 0.0 -> |0 - 1| = 1 in the upper bin.
    """
    conf = [1.0, 1.0, 1.0, 1.0]
    corr = [0, 0, 0, 0]
    assert ece(conf, corr, n_bins=5) == pytest.approx(1.0, abs=1e-6)


def test_ece_known():
    """Analytical case: 2 bins, 4 examples.

    Bin [0,0.5): 2 examples conf=0.3, 1 correct -> acc=0.5, conf=0.3
    Bin [0.5,1.0]: 2 examples conf=0.8, 1 correct -> acc=0.5, conf=0.8
    ECE = 0.5*|0.5-0.3| + 0.5*|0.5-0.8| = 0.5*0.2 + 0.5*0.3 = 0.1 + 0.15 = 0.25
    """
    conf = [0.3, 0.3, 0.8, 0.8]
    corr = [1, 0, 1, 0]
    val = ece(conf, corr, n_bins=2)
    assert val == pytest.approx(0.25, abs=1e-6)


def test_ece_15_bins():
    """Verify ECE calculation over 15 bins."""
    conf = [i/30 for i in range(30)]  # 0..~1
    corr = [1 if c > 0.5 else 0 for c in conf]
    val = ece(conf, corr, n_bins=15)
    assert 0.0 <= val <= 1.0


# ---------------------------------------------------------------------------
# ACE (equal-mass)
# ---------------------------------------------------------------------------

def test_ace_perfect_calibration():
    conf = [0.9] * 10
    corr = [1]*9 + [0]*1
    assert ace(conf, corr, n_bins=5) == pytest.approx(0.0, abs=1e-6)


def test_ace_handles_imbalanced():
    """Verify ACE handles highly skewed confidence distributions stably."""
    conf = [0.99] * 90 + [0.1] * 10
    corr = [1] * 85 + [0] * 5 + [1] * 5 + [0] * 5
    val = ace(conf, corr, n_bins=15)
    assert 0.0 <= val <= 1.0


# ---------------------------------------------------------------------------
# AURC
# ---------------------------------------------------------------------------

def test_aurc_perfect():
    """All correct predictions -> risk = 0 everywhere -> AURC = 0."""
    conf = [0.9, 0.8, 0.7, 0.6]
    corr = [1, 1, 1, 1]
    rc = risk_coverage_curve(conf, corr)
    assert rc.aurc == pytest.approx(0.0, abs=1e-6)
    assert rc.error_at_full_coverage == 0.0


def test_aurc_all_wrong():
    """All incorrect predictions -> risk = 1 everywhere -> AURC = 1."""
    conf = [0.9, 0.8, 0.7, 0.6]
    corr = [0, 0, 0, 0]
    rc = risk_coverage_curve(conf, corr)
    assert rc.aurc == pytest.approx(1.0, abs=1e-6)


def test_aurc_ordering_matters():
    """Confidence ranking test: correct high-confidence predictions decrease AURC."""
    conf_good = [0.9, 0.8, 0.7, 0.6]
    corr_good = [1, 1, 0, 0]
    # Identical confidence values but inverted correctness -> confident predictions are wrong
    corr_bad = [0, 0, 1, 1]
    rc_good = risk_coverage_curve(conf_good, corr_good)
    rc_bad = risk_coverage_curve(conf_good, corr_bad)
    assert rc_good.aurc < rc_bad.aurc


def test_aurc_standalone():
    conf = [0.9, 0.5]
    corr = [1, 0]
    assert isinstance(aurc(conf, corr), float)


# ---------------------------------------------------------------------------
# Accuracy / latence
# ---------------------------------------------------------------------------

def test_accuracy():
    assert accuracy([1, 1, 0, 1]) == pytest.approx(0.75)


def test_latency_stats():
    lat = [10, 20, 30, 40, 50]
    stats = latency_stats(lat)
    assert stats.p50 == pytest.approx(30.0)
    assert stats.p95 > 0
    assert stats.throughput_qps > 0


# ---------------------------------------------------------------------------
# compute_all
# ---------------------------------------------------------------------------

def test_compute_all():
    probs = [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]]
    labels = [0, 1]
    m = compute_all(probs, labels)
    assert m.n == 2
    assert 0.0 <= m.ece <= 1.0
    assert 0.0 <= m.accuracy <= 1.0
    assert m.nll >= 0.0
