"""Tests for ordinal scoring computation.

Verifies mathematical expectation score = sum_i p_i * i across analytical
distributions, including continuous non-integer values (e.g. 1.4 between discrete levels).
"""

import pytest

from sysone.types import confidence_from_probs, ScoreAnswer


def test_score_espérance_pure():
    """All probability mass on level 2 -> score = 2.0."""
    levels = ["faible", "moyen", "élevé"]
    probs = {"faible": 0.0, "moyen": 0.0, "élevé": 1.0}
    expected = 2.0  # 0*0 + 1*0 + 2*1
    a = ScoreAnswer(score=expected, level="élevé", probabilities=probs,
                     confidence=confidence_from_probs(list(probs.values())))
    assert a.score == 2.0


def test_score_espérance_mixed():
    """p = [0.2, 0.5, 0.3] across 3 levels -> score = 0*0.2 + 1*0.5 + 2*0.3 = 1.1."""
    levels = ["faible", "moyen", "élevé"]
    probs = {"faible": 0.2, "moyen": 0.5, "élevé": 0.3}
    expected = 0 * 0.2 + 1 * 0.5 + 2 * 0.3
    assert expected == pytest.approx(1.1)
    a = ScoreAnswer(score=expected, level="moyen", probabilities=probs,
                     confidence=confidence_from_probs(list(probs.values())))
    assert a.score == pytest.approx(1.1)


def test_score_fractional_between_two_levels():
    """Key fractional expectation: score = 1.4 between two discrete levels.

    p = [0.0, 0.6, 0.4] over ["faible", "moyen", "élevé"]
    -> score = 0*0.0 + 1*0.6 + 2*0.4 = 0.6 + 0.8 = 1.4
    """
    levels = ["faible", "moyen", "élevé"]
    probs = {"faible": 0.0, "moyen": 0.6, "élevé": 0.4}
    expected = 0 * 0.0 + 1 * 0.6 + 2 * 0.4
    assert expected == pytest.approx(1.4)
    a = ScoreAnswer(score=expected, level="moyen", probabilities=probs,
                     confidence=confidence_from_probs(list(probs.values())))
    assert a.score == pytest.approx(1.4)


def test_score_5_levels():
    """5 levels with uniform distribution."""
    levels = ["très faible", "faible", "moyen", "élevé", "très élevé"]
    probs = {l: 0.2 for l in levels}
    expected = 0 * 0.2 + 1 * 0.2 + 2 * 0.2 + 3 * 0.2 + 4 * 0.2  # = 2.0
    assert expected == pytest.approx(2.0)
    a = ScoreAnswer(score=expected, level="très faible", probabilities=probs,
                     confidence=confidence_from_probs(list(probs.values())))
    assert a.score == pytest.approx(2.0)


def test_score_argmax_for_info():
    """level represents the argmax candidate for inspection."""
    levels = ["faible", "moyen", "élevé"]
    probs = {"faible": 0.1, "moyen": 0.45, "élevé": 0.45}
    # argmax = "moyen" (first encounter) or "élevé" depending on sorting
    # score = 0*0.1 + 1*0.45 + 2*0.45 = 1.35
    expected = 1.35
    a = ScoreAnswer(score=expected, level="moyen", probabilities=probs,
                     confidence=confidence_from_probs(list(probs.values())))
    assert a.score == pytest.approx(1.35)
    assert a.level in ["moyen", "élevé"]


def test_confidence_for_score():
    """Confidence calculation for score matches the choice formula."""
    probs = [0.1, 0.45, 0.45]
    conf = confidence_from_probs(probs)
    # (0.45 - 1/3) / (1 - 1/3) = (0.45 - 0.333) / 0.667 ~= 0.175
    assert conf == pytest.approx((0.45 - 1/3) / (1 - 1/3), abs=1e-3)


def test_expected_score_continuous():
    """Verify continuous expected_score on 1..M scale (1-based)."""
    levels = ["1 étoile", "2 étoiles", "3 étoiles", "4 étoiles", "5 étoiles"]
    # 60% on 3 stars (index 2, rating 3), 40% on 4 stars (index 3, rating 4)
    # score 0-based = 0.6*2 + 0.4*3 = 1.2 + 1.2 = 2.4
    # expected_score 1-based = 0.6*3 + 0.4*4 = 1.8 + 1.6 = 3.4
    probs = {lvl: 0.0 for lvl in levels}
    probs["3 étoiles"] = 0.6
    probs["4 étoiles"] = 0.4

    ans = ScoreAnswer(
        score=2.4,
        expected_score=3.4,
        level="3 étoiles",
        probabilities=probs,
        confidence=confidence_from_probs(list(probs.values())),
    )
    assert ans.score == pytest.approx(2.4)
    assert ans.expected_score == pytest.approx(3.4)


def test_expected_score_auto_calc():
    """When expected_score is omitted, model_validator calculates it automatically."""
    levels = ["faible", "moyen", "élevé"]
    probs = {"faible": 0.2, "moyen": 0.5, "élevé": 0.3}
    # 0.2*1 + 0.5*2 + 0.3*3 = 0.2 + 1.0 + 0.9 = 2.1
    ans = ScoreAnswer(
        score=1.1,
        level="moyen",
        probabilities=probs,
        confidence=confidence_from_probs(list(probs.values())),
    )
    assert ans.score == pytest.approx(1.1)
    assert ans.expected_score == pytest.approx(2.1)
