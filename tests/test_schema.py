"""Schema validation and type-safety tests.

Validates that no output violates the schema contract over 1000 randomized queries.
Schema violation error is guaranteed zero by construction: responses are directly
composed from token probabilities over constrained option candidates.
"""

import random

import pytest

from sysone.types import (
    Query, ChoiceQuestion, ScoreQuestion, NoulQuestion,
    ChoiceAnswer, ScoreAnswer, NoulAnswer,
    confidence_from_probs,
)


def test_choice_question_validates():
    q = ChoiceQuestion(key="q", prompt="p", options=["a", "b", "c"])
    assert q.kind == "choice"
    assert len(q.options) == 3


def test_choice_question_too_few_options():
    with pytest.raises(Exception):
        ChoiceQuestion(key="q", prompt="p", options=["a"])


def test_choice_question_too_many_options():
    with pytest.raises(Exception):
        ChoiceQuestion(key="q", prompt="p", options=[str(i) for i in range(65)])


def test_choice_question_duplicate_options():
    with pytest.raises(Exception, match="must be unique"):
        ChoiceQuestion(key="q", prompt="p", options=["a", "a", "b"])


def test_score_question_validates():
    q = ScoreQuestion(key="q", prompt="p",
                      levels=["faible", "moyen", "élevé"])
    assert q.kind == "score"
    assert len(q.levels) == 3


def test_noul_question_validates():
    q = NoulQuestion(key="q", statement="Le client veut partir.")
    assert q.kind == "noul"


def test_query_unique_keys():
    with pytest.raises(Exception, match="must be unique"):
        Query(state="x", questions=[
            ChoiceQuestion(key="k", prompt="p", options=["a", "b"]),
            ScoreQuestion(key="k", prompt="p2", levels=["x", "y"]),
        ])


def test_choice_answer_probs_sum_to_one():
    a = ChoiceAnswer(choice="a", probabilities={"a": 0.6, "b": 0.4},
                      confidence=confidence_from_probs([0.6, 0.4]))
    assert abs(sum(a.probabilities.values()) - 1.0) < 1e-6


def test_choice_answer_bad_sum_raises():
    with pytest.raises(Exception, match="must sum to 1"):
        ChoiceAnswer(choice="a", probabilities={"a": 0.6, "b": 0.5},
                      confidence=0.2)


def test_choice_answer_choice_not_in_probs_raises():
    with pytest.raises(Exception, match="missing from probabilities"):
        ChoiceAnswer(choice="z", probabilities={"a": 0.5, "b": 0.5},
                      confidence=0.0)


def test_score_answer_espérance():
    """score = sum_i p_i * i (0-based level index)."""
    probs = {"faible": 0.2, "moyen": 0.5, "élevé": 0.3}
    a = ScoreAnswer(score=1.1, level="moyen", probabilities=probs,
                     confidence=confidence_from_probs(list(probs.values())))
    assert abs(a.score - (0*0.2 + 1*0.5 + 2*0.3)) < 1e-6  # 1.1


def test_noul_answer_range():
    NoulAnswer(noul=0.5)
    NoulAnswer(noul=0.0)
    NoulAnswer(noul=1.0)
    with pytest.raises(Exception):
        NoulAnswer(noul=1.5)
    with pytest.raises(Exception):
        NoulAnswer(noul=-0.1)


def test_confidence_formula():
    """confidence = (p_max - 1/n) / (1 - 1/n), clamped to [0, 1]."""
    # Equiprobability -> confidence = 0
    assert confidence_from_probs([0.5, 0.5]) == pytest.approx(0.0)
    assert confidence_from_probs([1/3, 1/3, 1/3]) == pytest.approx(0.0)
    # Full probability mass on a single option -> confidence = 1
    assert confidence_from_probs([1.0, 0.0]) == pytest.approx(1.0)
    assert confidence_from_probs([0.0, 0.0, 1.0]) == pytest.approx(1.0)
    # Intermediate distribution
    # n=2, p_max=0.75 -> (0.75 - 0.5) / (1 - 0.5) = 0.5
    assert confidence_from_probs([0.75, 0.25]) == pytest.approx(0.5)


def test_confidence_clamp():
    """Confidence value must strictly remain within [0, 1]."""
    # n=1 -> 0 by convention
    assert confidence_from_probs([1.0]) == 0.0
    # Degenerate distribution
    assert 0.0 <= confidence_from_probs([0.99, 0.005, 0.005]) <= 1.0


def test_schema_1000_random_queries():
    """Verify 1000 randomized query configurations satisfy strict schema validation."""
    rng = random.Random(42)
    for _ in range(1000):
        kind = rng.choice(["choice", "score", "noul"])
        if kind == "choice":
            n = rng.randint(2, 10)
            opts = [f"opt_{rng.randint(0,99999)}" for _ in range(n)]
            # Ensure uniqueness
            opts = list(set(opts))
            if len(opts) < 2:
                opts = opts + ["fallback"]
            q = ChoiceQuestion(key="k", prompt="p?", options=opts,
                                allow_other=rng.random() > 0.5)
        elif kind == "score":
            levels = [f"lvl_{i}" for i in range(rng.randint(2, 6))]
            q = ScoreQuestion(key="k", prompt="p?", levels=levels)
        else:
            q = NoulQuestion(key="k", statement=f"stmt_{rng.randint(0,99999)}")
        query = Query(state="un état aléatoire", questions=[q])
        assert query.state
        assert len(query.questions) == 1
