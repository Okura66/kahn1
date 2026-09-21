"""Tests for the Kahn1 vs JEV race: scoring, formatting and answer normalization.

All pure logic — no model and no network.
"""

import pytest

from sysone import jev, race
from sysone.types import ChoiceAnswer, NoulAnswer, ScoreAnswer

CHOICE_ITEM = {
    "id": "banking77:1", "source": "banking77", "kind": "choice",
    "state": "How do I locate my card?", "prompt": "What is the customer's intent?",
    "options": ["card_arrival", "card_lost", "top_up"], "gold": "card_arrival",
}
SCORE_ITEM = {
    "id": "sst5:1", "source": "sst5", "kind": "score",
    "state": "a gorgeous, witty film", "prompt": "Rate the sentiment of this review:",
    "levels": ["very negative", "negative", "neutral", "positive", "very positive"],
    "gold": "very positive",
}
NOUL_ITEM = {
    "id": "boolq:1", "source": "boolq", "kind": "noul",
    "state": "The park opened in 1851.", "statement": "Did the park open in 1851?",
    "gold": True,
}


# --- Scoring against ground truth -------------------------------------------

def test_is_correct_per_primitive():
    assert race.is_correct(CHOICE_ITEM, "card_arrival")
    assert not race.is_correct(CHOICE_ITEM, "card_lost")
    assert race.is_correct(SCORE_ITEM, "very positive")
    assert not race.is_correct(SCORE_ITEM, "positive")
    assert race.is_correct(NOUL_ITEM, True)
    assert not race.is_correct(NOUL_ITEM, False)


def test_run_tallies_score_and_per_kind():
    run = race.EngineRun(name="x", n_items=3)
    run.record(CHOICE_ITEM, True, 10.0)
    run.record(SCORE_ITEM, False, 20.0)
    run.record(NOUL_ITEM, True, 30.0)

    assert run.answered == 3 and run.correct == 2
    assert run.score == pytest.approx(200.0 / 3)
    summary = run.summary(1234.0)
    assert summary["by_kind"] == {
        "choice": {"correct": 1, "total": 1},
        "noul": {"correct": 1, "total": 1},
        "score": {"correct": 0, "total": 1},
    }
    assert summary["p50_ms"] == 20.0


def test_empty_run_scores_zero_without_dividing_by_zero():
    assert race.EngineRun(name="x", n_items=5).score == 0.0


# --- Line formatting --------------------------------------------------------

def test_line_key_elides_long_states():
    item = {"state": "word " * 50}
    key = race.line_key(item)
    assert len(key) == race.KEY_WIDTH
    assert key.endswith("…")


def test_line_key_collapses_whitespace():
    assert race.line_key({"state": "a\n  b\tc"}) == "a b c"


def test_render_line_omits_trailing_comma_on_last_item():
    mid = race.render_line(CHOICE_ITEM, {"choice": "card_arrival"}, last=False)
    end = race.render_line(CHOICE_ITEM, {"choice": "card_arrival"}, last=True)
    assert mid == '  "How do I locate my card?": {"choice": "card_arrival"},'
    assert end.endswith("}")
    assert not end.endswith(",")


# --- Kahn1 answer normalization ---------------------------------------------

def test_normalize_kahn1_choice():
    ans = ChoiceAnswer(choice="card_arrival", confidence=0.9,
                       probabilities={"card_arrival": 0.8, "card_lost": 0.2})
    assert race.normalize_kahn1(CHOICE_ITEM, ans)["pick"] == "card_arrival"


def test_normalize_kahn1_score_picks_the_argmax_level():
    probs = dict(zip(SCORE_ITEM["levels"], [0.01, 0.02, 0.03, 0.24, 0.70]))
    ans = ScoreAnswer(score=3.6, level="very positive", probabilities=probs,
                      confidence=0.6, expected_score=4.6)
    assert race.normalize_kahn1(SCORE_ITEM, ans)["pick"] == "very positive"


@pytest.mark.parametrize("p_yes,expected", [(0.9, True), (0.1, False)])
def test_normalize_kahn1_noul_thresholds_at_half(p_yes, expected):
    out = race.normalize_kahn1(NOUL_ITEM, NoulAnswer(noul=p_yes))
    assert out["pick"] is expected


# --- JEV payloads and answer normalization ----------------------------------

def test_question_payload_shapes():
    assert jev.question_payload(CHOICE_ITEM) == {
        "type": "choice",
        "instructions": CHOICE_ITEM["prompt"],
        "criteria": {o: o for o in CHOICE_ITEM["options"]},
    }
    assert jev.question_payload(SCORE_ITEM)["criteria"] == SCORE_ITEM["levels"]
    assert jev.question_payload(NOUL_ITEM) == {
        "type": "noul", "instructions": NOUL_ITEM["statement"],
    }


def test_question_payload_rejects_unknown_kind():
    with pytest.raises(ValueError, match="Unsupported item kind"):
        jev.question_payload({"kind": "wat"})


def test_normalize_jev_score_uses_probability_argmax_not_the_mean():
    # A bimodal distribution whose mean (2.0) is NOT the most likely level.
    answer = {"type": "score", "score": 2.0, "confidence": 0.5,
              "probabilities": {"0": 0.45, "1": 0.0, "2": 0.1, "3": 0.0, "4": 0.45}}
    assert jev.normalize(SCORE_ITEM, answer)["pick"] in {"very negative", "very positive"}

    peaked = {"type": "score", "score": 3.9, "confidence": 0.9,
              "probabilities": {"0": 0.0, "1": 0.0, "2": 0.0, "3": 0.1, "4": 0.9}}
    assert jev.normalize(SCORE_ITEM, peaked)["pick"] == "very positive"


def test_normalize_jev_score_falls_back_to_the_scalar_and_clamps():
    out = jev.normalize(SCORE_ITEM, {"type": "score", "score": 99.0})
    assert out["pick"] == "very positive"  # clamped to the last level


def test_normalize_jev_choice_and_noul():
    assert jev.normalize(CHOICE_ITEM, {"choice": "card_lost", "confidence": 0.7})["pick"] == "card_lost"
    assert jev.normalize(NOUL_ITEM, {"noul": 0.95})["pick"] is True
    assert jev.normalize(NOUL_ITEM, {"noul": 0.05})["pick"] is False


# --- Credentials ------------------------------------------------------------

def test_missing_key_raises_instead_of_faking_an_answer(monkeypatch):
    monkeypatch.delenv("JEV_API_KEY", raising=False)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    assert jev.has_key() is False
    with pytest.raises(jev.JevKeyMissing):
        jev.api_key()


def test_load_items_reports_how_to_build_the_set(tmp_path):
    with pytest.raises(FileNotFoundError, match="build_race_set"):
        race.load_items(tmp_path / "missing.json")
