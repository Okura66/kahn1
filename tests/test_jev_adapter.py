"""Unit tests for the JEV / TypeSafe schema adapter."""

import pytest
from fastapi.testclient import TestClient

from sysone.types import Query, ChoiceQuestion, ScoreQuestion, NoulQuestion
from sysone.server import app



CANONICAL_JEV_SCHEMA = {
    "category": {
        "type": "choice",
        "instructions": "Determine the broad category of this support ticket",
        "criteria": {
            "bug_report": "The user is reporting something that is broken or producing errors",
            "billing": "Charges, invoices, refunds, subscriptions",
            "feature_request": "The user is requesting new functionality",
            "account": "Login, permissions, profile, security",
        },
    },
    "bug_severity": {
        "type": "score",
        "instructions": "How severe is the reported issue",
        "criteria": [
            "Cosmetic; no impact to functionality",
            "Broken or degraded feature; workaround exists",
            "Blocking issue; no workaround exists",
        ],
    },
    "has_reproducible_steps": {
        "type": "noul",
        "instructions": "The user describes specific steps to reproduce the issue",
    },
    "refund_requested": {
        "type": "noul",
        "instructions": "The user is explicitly asking for a refund or credit",
    },
    "frustration": {
        "type": "score",
        "instructions": "How frustrated the user appears",
        "criteria": [
            "Calm, matter-of-fact",
            "Frustrated but civil",
            "Very angry",
        ],
    },
}


def test_query_from_jev_canonical():
    state = "Je ne peux plus me connecter à mon compte."
    query = Query.from_jev(state=state, schema=CANONICAL_JEV_SCHEMA)

    assert query.state == state
    assert len(query.questions) == 5

    # 1. category
    q_cat = query.questions[0]
    assert isinstance(q_cat, ChoiceQuestion)
    assert q_cat.key == "category"
    assert q_cat.prompt == "Determine the broad category of this support ticket"
    assert len(q_cat.options) == 4
    assert q_cat.options[0].startswith("bug_report:")
    assert q_cat.allow_other is True

    # 2. bug_severity
    q_sev = query.questions[1]
    assert isinstance(q_sev, ScoreQuestion)
    assert q_sev.key == "bug_severity"
    assert len(q_sev.levels) == 3

    # 3. has_reproducible_steps
    q_steps = query.questions[2]
    assert isinstance(q_steps, NoulQuestion)
    assert q_steps.key == "has_reproducible_steps"
    assert "reproduce" in q_steps.statement

    # 4. refund_requested
    q_refund = query.questions[3]
    assert isinstance(q_refund, NoulQuestion)
    assert q_refund.key == "refund_requested"

    # 5. frustration
    q_frust = query.questions[4]
    assert isinstance(q_frust, ScoreQuestion)
    assert q_frust.key == "frustration"
    assert len(q_frust.levels) == 3


def test_query_from_jev_list_choice():
    schema = {
        "sentiment": {
            "type": "choice",
            "instructions": "Sentiment général ?",
            "criteria": ["positif", "neutre", "négatif"],
        }
    }
    query = Query.from_jev(state="Super produit !", schema=schema)
    assert len(query.questions) == 1
    q = query.questions[0]
    assert isinstance(q, ChoiceQuestion)
    assert q.options == ["positif", "neutre", "négatif"]


def test_query_from_jev_invalid_type():
    schema = {
        "invalid": {
            "type": "unsupported_type",
            "instructions": "test",
        }
    }
    with pytest.raises(ValueError, match="Unsupported"):
        Query.from_jev(state="test", schema=schema)


def test_server_jev_integration(monkeypatch):
    """Verify FastAPI evaluation endpoint using native JEV schema."""
    from sysone.types import ChoiceAnswer, ScoreAnswer, NoulAnswer, EvaluateResponse
    import sysone.server as srv

    class FakeEngine:
        def evaluate(self, query: Query, n_permutations: int = 3):
            answers = {}
            for q in query.questions:
                if q.kind == "choice":
                    answers[q.key] = ChoiceAnswer(
                        choice=q.options[0],
                        probabilities={opt: (1.0 if i == 0 else 0.0) for i, opt in enumerate(q.options)},
                        confidence=1.0,
                    )
                elif q.kind == "score":
                    answers[q.key] = ScoreAnswer(
                        score=0.0,
                        level=q.levels[0],
                        probabilities={lvl: (1.0 if i == 0 else 0.0) for i, lvl in enumerate(q.levels)},
                        confidence=1.0,
                    )
                elif q.kind == "noul":
                    answers[q.key] = NoulAnswer(noul=0.9)
            return EvaluateResponse(
                answers=answers,
                latency_ms=42.0,
                cache_hit_rate=0.98,
            )

    monkeypatch.setattr(srv, "get_engine", lambda: FakeEngine())
    monkeypatch.setattr(srv, "_calibrated", None)

    client = TestClient(app)

    # Test via POST /v1/evaluate avec 'schema'
    payload = {
        "state": "Support ticket text",
        "schema": CANONICAL_JEV_SCHEMA,
    }
    res = client.post("/v1/evaluate", json=payload)
    assert res.status_code == 200, res.text
    data = res.json()
    assert len(data["answers"]) == 5
    assert data["cache_hit_rate"] == 0.98

    # Test via POST /v1/evaluate/jev
    res_jev = client.post("/v1/evaluate/jev", json=payload)
    assert res_jev.status_code == 200, res_jev.text
    data_jev = res_jev.json()
    assert len(data_jev["answers"]) == 5
    assert data_jev["answers"]["category"]["choice"].startswith("bug_report:")
