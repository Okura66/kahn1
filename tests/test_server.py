"""Integration tests for the sysone FastAPI server.

Verifies:
  - GET /health endpoint
  - POST /v1/evaluate endpoint with strict typing across primitives
  - POST /v1/calibrate/load for dynamic temperature scaling reload
  - Absolute adherence to schema boundaries with zero type errors
"""

import json
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from sysone.server import app, get_engine
from sysone.types import (
    ChoiceAnswer, ScoreAnswer, NoulAnswer, EvaluateResponse,
)
from sysone.calibrate import TemperatureConfig


@pytest.fixture
def client():
    return TestClient(app)


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_evaluate_endpoint_mocked(client):
    mock_engine = MagicMock()
    mock_resp = EvaluateResponse(
        answers={
            "sentiment": ChoiceAnswer(
                choice="positif",
                confidence=0.92,
                probabilities={"positif": 0.92, "négatif": 0.08},
            ),
            "clarity": ScoreAnswer(
                level="élevé",
                score=4.2,
                confidence=0.85,
                probabilities={"faible": 0.05, "moyen": 0.10, "élevé": 0.85},
            ),
            "is_valid": NoulAnswer(
                noul=0.97,
            ),

        },
        latency_ms=18.5,
        cache_hit_rate=0.95,
    )
    mock_engine.evaluate.return_value = mock_resp

    with patch("sysone.server.get_engine", return_value=mock_engine), \
         patch("sysone.server._calibrated", None):
        payload = {
            "state": "Cet assistant est remarquablement rapide et fiable.",
            "questions": [
                {
                    "key": "sentiment",
                    "kind": "choice",
                    "prompt": "Quel est le sentiment ?",
                    "options": ["positif", "négatif"],
                },
                {
                    "key": "clarity",
                    "kind": "score",
                    "prompt": "Niveau de clarté",
                    "levels": ["faible", "moyen", "élevé"],
                },
                {
                    "key": "is_valid",
                    "kind": "noul",
                    "statement": "Le texte est compréhensible.",
                },
            ],
            "n_permutations": 3,
        }
        resp = client.post("/v1/evaluate", json=payload)
        assert resp.status_code == 200
        data = resp.json()

        assert "answers" in data
        assert data["latency_ms"] == 18.5
        assert data["cache_hit_rate"] == 0.95

        # Validate typed Choice response
        ans_c = data["answers"]["sentiment"]
        assert ans_c["choice"] == "positif"
        assert ans_c["confidence"] == 0.92
        assert ans_c["probabilities"]["positif"] == 0.92

        # Validate Score response
        ans_s = data["answers"]["clarity"]
        assert ans_s["level"] == "élevé"
        assert ans_s["score"] == 4.2

        # Validate Noul response
        ans_n = data["answers"]["is_valid"]
        assert ans_n["noul"] == 0.97



def test_load_calibration_endpoint(client, tmp_path):
    calib_file = tmp_path / "test_calibration.json"
    cfg = TemperatureConfig(choice=1.35, score=0.95, noul=1.10)
    cfg.save(calib_file)

    resp = client.post(f"/v1/calibrate/load?path={calib_file}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "loaded"
    assert data["config"]["choice"] == 1.35
    assert data["config"]["score"] == 0.95
    assert data["config"]["noul"] == 1.10


@pytest.mark.parametrize("origin, allowed", [
    ("https://kahn1.com", True),
    ("http://localhost:8000", True),
    ("http://127.0.0.1:5500", True),
    # No wildcard: other subdomains, look-alikes and GitHub Pages sites are refused.
    ("https://demo.kahn1.com", False),
    ("https://www.kahn1.com", False),
    ("https://kahn1.com.evil.example", False),
    ("http://kahn1.com", False),
    ("https://someone.github.io", False),
])
def test_cors_allows_only_the_site_and_local_pages(client, origin, allowed):
    resp = client.get("/health", headers={"Origin": origin})
    assert (resp.headers.get("access-control-allow-origin") == origin) is allowed
