"""Integration tests for the Kahn1 vs Gemini Flash Battle Arena interface."""

from unittest.mock import AsyncMock, patch
import pytest
from fastapi.testclient import TestClient

from sysone.server import app
from sysone.arena import SHOWCASE_PRESETS, get_dataset_index


@pytest.fixture
def client():
    return TestClient(app)


def test_arena_html_endpoint(client):
    """Ensure HTML UI is correctly served on / and /arena endpoints."""
    resp_root = client.get("/")
    assert resp_root.status_code == 200
    assert "Kahn1" in resp_root.text or "kahn1" in resp_root.text
    assert "Gemini" in resp_root.text

    resp_arena = client.get("/arena")
    assert resp_arena.status_code == 200
    assert "16:9" in resp_arena.text
    assert "Multi-Questions" in resp_arena.text


def test_arena_presets_endpoint(client):
    """Verify availability of showcase presets."""
    resp = client.get("/api/arena/presets")
    assert resp.status_code == 200
    presets = resp.json()
    assert len(presets) >= 3
    assert any(p["id"] == "banking_card_lost" for p in presets)
    assert any(p["id"] == "massive_smarthome" for p in presets)


def test_arena_random_sample(client):
    """Verify random sample extraction from holdout set."""
    resp = client.get("/api/arena/sample")
    assert resp.status_code == 200
    data = resp.json()
    assert "state" in data
    assert "options" in data
    assert len(data["options"]) > 0


def test_arena_battle_execution(client):
    """Verify duel battle execution with mocked Gemini response."""
    mock_gemini_resp = {
        "model": "Gemini Flash",
        "mode": "System 2 • Autorégressif (JSON)",
        "choice": "compromised_card",
        "confidence": 0.95,
        "probabilities": {"compromised_card": 0.95},
        "latency_ms": 650.0,
        "tokens_generated": 32,
        "tokens_billed": 110,
        "raw_json": '{"choice": "compromised_card", "confidence": 0.95}',
        "schema_error": False,
        "schema_type": "JSON Text Generation"
    }

    with patch("sysone.arena.run_gemini_prediction", new_callable=AsyncMock) as mock_gemini:
        mock_gemini.return_value = mock_gemini_resp

        payload = {
            "state": "Someone used my card at an ATM in London!",
            "prompt": "Catégorie :",
            "options": ["compromised_card", "lost_card", "order_card"],
            "ground_truth": "compromised_card",
            "preset_id": "banking_card_lost",
        }

        resp = client.post("/api/arena/battle", json=payload)
        assert resp.status_code == 200
        data = resp.json()

        assert "kahn1" in data
        assert "gemini" in data
        assert "jev" in data
        assert data["speedup_factor"] > 1.0
        assert data["kahn1"]["tokens_generated"] == 0
        assert data["gemini"]["tokens_generated"] > 0
        assert "latency_ms" in data["kahn1"]


def test_arena_jev_batch_eval(client):
    """Verify single batch item evaluation for TypeSafe JEV."""
    item = {
        "id": 1,
        "state": "Someone used my card at an ATM in London!",
        "prompt": "Catégorie :",
        "options": ["compromised_card", "lost_card"],
        "ground_truth": "compromised_card",
    }
    resp = client.post("/api/arena/batch/eval_jev", json=item)
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == 1
    assert "choice" in data
    assert data["choice"] in item["options"]
    assert "latency_ms" in data
    assert data["tokens_generated"] == 0


def test_arena_schema_presets_and_battle(client):
    """Verify multi-question schema presets and schema battle execution."""
    resp_presets = client.get("/api/arena/schema_presets")
    assert resp_presets.status_code == 200
    presets = resp_presets.json()
    assert len(presets) >= 3
    assert presets[0]["id"] == "fintech_fraud_escalation"
    assert len(presets[0]["questions"]) >= 5

    # Test schema battle execution with default multiplier (x1 -> 6 Qs)
    p = presets[0]
    payload = {
        "state": p["state"],
        "questions": p["questions"],
        "preset_id": p["id"],
        "include_jev": True,
        "multiplier": 1,
    }
    resp_battle = client.post("/api/arena/schema_battle", json=payload)
    assert resp_battle.status_code == 200
    battle_data = resp_battle.json()

    assert "kahn1" in battle_data
    assert "gemini" in battle_data
    assert "jev" in battle_data
    assert battle_data["speedup_factor"] > 1.0
    assert battle_data["multiplier"] == 1
    assert battle_data["total_questions"] == 6
    assert len(battle_data["kahn1"]["answers"]) == 6
    assert battle_data["kahn1"]["tokens_generated"] == 0

    # Verify telemetry breakdown
    assert "telemetry" in battle_data
    telem = battle_data["telemetry"]
    assert "kahn1" in telem and "gemini" in telem and "jev" in telem
    assert telem["kahn1"]["network_io_ms"] == 0.0
    assert telem["kahn1"]["compute_ms"] > 0
    assert telem["gemini"]["network_io_ms"] > 0
    assert telem["gemini"]["tokens_generated"] > 0

    # Test multiplier x2 (12 Qs)
    payload["multiplier"] = 2
    resp_x2 = client.post("/api/arena/schema_battle", json=payload)
    assert resp_x2.status_code == 200
    data_x2 = resp_x2.json()
    assert data_x2["multiplier"] == 2
    assert data_x2["total_questions"] == 12
    assert len(data_x2["kahn1"]["answers"]) == 12


