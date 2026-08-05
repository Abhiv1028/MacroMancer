"""Tests for Phase 6 Part 3: online bandit updates and transitions."""

from __future__ import annotations

from datetime import date

import pytest

from backend.models import RLTransition
from backend.services import optimizer, rl_bandit, rl_online

_FAKE_FOODS = [
    {"food_id": 1, "name": "Chicken", "protein_g": 46, "carbs_g": 0, "fat_g": 5,
     "calories": 230, "suggested_grams": 150, "predicted_score": 0.9, "strategy_fit": 0.8},
]


@pytest.fixture(autouse=True)
def _fresh_bandit():
    rl_bandit.reset_bandit()
    yield
    rl_bandit.reset_bandit()


def _make_user(client) -> int:
    return client.post(
        "/users",
        json={"age": 30, "weight_kg": 80, "height_cm": 180,
              "activity_level": "moderate", "goal": "cut", "sex": "male"},
    ).json()["user"]["id"]


def _recommend(client, monkeypatch, uid):
    """Trigger a recommendation so RLState + RLActionLog exist for today."""
    monkeypatch.setattr(optimizer, "get_top_recommendations", lambda **kw: _FAKE_FOODS)
    client.post("/api/v1/rl/recommend", json={"user_id": uid})


# --- update_bandit_from_day --------------------------------------------------
def test_update_skips_without_action_log(client, db_session):
    uid = _make_user(client)
    # Reward exists but no action was ever selected.
    client.post("/api/v1/rl/compute_reward", json={"user_id": uid})
    result = rl_online.update_bandit_from_day(db_session, uid, date.today())
    assert result["status"] == "skipped"
    assert result["reason"] == "no_action_log"


def test_update_skips_without_reward(client, db_session, monkeypatch):
    uid = _make_user(client)
    _recommend(client, monkeypatch, uid)  # action log but no reward computed
    result = rl_online.update_bandit_from_day(db_session, uid, date.today())
    assert result["status"] == "skipped"
    assert result["reason"] == "no_reward"


def test_update_applies_and_saves_transition(client, db_session, monkeypatch):
    uid = _make_user(client)
    _recommend(client, monkeypatch, uid)
    client.post("/api/v1/rl/compute_reward", json={"user_id": uid})

    before = rl_bandit.get_bandit().update_count
    result = rl_online.update_bandit_from_day(db_session, uid, date.today())
    assert result["status"] == "updated"
    assert result["action"] in {"high_protein", "balanced", "low_carb", "high_fat"}
    assert rl_bandit.get_bandit().update_count == before + 1

    tx = db_session.query(RLTransition).filter(RLTransition.user_id == uid).all()
    assert len(tx) == 1
    assert tx[0].reward_id is not None and tx[0].action_log_id is not None


def test_update_is_idempotent(client, db_session, monkeypatch):
    uid = _make_user(client)
    _recommend(client, monkeypatch, uid)
    client.post("/api/v1/rl/compute_reward", json={"user_id": uid})

    first = rl_online.update_bandit_from_day(db_session, uid, date.today())
    assert first["status"] == "updated"
    count_after_first = rl_bandit.get_bandit().update_count
    second = rl_online.update_bandit_from_day(db_session, uid, date.today())
    assert second["status"] == "skipped"
    assert second["reason"] == "already_updated"
    # No extra weight update, no extra transition.
    assert rl_bandit.get_bandit().update_count == count_after_first
    assert db_session.query(RLTransition).filter(RLTransition.user_id == uid).count() == 1


# --- background task ---------------------------------------------------------
def test_task_updates_when_day_complete(client, db_session, monkeypatch):
    uid = _make_user(client)
    _recommend(client, monkeypatch, uid)
    rl_online.reward_and_update_task(uid, date.today(), day_complete=True)
    assert db_session.query(RLTransition).filter(RLTransition.user_id == uid).count() == 1


def test_task_reward_only_when_incomplete(client, db_session, monkeypatch):
    uid = _make_user(client)
    _recommend(client, monkeypatch, uid)
    rl_online.reward_and_update_task(uid, date.today(), day_complete=False)
    # Reward computed, but no transition (no online update).
    assert db_session.query(RLTransition).filter(RLTransition.user_id == uid).count() == 0


# --- endpoints ---------------------------------------------------------------
def test_manual_update_endpoint(client, monkeypatch):
    uid = _make_user(client)
    _recommend(client, monkeypatch, uid)
    client.post("/api/v1/rl/compute_reward", json={"user_id": uid})

    resp = client.post("/api/v1/rl/update", json={"user_id": uid})
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "updated"
    # Second call is a no-op.
    assert client.post("/api/v1/rl/update", json={"user_id": uid}).json()["status"] == "skipped"


def test_status_endpoint(client, monkeypatch):
    uid = _make_user(client)
    before = client.get("/api/v1/rl/status").json()
    assert before["n_actions"] == 4 and before["context_dim"] == 7
    assert before["epsilon"] <= 1.0

    _recommend(client, monkeypatch, uid)
    client.post("/api/v1/rl/compute_reward", json={"user_id": uid})
    client.post("/api/v1/rl/update", json={"user_id": uid})
    after = client.get("/api/v1/rl/status").json()
    assert after["update_count"] == before["update_count"] + 1


def test_update_unknown_user_404(client):
    resp = client.post("/api/v1/rl/update", json={"user_id": 999999})
    assert resp.status_code == 404
