"""Tests for Phase 6 Part 2: LinUCB bandit, action mapping, and /rl/recommend."""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from backend.models import RLActionLog
from backend.services import optimizer, rl_actions, rl_bandit
from backend.services.rl_bandit import EPSILON_MIN, LinearUCB


def _make_user(client) -> int:
    return client.post(
        "/users",
        json={"age": 30, "weight_kg": 80, "height_cm": 180,
              "activity_level": "moderate", "goal": "cut", "sex": "male"},
    ).json()["user"]["id"]


# --- LinearUCB core ----------------------------------------------------------
def test_init_shapes():
    b = LinearUCB(n_actions=4, context_dim=7)
    assert len(b.A) == 4 and b.A[0].shape == (7, 7)
    assert b.b[0].shape == (7,) and b.theta[0].shape == (7,)


def test_select_action_returns_valid_index():
    b = LinearUCB(n_actions=4, context_dim=3, seed=0)
    ctx = np.array([0.5, 0.2, 0.9])
    for _ in range(20):
        a = b.select_action(ctx)
        assert 0 <= a < 4


def test_best_ucb_action_is_deterministic():
    b = LinearUCB(n_actions=4, context_dim=3)
    ctx = np.array([0.1, 0.2, 0.3])
    # All arms identical at init -> argmax returns the first.
    assert b.best_ucb_action(ctx) == 0


def test_epsilon_decays_with_updates():
    b = LinearUCB(n_actions=4, context_dim=3, seed=1)
    assert b.epsilon() == pytest.approx(1.0)
    ctx = np.ones(3)
    for _ in range(100):
        b.update(0, ctx, 1.0)
    # 1/sqrt(101) ~= 0.0995 -> decayed but not yet at the floor.
    assert EPSILON_MIN < b.epsilon() < 1.0
    for _ in range(400):  # 500 total -> 1/sqrt(501) < 0.05 -> clamped to floor
        b.update(0, ctx, 1.0)
    assert b.epsilon() == pytest.approx(EPSILON_MIN, abs=1e-9)


def test_update_modifies_weights_and_count():
    b = LinearUCB(n_actions=2, context_dim=3, seed=2)
    ctx = np.array([1.0, 0.0, 0.5])
    theta_before = b.theta[0].copy()
    a_before = b.A[0].copy()
    b.update(0, ctx, reward=1.0)
    assert b.update_count == 1
    assert not np.allclose(b.theta[0], theta_before)
    assert not np.allclose(b.A[0], a_before)
    # Untouched arm stays at init.
    assert np.allclose(b.theta[1], np.zeros(3))


def test_update_rewards_align_action_with_context():
    """After rewarding action 0 for a context, its UCB should lead there."""
    b = LinearUCB(n_actions=3, context_dim=2, alpha=0.0, seed=3)
    ctx = np.array([1.0, 0.0])
    for _ in range(30):
        b.update(0, ctx, reward=1.0)
    # alpha=0 -> pure exploitation; action 0 was consistently rewarded.
    assert b.best_ucb_action(ctx) == 0


def test_save_load_roundtrip(tmp_path):
    b = LinearUCB(n_actions=3, context_dim=4, alpha=1.5, seed=4)
    b.update(1, np.array([0.2, 0.4, 0.6, 0.8]), reward=0.7)
    path = tmp_path / "weights.json"
    b.save(path)

    loaded = LinearUCB.load(path)
    assert loaded.n_actions == 3 and loaded.context_dim == 4
    assert loaded.alpha == 1.5 and loaded.update_count == 1
    assert np.allclose(loaded.theta[1], b.theta[1])
    assert np.allclose(loaded.A[1], b.A[1])


def test_cold_start_is_fully_exploratory():
    b = LinearUCB(n_actions=4, context_dim=3, seed=5)
    assert b.update_count == 0
    assert b.epsilon() == pytest.approx(1.0)  # explore w.p. 1


def test_singleton_reset(tmp_path, monkeypatch):
    from backend.config import settings

    monkeypatch.setattr(settings, "RL_BANDIT_PATH", tmp_path / "b.json")
    rl_bandit.reset_bandit()
    b1 = rl_bandit.get_bandit()
    b1.update(0, np.zeros(b1.context_dim), 1.0)
    rl_bandit.save_bandit()
    assert (tmp_path / "b.json").exists()
    rl_bandit.reset_bandit()
    assert not (tmp_path / "b.json").exists()


# --- action mapping ----------------------------------------------------------
def test_apply_action_returns_macros(client, db_session):
    uid = _make_user(client)
    macros = rl_actions.apply_action(db_session, "high_protein", uid, date.today())
    assert set(macros) == {"protein_g", "carbs_g", "fat_g", "calories"}
    assert macros["protein_g"] > 0 and macros["calories"] > 0


# --- endpoint (optimizer mocked) --------------------------------------------
_FAKE_FOODS = [
    {"food_id": 1, "name": "Chicken", "protein_g": 46, "carbs_g": 0, "fat_g": 5,
     "calories": 230, "suggested_grams": 150, "predicted_score": 0.9, "strategy_fit": 0.8},
    {"food_id": 2, "name": "Rice", "protein_g": 4, "carbs_g": 45, "fat_g": 0.5,
     "calories": 210, "suggested_grams": 200, "predicted_score": 0.7, "strategy_fit": 0.6},
]


def test_recommend_endpoint(client, monkeypatch):
    rl_bandit.reset_bandit()
    uid = _make_user(client)
    monkeypatch.setattr(optimizer, "get_top_recommendations", lambda **kw: _FAKE_FOODS)

    resp = client.post("/api/v1/rl/recommend", json={"user_id": uid})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["action_name"] in rl_actions.ACTION_NAMES
    assert set(body["recommended_macros"]) == {"protein_g", "carbs_g", "fat_g", "calories"}
    assert len(body["foods"]) == 2
    assert body["used_rl"] is True


def test_recommend_logs_action(client, db_session, monkeypatch):
    rl_bandit.reset_bandit()
    uid = _make_user(client)
    monkeypatch.setattr(optimizer, "get_top_recommendations", lambda **kw: _FAKE_FOODS)
    client.post("/api/v1/rl/recommend", json={"user_id": uid})
    logs = db_session.query(RLActionLog).filter(RLActionLog.user_id == uid).all()
    assert len(logs) >= 1
    assert "protein_remaining_ratio" in logs[0].context_json


def test_recommend_unknown_user_404(client):
    resp = client.post("/api/v1/rl/recommend", json={"user_id": 999999})
    assert resp.status_code == 404
