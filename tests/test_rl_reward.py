"""Tests for Phase 6 Part 1: RL reward, context, actions, and endpoints."""

from __future__ import annotations

from datetime import date, datetime

import pytest

from backend.models import Feedback, MealLog, RLAction, RLReward, RLState, User
from backend.services import rl_actions
from backend.services.macro_calculator import get_or_create_targets
from backend.services.rl_reward import CONTEXT_DIM, build_context, compute_reward


def _make_user(client, weight=75) -> int:
    return client.post(
        "/users",
        json={"age": 30, "weight_kg": weight, "height_cm": 180,
              "activity_level": "moderate", "goal": "cut", "sex": "male"},
    ).json()["user"]["id"]


def _make_food(client) -> int:
    return client.post(
        "/foods",
        json={"name": "RL Test Food", "protein_per_100g": 20,
              "carbs_per_100g": 10, "fat_per_100g": 5},
    ).json()["id"]


def _log_exact(db, user, food_id, day, protein, carbs, fat):
    """Insert a MealLog with exact macros (bypassing food scaling)."""
    db.add(
        MealLog(
            user_id=user.id, food_id=food_id, grams_consumed=100.0,
            meal_type="lunch", timestamp=datetime(day.year, day.month, day.day, 12),
            protein_g=protein, carbs_g=carbs, fat_g=fat,
            calories=protein * 4 + carbs * 4 + fat * 9,
        )
    )
    db.commit()


# --- reward formula ----------------------------------------------------------
def test_perfect_macros_no_feedback_gives_half(client, db_session):
    uid = _make_user(client)
    fid = _make_food(client)
    user = db_session.get(User, uid)
    day = date.today()
    t = get_or_create_targets(db_session, user, day)
    _log_exact(db_session, user, fid, day, t.protein_g, t.carbs_g, t.fat_g)

    reward = compute_reward(db_session, uid, day)
    # macro_adherence == 1.0, no feedback -> 0.5*1 = 0.5
    assert reward == pytest.approx(0.5, abs=0.01)


def test_perfect_macros_max_feedback_gives_one(client, db_session):
    uid = _make_user(client)
    fid = _make_food(client)
    user = db_session.get(User, uid)
    day = date.today()
    t = get_or_create_targets(db_session, user, day)
    _log_exact(db_session, user, fid, day, t.protein_g, t.carbs_g, t.fat_g)
    meal = db_session.query(MealLog).filter(MealLog.user_id == uid).first()
    db_session.add(
        Feedback(user_id=uid, meal_log_id=meal.id, enjoyment=5, satiety=5, energy=5)
    )
    db_session.commit()

    reward = compute_reward(db_session, uid, day)
    # 0.5*1 + 0.3*1 + 0.2*1 = 1.0
    assert reward == pytest.approx(1.0, abs=0.01)


def test_no_meals_gives_zero_adherence(client, db_session):
    uid = _make_user(client)
    day = date.today()
    reward = compute_reward(db_session, uid, day)
    assert reward == pytest.approx(0.0, abs=0.01)
    row = db_session.query(RLReward).filter(RLReward.user_id == uid).one()
    assert row.components["macro_adherence"] == pytest.approx(0.0, abs=0.01)


def test_compute_reward_is_idempotent(client, db_session):
    uid = _make_user(client)
    day = date.today()
    compute_reward(db_session, uid, day)
    compute_reward(db_session, uid, day)
    assert db_session.query(RLReward).filter(RLReward.user_id == uid).count() == 1


def test_components_recorded(client, db_session):
    uid = _make_user(client)
    fid = _make_food(client)
    user = db_session.get(User, uid)
    day = date.today()
    t = get_or_create_targets(db_session, user, day)
    _log_exact(db_session, user, fid, day, t.protein_g / 2, t.carbs_g / 2, t.fat_g / 2)
    compute_reward(db_session, uid, day)
    comp = db_session.query(RLReward).filter(RLReward.user_id == uid).one().components
    assert set(comp) == {"macro_adherence", "feedback_score", "satiety"}
    assert 0.0 <= comp["macro_adherence"] <= 1.0


# --- context builder ---------------------------------------------------------
def test_build_context_shape_and_range(client, db_session):
    uid = _make_user(client)
    day = date.today()
    vec = build_context(db_session, uid, day)
    assert vec.shape == (CONTEXT_DIM,)
    assert all(0.0 <= v <= 1.0 for v in vec)
    state = db_session.query(RLState).filter(RLState.user_id == uid).one()
    assert "protein_remaining_ratio" in state.context_vector


# --- action space ------------------------------------------------------------
def test_actions_seed_idempotent(db_session):
    rl_actions.seed_actions(db_session)
    rl_actions.seed_actions(db_session)  # second call is a no-op
    names = {a.name for a in db_session.query(RLAction).all()}
    assert {"high_protein", "balanced", "low_carb", "high_fat"} <= names


def test_action_split_to_grams():
    grams = rl_actions.split_to_grams("high_protein", 2000)
    # 40% protein of 2000 kcal / 4 = 200 g
    assert grams["protein"] == pytest.approx(200.0, abs=0.1)


# --- endpoints ---------------------------------------------------------------
def test_compute_reward_endpoint(client):
    uid = _make_user(client)
    resp = client.post("/api/v1/rl/compute_reward", json={"user_id": uid})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["user_id"] == uid
    assert "macro_adherence" in body["components"]
    assert "protein_remaining_ratio" in body["context"]


def test_rewards_endpoint_lists(client):
    uid = _make_user(client)
    client.post("/api/v1/rl/compute_reward", json={"user_id": uid})
    resp = client.get(f"/api/v1/rl/rewards/{uid}")
    assert resp.status_code == 200
    assert len(resp.json()) >= 1


def test_compute_reward_unknown_user_404(client):
    resp = client.post("/api/v1/rl/compute_reward", json={"user_id": 999999})
    assert resp.status_code == 404
