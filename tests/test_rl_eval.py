"""Tests for Phase 6 Part 4: RL evaluation, fallback, override, retrain."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from backend.config import settings
from backend.models import (
    RLAction,
    RLActionLog,
    RLFallbackLog,
    RLReward,
    RLTransition,
)
from backend.services import optimizer, rl_actions, rl_bandit, rl_eval


def _make_user(client) -> int:
    return client.post(
        "/users",
        json={"age": 30, "weight_kg": 80, "height_cm": 180,
              "activity_level": "moderate", "goal": "cut", "sex": "male"},
    ).json()["user"]["id"]


def _action_id(db) -> int:
    rl_actions.seed_actions(db)
    return db.query(RLAction).first().id


def _seed_days(db, uid, action_id, rl_rewards, xgb_rewards):
    """Seed RLReward (+RLActionLog on RL days) on distinct recent dates."""
    today = date.today()
    offset = 0
    for r in rl_rewards:
        day = today - timedelta(days=offset)
        db.add(RLReward(user_id=uid, date=day, reward=r, components={}))
        db.add(RLActionLog(user_id=uid, date=day, action_id=action_id, context_json={}))
        offset += 1
    for r in xgb_rewards:
        day = today - timedelta(days=offset)
        db.add(RLReward(user_id=uid, date=day, reward=r, components={}))
        offset += 1
    db.commit()


# --- evaluate_rl_vs_xgboost --------------------------------------------------
def test_insufficient_data(client, db_session):
    uid = _make_user(client)
    aid = _action_id(db_session)
    _seed_days(db_session, uid, aid, [0.8, 0.8], [0.4, 0.4])  # 2 + 2 < 5
    res = rl_eval.evaluate_rl_vs_xgboost(db_session, uid)
    assert res["recommendation"] == "insufficient_data"
    assert res["rl_days"] == 2 and res["xgboost_days"] == 2


def test_use_rl_when_better(client, db_session):
    uid = _make_user(client)
    aid = _action_id(db_session)
    _seed_days(db_session, uid, aid, [0.8] * 5, [0.4] * 5)
    res = rl_eval.evaluate_rl_vs_xgboost(db_session, uid)
    assert res["recommendation"] == "use_rl"
    assert res["improvement_percent"] == pytest.approx(100.0, abs=0.1)


def test_use_xgboost_when_worse(client, db_session):
    uid = _make_user(client)
    aid = _action_id(db_session)
    _seed_days(db_session, uid, aid, [0.3] * 5, [0.6] * 5)
    res = rl_eval.evaluate_rl_vs_xgboost(db_session, uid)
    assert res["recommendation"] == "use_xgboost"
    assert res["improvement_percent"] == pytest.approx(-50.0, abs=0.1)


# --- resolve_use_rl (override + eval) ----------------------------------------
def test_resolve_default_uses_rl(client, db_session):
    uid = _make_user(client)
    use_rl, reason, _ = rl_eval.resolve_use_rl(db_session, uid)
    assert use_rl is True and reason is None  # insufficient data -> keep exploring


def test_resolve_respects_override_off(client, db_session):
    uid = _make_user(client)
    client.post("/api/v1/rl/switch", json={"user_id": uid, "force_rl": False})
    use_rl, reason, _ = rl_eval.resolve_use_rl(db_session, uid)
    assert use_rl is False and reason == "manual_override"


def test_resolve_falls_back_when_underperforming(client, db_session):
    uid = _make_user(client)
    aid = _action_id(db_session)
    _seed_days(db_session, uid, aid, [0.2] * 5, [0.7] * 5)
    use_rl, reason, _ = rl_eval.resolve_use_rl(db_session, uid)
    assert use_rl is False and reason == "rl_underperforming"


# --- endpoints ---------------------------------------------------------------
def test_evaluate_endpoint(client, db_session):
    uid = _make_user(client)
    aid = _action_id(db_session)
    _seed_days(db_session, uid, aid, [0.8] * 5, [0.4] * 5)
    resp = client.get(f"/api/v1/rl/evaluate/{uid}")
    assert resp.status_code == 200
    assert resp.json()["recommendation"] == "use_rl"


def test_switch_endpoint_toggles(client):
    uid = _make_user(client)
    r1 = client.post("/api/v1/rl/switch", json={"user_id": uid, "force_rl": False})
    assert r1.status_code == 200 and r1.json()["forced_rl"] is False
    r2 = client.post("/api/v1/rl/switch", json={"user_id": uid, "force_rl": True})
    assert r2.json()["forced_rl"] is True  # upsert, not duplicate


def test_recommend_falls_back_with_override(client, db_session, monkeypatch):
    rl_bandit.reset_bandit()
    uid = _make_user(client)
    client.post("/api/v1/rl/switch", json={"user_id": uid, "force_rl": False})
    monkeypatch.setattr(
        optimizer, "get_top_recommendations",
        lambda **kw: [{"food_id": 1, "name": "Chicken", "protein_g": 46, "carbs_g": 0,
                       "fat_g": 5, "calories": 230, "suggested_grams": 150,
                       "predicted_score": 0.9, "strategy_fit": None}],
    )
    resp = client.post("/api/v1/rl/recommend", json={"user_id": uid})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["used_rl"] is False
    assert body["action_name"] == "xgboost"
    assert body["fallback_reason"] == "manual_override"
    assert db_session.query(RLFallbackLog).filter(RLFallbackLog.user_id == uid).count() >= 1


# --- monitoring + offline retrain --------------------------------------------
def test_record_performance(client, db_session):
    uid = _make_user(client)
    evaluation = {"rl_avg_reward": 0.8, "xgboost_avg_reward": 0.4, "improvement_percent": 100.0}
    row = rl_eval.record_performance(db_session, uid, date.today(), evaluation)
    assert row.rl_reward_avg == 0.8 and row.improvement == 100.0


def test_retrain_script_replays_transitions(client, db_session):
    from scripts import retrain_rl

    uid = _make_user(client)
    aid = _action_id(db_session)
    existing = db_session.query(RLTransition).count()
    for i in range(3):
        day = date.today() - timedelta(days=i)
        rew = RLReward(user_id=uid, date=day, reward=0.5, components={})
        al = RLActionLog(
            user_id=uid, date=day, action_id=aid,
            context_json={"protein_remaining_ratio": 0.5, "hour": 0.5},
        )
        db_session.add_all([rew, al])
        db_session.flush()
        db_session.add(
            RLTransition(user_id=uid, action_log_id=al.id, reward_id=rew.id)
        )
    db_session.commit()

    summary = retrain_rl.retrain()
    assert summary["transitions_replayed"] == existing + 3
    assert summary["update_count"] == summary["transitions_replayed"]
    assert settings.RL_BANDIT_PATH.exists()
