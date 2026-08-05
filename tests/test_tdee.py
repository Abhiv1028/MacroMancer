"""Tests for adaptive TDEE: the service formula, fallback, and endpoints.

The endpoint tests mock the TDEE service so they don't depend on seeded history.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from backend.models import BodyComposition, DailySummary
from backend.services import tdee_calculator
from backend.services.tdee_calculator import TDEEResult, calculate_adaptive_tdee


def _make_user(client, weight=75) -> int:
    return client.post(
        "/users",
        json={
            "age": 30,
            "weight_kg": weight,
            "height_cm": 180,
            "activity_level": "moderate",
            "goal": "cut",
            "sex": "male",
        },
    ).json()["user"]["id"]


def test_fallback_when_insufficient_data(client, db_session):
    user_id = _make_user(client)
    result = calculate_adaptive_tdee(db_session, user_id)
    assert result.method_used == "mifflin"
    assert result.tdee > 0
    assert result.last_updated is None


def test_adaptive_formula(client, db_session):
    user_id = _make_user(client)
    today = date.today()

    # 14 days of 2000 kcal intake.
    for i in range(14):
        db_session.add(
            DailySummary(
                user_id=user_id,
                date=today - timedelta(days=13 - i),
                total_calories=2000,
                total_protein=0,
                total_carbs=0,
                total_fat=0,
            )
        )
    # Weight fell 80 -> 79 across the window (1 kg lost).
    db_session.add(
        BodyComposition(user_id=user_id, date=today - timedelta(days=13), weight_kg=80)
    )
    db_session.add(BodyComposition(user_id=user_id, date=today, weight_kg=79))
    db_session.commit()

    result = calculate_adaptive_tdee(db_session, user_id, days=14)
    assert result.method_used == "adaptive"
    # (14*2000 + 1*7700) / 14 = 2550
    assert result.tdee == pytest.approx(2550.0, abs=1.0)
    assert result.last_updated == today


def test_tdee_endpoint_with_mock(client, monkeypatch):
    user_id = _make_user(client)

    def _fake(db, uid, days=14):
        return TDEEResult(tdee=3200.0, method_used="adaptive", last_updated=date.today())

    monkeypatch.setattr(tdee_calculator, "calculate_adaptive_tdee", _fake)

    resp = client.get(f"/api/v1/users/{user_id}/tdee")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["method_used"] == "adaptive"
    assert body["adaptive_tdee"] == 3200.0
    assert body["current_tdee"] > 0


def test_tdee_endpoint_fallback_null_adaptive(client, monkeypatch):
    user_id = _make_user(client)

    def _fake(db, uid, days=14):
        return TDEEResult(tdee=2500.0, method_used="mifflin", last_updated=None)

    monkeypatch.setattr(tdee_calculator, "calculate_adaptive_tdee", _fake)

    body = client.get(f"/api/v1/users/{user_id}/tdee").json()
    assert body["method_used"] == "mifflin"
    assert body["adaptive_tdee"] is None


def test_update_goals_recalculates_targets(client):
    user_id = _make_user(client)
    resp = client.put(f"/api/v1/users/{user_id}/goals", json={"goal": "bulk"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["goal"] == "bulk"
    assert body["targets"]["calories"] > 0
    # No history -> static fallback.
    assert body["method_used"] == "mifflin"


def test_update_goals_invalid_422(client):
    user_id = _make_user(client)
    resp = client.put(f"/api/v1/users/{user_id}/goals", json={"goal": "shred"})
    assert resp.status_code == 422


def test_update_goals_unknown_user_404(client):
    resp = client.put("/api/v1/users/999999/goals", json={"goal": "cut"})
    assert resp.status_code == 404
