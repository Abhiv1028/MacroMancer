"""Tests for Phase 4 numerical safety: input bounds (422), clamping, TDEE guards."""

from __future__ import annotations

from datetime import date, timedelta

from backend.models import BodyComposition, DailySummary
from backend.services.macro_calculator import calculate_macro_targets, clamp
from backend.services.tdee_calculator import calculate_adaptive_tdee


# --- input validation returns 422, never 500 ---------------------------------
def test_user_weight_out_of_range_422(client):
    resp = client.post(
        "/users",
        json={"age": 30, "weight_kg": 500, "height_cm": 180,
              "activity_level": "moderate", "goal": "cut"},
    )
    assert resp.status_code == 422


def test_user_age_too_low_422(client):
    resp = client.post(
        "/users",
        json={"age": 5, "weight_kg": 70, "height_cm": 180,
              "activity_level": "moderate", "goal": "cut"},
    )
    assert resp.status_code == 422


def test_food_macro_out_of_range_422(client):
    resp = client.post(
        "/foods",
        json={"name": "Overflow", "protein_per_100g": 5000,
              "carbs_per_100g": 0, "fat_per_100g": 0},
    )
    assert resp.status_code == 422


def test_body_comp_weight_out_of_range_422(client):
    user_id = client.post(
        "/users",
        json={"age": 30, "weight_kg": 75, "height_cm": 180,
              "activity_level": "moderate", "goal": "cut"},
    ).json()["user"]["id"]
    resp = client.post(
        "/api/v1/body_composition",
        json={"user_id": user_id, "weight_kg": 0.5},
    )
    assert resp.status_code == 422


# --- clamp helper ------------------------------------------------------------
def test_clamp_bounds_and_nan():
    assert clamp(50, 0, 100) == 50
    assert clamp(-10, 0, 100) == 0
    assert clamp(999, 0, 100) == 100
    assert clamp(float("nan"), 0, 100) == 0
    assert clamp("bad", 5, 100) == 5  # type: ignore[arg-type]


def test_macro_outputs_stay_within_bounds():
    # Extreme (but schema-permitted at the function level) inputs -> clamped.
    t = calculate_macro_targets(400, 300, 10, "extra", "bulk", "male")
    assert 0 <= t.calories <= 10000
    assert 0 <= t.protein_g <= 1000
    assert 0 <= t.carbs_g <= 1000
    assert 0 <= t.fat_g <= 500


# --- TDEE guards -------------------------------------------------------------
def test_tdee_rejects_extreme_weight_change(client, db_session):
    user_id = client.post(
        "/users",
        json={"age": 30, "weight_kg": 80, "height_cm": 180,
              "activity_level": "moderate", "goal": "cut"},
    ).json()["user"]["id"]
    today = date.today()
    for i in range(10):
        db_session.add(
            DailySummary(
                user_id=user_id, date=today - timedelta(days=9 - i),
                total_calories=2000, total_protein=0, total_carbs=0, total_fat=0,
            )
        )
    # 60 kg swing over 10 days = 6 kg/day -> implausible -> fallback.
    db_session.add(BodyComposition(user_id=user_id, date=today - timedelta(days=9), weight_kg=80))
    db_session.add(BodyComposition(user_id=user_id, date=today, weight_kg=20))
    db_session.commit()

    result = calculate_adaptive_tdee(db_session, user_id, days=14)
    assert result.method_used == "mifflin"
