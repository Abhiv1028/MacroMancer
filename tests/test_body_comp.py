"""Tests for the body-composition endpoint and its target recalculation."""

from __future__ import annotations

from datetime import date


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


def test_body_comp_updates_weight_and_targets(client):
    user_id = _make_user(client, weight=80)
    before = client.get(f"/users/{user_id}/targets").json()[0]

    resp = client.post(
        "/api/v1/body_composition",
        json={
            "user_id": user_id,
            "weight_kg": 72,
            "body_fat_percent": 15.5,
            "lean_mass_kg": 60.8,
        },
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["weight_kg"] == 72
    assert body["date"] == date.today().isoformat()
    assert "updated_targets" in body

    # The user's canonical weight was updated...
    user = client.get(f"/users/{user_id}").json()
    assert user["weight_kg"] == 72

    # ...and targets shifted (lower weight -> lower protein target for a cut).
    after = body["updated_targets"]
    assert after["protein_g"] != before["protein_g"]
    assert abs(after["protein_g"] - 72 * 2.2) < 0.1  # 2.2 g/kg


def test_body_comp_explicit_date(client):
    user_id = _make_user(client)
    resp = client.post(
        "/api/v1/body_composition",
        json={"user_id": user_id, "date": "2026-06-01", "weight_kg": 74},
    )
    assert resp.status_code == 201
    assert resp.json()["date"] == "2026-06-01"


def test_body_comp_unknown_user_404(client):
    resp = client.post(
        "/api/v1/body_composition",
        json={"user_id": 999999, "weight_kg": 70},
    )
    assert resp.status_code == 404


def test_body_comp_updates_daily_summary(client, db_session):
    from backend.models import DailySummary

    user_id = _make_user(client)
    client.post(
        "/api/v1/body_composition",
        json={"user_id": user_id, "weight_kg": 73.5},
    )
    # Background task should have upserted a summary carrying the weight.
    summary = (
        db_session.query(DailySummary)
        .filter(DailySummary.user_id == user_id)
        .first()
    )
    assert summary is not None
    assert summary.weight_kg == 73.5
