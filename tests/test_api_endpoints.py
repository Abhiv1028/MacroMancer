"""API-level tests using FastAPI's TestClient (no running server needed)."""

from __future__ import annotations

import pytest

from tests.conftest import xgboost_available


def _create_user(client) -> int:
    resp = client.post(
        "/users",
        json={
            "age": 30,
            "weight_kg": 75,
            "height_cm": 180,
            "activity_level": "moderate",
            "goal": "cut",
            "sex": "male",
        },
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["targets"]["protein_g"] == pytest.approx(165.0, abs=0.1)
    return body["user"]["id"]


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_create_user_and_targets(client):
    user_id = _create_user(client)

    # Single day (default) -> list of length 1.
    resp = client.get(f"/users/{user_id}/targets")
    assert resp.status_code == 200
    targets = resp.json()
    assert isinstance(targets, list) and len(targets) == 1

    # A week's worth pre-generated in one call.
    resp = client.get(f"/users/{user_id}/targets", params={"days": 7})
    assert resp.status_code == 200
    assert len(resp.json()) == 7


def test_user_not_found(client):
    assert client.get("/users/999999/targets").status_code == 404


def test_create_and_search_food(client):
    resp = client.post(
        "/foods",
        json={
            "name": "Test Chicken Breast",
            "protein_per_100g": 31,
            "carbs_per_100g": 0,
            "fat_per_100g": 3.6,
            "portions": [{"description": "1 breast", "gram_weight": 174}],
        },
    )
    assert resp.status_code == 201, resp.text
    food = resp.json()
    assert food["is_custom"] is True
    # Energy auto-computed: 31*4 + 0 + 3.6*9 = 156.4
    energy = next(n for n in food["nutrients"] if n["nutrient_name"] == "Energy")
    assert energy["amount_per_100g"] == pytest.approx(156.4, abs=0.1)

    found = client.get("/foods", params={"search": "test chicken"}).json()
    assert any(f["name"] == "Test Chicken Breast" for f in found)

    # Pagination: limit=1 returns at most one result.
    assert len(client.get("/foods", params={"limit": 1}).json()) <= 1


def test_log_meal_computes_macros(client):
    user_id = _create_user(client)
    food = client.post(
        "/foods",
        json={
            "name": "Meal Test Rice",
            "protein_per_100g": 2.7,
            "carbs_per_100g": 28,
            "fat_per_100g": 0.3,
        },
    ).json()

    resp = client.post(
        "/meals",
        json={
            "user_id": user_id,
            "food_id": food["id"],
            "grams": 200,
            "meal_type": "lunch",
        },
    )
    assert resp.status_code == 201, resp.text
    meal = resp.json()
    # 200g => 2x the per-100g values.
    assert meal["protein_g"] == pytest.approx(5.4, abs=0.01)
    assert meal["carbs_g"] == pytest.approx(56.0, abs=0.01)
    assert meal["fat_g"] == pytest.approx(0.6, abs=0.01)


def test_meal_invalid_type_rejected(client):
    user_id = _create_user(client)
    food = client.post(
        "/foods",
        json={"name": "X", "protein_per_100g": 1, "carbs_per_100g": 1, "fat_per_100g": 1},
    ).json()
    resp = client.post(
        "/meals",
        json={"user_id": user_id, "food_id": food["id"], "grams": 100, "meal_type": "brunch"},
    )
    assert resp.status_code == 422


@pytest.mark.skipif(
    not xgboost_available(),
    reason="XGBoost/OpenMP runtime not available in this environment",
)
def test_optimize_returns_ranked_recommendations(client):
    user_id = _create_user(client)
    ids = []
    for spec in [
        {"name": "Opt Chicken", "protein_per_100g": 31, "carbs_per_100g": 0, "fat_per_100g": 3.6},
        {"name": "Opt Rice", "protein_per_100g": 2.7, "carbs_per_100g": 28, "fat_per_100g": 0.3},
        {"name": "Opt Broccoli", "protein_per_100g": 2.8, "carbs_per_100g": 7, "fat_per_100g": 0.4},
    ]:
        ids.append(client.post("/foods", json=spec).json()["id"])

    resp = client.post(
        "/optimize",
        json={
            "user_id": user_id,
            "current_macros": {"protein_g": 140, "carbs_g": 240, "fat_g": 50},
            "meal_type": "dinner",
            "available_food_ids": ids,
        },
    )
    assert resp.status_code == 200, resp.text
    recs = resp.json()["recommendations"]
    assert 1 <= len(recs) <= 5
    # Scores must be sorted descending.
    scores = [r["predicted_score"] for r in recs]
    assert scores == sorted(scores, reverse=True)
    for r in recs:
        assert set(r["macros_per_serving"]) == {"protein_g", "carbs_g", "fat_g", "calories"}
        assert r["suggested_grams"] > 0
