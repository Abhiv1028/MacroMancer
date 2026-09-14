"""Tests for the Phase-4 persistence GET/DELETE endpoints (Day-4 additions)."""

from __future__ import annotations

from datetime import date


def _make_user(client) -> int:
    return client.post(
        "/users",
        json={"age": 30, "weight_kg": 80, "height_cm": 180,
              "activity_level": "moderate", "goal": "cut", "sex": "male"},
    ).json()["user"]["id"]


def _make_food(client, name="GE Food") -> int:
    return client.post(
        "/foods",
        json={"name": name, "protein_per_100g": 20, "carbs_per_100g": 10, "fat_per_100g": 5},
    ).json()["id"]


# --- GET /meals + DELETE /meals/{id} ----------------------------------------
def test_list_and_delete_meals(client):
    uid = _make_user(client)
    fid = _make_food(client)
    m1 = client.post("/meals", json={"user_id": uid, "food_id": fid, "grams": 100, "meal_type": "lunch"}).json()
    client.post("/meals", json={"user_id": uid, "food_id": fid, "grams": 150, "meal_type": "dinner"})

    meals = client.get("/meals", params={"user_id": uid}).json()
    assert len(meals) == 2
    # most-recent first
    assert meals[0]["grams_consumed"] in (100.0, 150.0)

    # filter by today
    today = client.get("/meals", params={"user_id": uid, "date": date.today().isoformat()}).json()
    assert len(today) == 2

    # delete one
    resp = client.delete(f"/meals/{m1['id']}")
    assert resp.status_code == 200 and resp.json()["status"] == "deleted"
    assert len(client.get("/meals", params={"user_id": uid}).json()) == 1


def test_list_meals_unknown_user_404(client):
    assert client.get("/meals", params={"user_id": 999999}).status_code == 404


def test_delete_unknown_meal_404(client):
    assert client.delete("/meals/999999").status_code == 404


# --- GET /grocery_lists ------------------------------------------------------
def test_list_grocery_lists(client):
    uid = _make_user(client)
    client.post("/api/v1/grocery_lists", json={"user_id": uid, "name": "L1",
                "meal_plan": [{"name": "Chicken", "grams": 300}]})
    client.post("/api/v1/grocery_lists", json={"user_id": uid, "name": "L2",
                "meal_plan": [{"name": "Rice", "grams": 200}, {"name": "Broccoli", "grams": 100}]})
    lists = client.get("/api/v1/grocery_lists", params={"user_id": uid}).json()
    assert len(lists) == 2
    names = {row["name"] for row in lists}
    assert names == {"L1", "L2"}
    l2 = next(r for r in lists if r["name"] == "L2")
    assert l2["item_count"] == 2


def test_list_grocery_lists_unknown_user_404(client):
    assert client.get("/api/v1/grocery_lists", params={"user_id": 999999}).status_code == 404


# --- GET /body_composition/history/{user_id} --------------------------------
def test_body_composition_history(client):
    uid = _make_user(client)
    client.post("/api/v1/body_composition", json={"user_id": uid, "date": "2026-01-01", "weight_kg": 82})
    client.post("/api/v1/body_composition", json={"user_id": uid, "date": "2026-01-15", "weight_kg": 80})
    hist = client.get(f"/api/v1/body_composition/history/{uid}").json()
    assert len(hist) == 2
    # oldest first
    assert hist[0]["date"] == "2026-01-01" and hist[0]["weight_kg"] == 82
    assert hist[1]["weight_kg"] == 80


def test_body_composition_history_unknown_user_404(client):
    assert client.get("/api/v1/body_composition/history/999999").status_code == 404
