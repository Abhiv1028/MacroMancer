"""Tests for the restaurant-mode endpoints (OCR mocked)."""

from __future__ import annotations

import pytest

from backend.routes import restaurant
from backend.services import cache
from backend.services.ocr_service import OCRUnavailableError

RECEIPT_TEXT = """
The Green Fork
Grilled Chicken Breast    $12.99
Broccoli                   5.00
Zzzxqq Widget              9.99
TOTAL                     27.98
"""


def _make_user(client) -> int:
    return client.post(
        "/users",
        json={"age": 30, "weight_kg": 75, "height_cm": 180,
              "activity_level": "moderate", "goal": "cut", "sex": "male"},
    ).json()["user"]["id"]


def _make_food(client, name, p=20, c=0, f=5, category=None) -> int:
    body = {"name": name, "protein_per_100g": p, "carbs_per_100g": c, "fat_per_100g": f}
    if category:
        body["category"] = category
    return client.post("/foods", json=body).json()["id"]


@pytest.fixture(autouse=True)
def _clear_match_cache():
    cache.clear_food_match()
    yield
    cache.clear_food_match()


def _mock_ocr(monkeypatch, text=RECEIPT_TEXT):
    async def fake(image_bytes):
        return text
    monkeypatch.setattr(restaurant, "extract_text_from_image", fake)


def test_upload_receipt_parses_and_matches(client, monkeypatch):
    user_id = _make_user(client)
    _make_food(client, "Grilled Chicken Breast")
    _make_food(client, "Broccoli, steamed")
    _mock_ocr(monkeypatch)

    resp = client.post(
        "/api/v1/restaurant/upload_receipt",
        data={"user_id": str(user_id)},
        files={"file": ("receipt.png", b"fakebytes", "image/png")},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "Grilled Chicken Breast" in body["receipt_text"]

    items = {i["dish_name"]: i for i in body["parsed_items"]}
    chicken = next(i for k, i in items.items() if "chicken" in k.lower())
    assert chicken["matched_food"] is not None
    assert chicken["confidence"] >= 70

    widget = next(i for k, i in items.items() if "widget" in k.lower())
    assert widget["matched_food"] is None
    assert widget["warning"] is not None


def test_upload_receipt_ocr_unavailable_503(client, monkeypatch):
    user_id = _make_user(client)

    async def boom(image_bytes):
        raise OCRUnavailableError("install tesseract")

    monkeypatch.setattr(restaurant, "extract_text_from_image", boom)
    resp = client.post(
        "/api/v1/restaurant/upload_receipt",
        data={"user_id": str(user_id)},
        files={"file": ("receipt.png", b"x", "image/png")},
    )
    assert resp.status_code == 503


def test_upload_receipt_unknown_user_404(client, monkeypatch):
    _mock_ocr(monkeypatch)
    resp = client.post(
        "/api/v1/restaurant/upload_receipt",
        data={"user_id": "999999"},
        files={"file": ("r.png", b"x", "image/png")},
    )
    assert resp.status_code == 404


def test_log_meal_creates_log_and_meallog(client, db_session):
    from backend.models import MealLog

    user_id = _make_user(client)
    food_id = _make_food(client, "Restaurant Burger", p=15, c=30, f=20)

    resp = client.post(
        "/api/v1/restaurant/log_meal",
        json={
            "user_id": user_id,
            "restaurant_name": "The Green Fork",
            "notes": "cheat day",
            "items": [{"food_id": food_id, "grams": 250}],
        },
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["restaurant_name"] == "The Green Fork"
    assert len(body["items"]) == 1
    assert body["items"][0]["matched_food_id"] == food_id
    assert body["items"][0]["calories_estimate"] > 0

    # A MealLog was mirrored so the day's macros include this meal.
    meals = db_session.query(MealLog).filter(MealLog.user_id == user_id).all()
    assert len(meals) == 1
    assert meals[0].grams_consumed == 250


def test_log_meal_unknown_food_404(client):
    user_id = _make_user(client)
    resp = client.post(
        "/api/v1/restaurant/log_meal",
        json={"user_id": user_id, "items": [{"food_id": 999999, "grams": 100}]},
    )
    assert resp.status_code == 404


def test_list_restaurant_logs(client):
    user_id = _make_user(client)
    food_id = _make_food(client, "Taco")
    client.post(
        "/api/v1/restaurant/log_meal",
        json={"user_id": user_id, "restaurant_name": "Taqueria",
              "items": [{"food_id": food_id, "grams": 150}]},
    )
    resp = client.get(f"/api/v1/restaurant/logs/{user_id}")
    assert resp.status_code == 200
    logs = resp.json()
    assert len(logs) == 1
    assert logs[0]["restaurant_name"] == "Taqueria"


def test_substitute_returns_alternatives(client):
    user_id = _make_user(client)
    original = _make_food(client, "Fried Chicken", p=20, c=10, f=30, category="meat")
    _make_food(client, "Grilled Chicken", p=31, c=0, f=4, category="meat")
    _make_food(client, "Turkey Breast", p=29, c=0, f=2, category="meat")

    resp = client.post(
        "/api/v1/restaurant/substitute",
        json={"user_id": user_id, "food_id": original, "limit": 3},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["original_food_id"] == original
    assert len(body["substitutions"]) >= 1
    # The original should not be recommended as its own substitute.
    assert all(s["id"] != original for s in body["substitutions"])


def test_substitute_unknown_food_404(client):
    user_id = _make_user(client)
    resp = client.post(
        "/api/v1/restaurant/substitute",
        json={"user_id": user_id, "food_id": 999999},
    )
    assert resp.status_code == 404
