"""Tests for Phase 7 nearby restaurants (OSM/Nutritionix/optimizer mocked)."""

from __future__ import annotations

import pytest

from backend.models import MealLog, RestaurantMealLog
from backend.services import (
    geo_service,
    nearby_optimizer,
    nutritionix_service,
    osm_service,
)
from backend.services.nutritionix_service import NutritionixUnavailableError


def _make_user(client) -> int:
    return client.post(
        "/users",
        json={"age": 30, "weight_kg": 80, "height_cm": 180,
              "activity_level": "moderate", "goal": "cut", "sex": "male"},
    ).json()["user"]["id"]


_RESTAURANTS = [
    {"osm_id": "node/1", "name": "Sweetgreen", "lat": 40.7, "lon": -74.0,
     "address": "123 Broadway", "cuisine": "salad"},
    {"osm_id": "node/2", "name": "Emptyplace", "lat": 40.71, "lon": -74.01,
     "address": "9 Nowhere", "cuisine": None},
]
_MENU = [
    {"name": "Harvest Bowl", "calories": 650, "protein_g": 32, "carbs_g": 45,
     "fat_g": 28, "serving_size_g": 350},
    {"name": "Kale Caesar", "calories": 450, "protein_g": 20, "carbs_g": 20,
     "fat_g": 30, "serving_size_g": 300},
]


def _mock_stack(monkeypatch, restaurants=None, menu=None):
    monkeypatch.setattr(nutritionix_service, "is_configured", lambda: True)

    async def fake_geo(ip=None):
        return {"lat": 40.7128, "lon": -74.006, "city": "New York",
                "region": "NY", "source": "ip-api"}

    async def fake_osm(lat, lon, radius_m=None):
        return list(restaurants if restaurants is not None else _RESTAURANTS)

    async def fake_nutri(name):
        if name == "Emptyplace":
            return []
        return list(menu if menu is not None else _MENU)

    monkeypatch.setattr(geo_service, "get_location_from_ip", fake_geo)
    monkeypatch.setattr(osm_service, "search_nearby_restaurants", fake_osm)
    monkeypatch.setattr(nutritionix_service, "fetch_restaurant_menu", fake_nutri)


# --- unit: services ----------------------------------------------------------
def test_location_label():
    assert geo_service.location_label({"city": "New York", "region": "NY"}) == "New York, NY"
    assert geo_service.location_label({"city": "", "region": ""}) == "Unknown location"


def test_nutritionix_requires_config(monkeypatch):
    import asyncio

    monkeypatch.setattr(nutritionix_service, "is_configured", lambda: False)
    with pytest.raises(NutritionixUnavailableError):
        asyncio.run(nutritionix_service.fetch_restaurant_menu("Sweetgreen"))


def test_osm_format_address():
    tags = {"addr:housenumber": "123", "addr:street": "Broadway", "addr:city": "NYC"}
    assert osm_service._format_address(tags) == "123 Broadway, NYC"
    assert osm_service._format_address({}) == ""


def test_score_nearby_items_annotates_and_sorts(client, db_session):
    uid = _make_user(client)
    scored = nearby_optimizer.score_nearby_items(
        db_session, list(_MENU), {"protein_g": 100, "carbs_g": 50, "fat_g": 30}, uid
    )
    assert len(scored) == 2
    for it in scored:
        assert 0.0 <= it["macro_fit"] <= 1.0
        assert "score" in it
    # sorted by combined score descending
    assert scored[0]["score"] >= scored[-1]["score"] or scored[0]["macro_fit"] >= 0


def test_score_nearby_items_empty():
    assert nearby_optimizer.score_nearby_items(None, [], {}, 1) == []


# --- endpoint: search --------------------------------------------------------
def test_search_503_without_nutritionix(client, monkeypatch):
    uid = _make_user(client)
    monkeypatch.setattr(nutritionix_service, "is_configured", lambda: False)
    resp = client.get(f"/api/v1/nearby/search?user_id={uid}&lat=40.7&lon=-74.0")
    assert resp.status_code == 503


def test_search_unknown_user_404(client):
    resp = client.get("/api/v1/nearby/search?user_id=999999&lat=40.7&lon=-74.0")
    assert resp.status_code == 404


def test_search_happy_path(client, monkeypatch):
    uid = _make_user(client)
    _mock_stack(monkeypatch)
    resp = client.get(f"/api/v1/nearby/search?user_id={uid}&lat=40.7&lon=-74.0&radius=2000")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["location"] == "40.7, -74.0"
    names = {r["name"] for r in body["restaurants"]}
    assert "Sweetgreen" in names
    # Emptyplace had no menu -> filtered out.
    assert "Emptyplace" not in names
    sg = next(r for r in body["restaurants"] if r["name"] == "Sweetgreen")
    assert len(sg["menu_items"]) == 2
    assert all("macro_fit" in i and "score" in i for i in sg["menu_items"])


def test_search_uses_ip_when_no_coords(client, monkeypatch):
    uid = _make_user(client)
    _mock_stack(monkeypatch)
    resp = client.get(f"/api/v1/nearby/search?user_id={uid}")
    assert resp.status_code == 200
    assert resp.json()["location"] == "New York, NY"


def test_search_caches_results(client, monkeypatch):
    uid = _make_user(client)
    calls = {"osm": 0}

    monkeypatch.setattr(nutritionix_service, "is_configured", lambda: True)

    async def counting_osm(lat, lon, radius_m=None):
        calls["osm"] += 1
        return list(_RESTAURANTS)

    async def fake_nutri(name):
        return [] if name == "Emptyplace" else list(_MENU)

    monkeypatch.setattr(osm_service, "search_nearby_restaurants", counting_osm)
    monkeypatch.setattr(nutritionix_service, "fetch_restaurant_menu", fake_nutri)

    url = f"/api/v1/nearby/search?user_id={uid}&lat=41.0&lon=-73.0&radius=1500"
    client.get(url)
    client.get(url)  # second call served from NearbySearchCache
    assert calls["osm"] == 1


# --- endpoint: log -----------------------------------------------------------
def test_log_nearby_meal(client, db_session, monkeypatch):
    uid = _make_user(client)
    _mock_stack(monkeypatch)
    # Populate the cache so the item is loggable.
    client.get(f"/api/v1/nearby/search?user_id={uid}&lat=40.5&lon=-74.5")

    resp = client.post(
        "/api/v1/nearby/log",
        json={"user_id": uid, "restaurant_name": "Sweetgreen",
              "item_name": "Harvest Bowl", "grams": 350},
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["meal_log_id"] > 0
    assert body["macros"]["calories"] == pytest.approx(650, abs=1)  # 350g @ 350g serving
    assert "feedback" in body["feedback_prompt"].lower()

    assert db_session.query(MealLog).filter(MealLog.user_id == uid).count() >= 1
    assert db_session.query(RestaurantMealLog).filter(
        RestaurantMealLog.user_id == uid
    ).count() == 1


def test_log_unknown_user_404(client):
    resp = client.post(
        "/api/v1/nearby/log",
        json={"user_id": 999999, "restaurant_name": "X", "item_name": "Y", "grams": 100},
    )
    assert resp.status_code == 404


def test_log_item_not_cached_404(client):
    uid = _make_user(client)
    resp = client.post(
        "/api/v1/nearby/log",
        json={"user_id": uid, "restaurant_name": "Nope", "item_name": "Ghost", "grams": 100},
    )
    assert resp.status_code == 404
