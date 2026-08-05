"""Tests for Phase 4: categorization, aggregation, and grocery endpoints."""

from __future__ import annotations

import pytest

from backend.models import Conversation, ConversationSession
from backend.services.grocery_categorizer import categorize_food
from backend.services.grocery_generator import generate_grocery_items


# --- categorization ----------------------------------------------------------
@pytest.mark.parametrize(
    "name,expected",
    [
        ("Chicken Breast", "meat"),
        ("Ground Beef", "meat"),
        ("Pork Chop", "meat"),
        ("Salmon Fillet", "seafood"),
        ("Shrimp", "seafood"),
        ("Canned Tuna", "seafood"),
        ("Broccoli", "produce"),
        ("Fresh Spinach", "produce"),
        ("Banana", "produce"),
        ("Sweet Potato", "produce"),
        ("Cheddar Cheese", "dairy"),
        ("Greek Yogurt", "dairy"),
        ("Whole Milk", "dairy"),
        ("Large Egg", "dairy"),
        ("White Rice", "pantry"),
        ("Whole Wheat Pasta", "pantry"),
        ("Olive Oil", "pantry"),
        ("Raw Almonds", "pantry"),
        ("Cinnamon", "spices"),
        ("Black Pepper", "spices"),
        ("Garlic Powder", "spices"),  # beats generic "garlic" -> produce
        ("Ice Cream", "frozen"),
        ("Mystery Rock", "other"),
    ],
)
def test_categorize_food(name, expected):
    assert categorize_food(name) == expected


def test_categorize_empty_is_other():
    assert categorize_food("") == "other"
    assert categorize_food("   ") == "other"


# --- generator ---------------------------------------------------------------
def test_generate_empty_input():
    assert generate_grocery_items([]) == []
    assert generate_grocery_items(None) == []  # type: ignore[arg-type]


def test_generate_aggregates_duplicates():
    items = generate_grocery_items(
        [
            {"name": "Chicken", "grams": 100},
            {"name": "chicken", "grams": 150},  # different case -> same item
            {"name": "Rice", "grams": 200},
        ]
    )
    assert len(items) == 2
    chicken = next(i for i in items if i.food_name.lower() == "chicken")
    assert chicken.quantity == 250.0
    assert chicken.unit == "g"
    assert chicken.category == "meat"


def test_generate_scales_by_meal_prep_days():
    items = generate_grocery_items([{"name": "Rice", "grams": 100}], meal_prep_days=3)
    assert items[0].quantity == 300.0


def test_generate_converts_to_kg_over_1000g():
    items = generate_grocery_items([{"name": "Chicken", "grams": 1500}])
    assert items[0].unit == "kg"
    assert items[0].quantity == 1.5


def test_generate_skips_nameless_and_bad_rows():
    items = generate_grocery_items(
        [{"grams": 100}, {"name": "", "grams": 50}, "not a dict", {"name": "Egg", "grams": 60}]
    )
    assert len(items) == 1
    assert items[0].food_name == "Egg"


# --- endpoints ---------------------------------------------------------------
def _make_user(client) -> int:
    return client.post(
        "/users",
        json={
            "age": 30, "weight_kg": 75, "height_cm": 180,
            "activity_level": "moderate", "goal": "cut", "sex": "male",
        },
    ).json()["user"]["id"]


def test_create_list_from_meal_plan(client):
    user_id = _make_user(client)
    resp = client.post(
        "/api/v1/grocery_lists",
        json={
            "user_id": user_id,
            "name": "Weekly Prep",
            "meal_prep_days": 2,
            "meal_plan": [
                {"name": "Chicken Breast", "grams": 300},
                {"name": "Broccoli", "grams": 200},
                {"name": "Chicken Breast", "grams": 200},  # dup -> summed
            ],
        },
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert isinstance(body["grocery_list_id"], int)
    assert body["name"] == "Weekly Prep"
    names = {i["food_name"].lower() for i in body["items"]}
    assert "chicken breast" in names and "broccoli" in names
    # (300 + 200) * 2 days = 1000g -> stays grams (not > 1000)
    chicken = next(i for i in body["items"] if i["food_name"] == "Chicken Breast")
    assert chicken["quantity"] == 1000.0 and chicken["unit"] == "g"


def test_get_toggle_and_delete_list(client):
    user_id = _make_user(client)
    created = client.post(
        "/api/v1/grocery_lists",
        json={"user_id": user_id, "meal_plan": [{"name": "Rice", "grams": 200}]},
    ).json()
    list_id = created["grocery_list_id"]
    item_id = created["items"][0]["id"]

    # GET
    got = client.get(f"/api/v1/grocery_lists/{list_id}")
    assert got.status_code == 200
    assert got.json()["id"] == list_id
    assert got.json()["items"][0]["checked"] is False

    # PUT toggle
    toggled = client.put(f"/api/v1/grocery_items/{item_id}")
    assert toggled.status_code == 200
    assert toggled.json()["checked"] is True
    # toggling again flips back
    assert client.put(f"/api/v1/grocery_items/{item_id}").json()["checked"] is False

    # DELETE
    deleted = client.delete(f"/api/v1/grocery_lists/{list_id}")
    assert deleted.status_code == 200
    assert deleted.json()["status"] == "deleted"
    assert client.get(f"/api/v1/grocery_lists/{list_id}").status_code == 404


def test_create_list_from_session(client, db_session):
    user_id = _make_user(client)
    session = ConversationSession(user_id=user_id, title="prep chat")
    db_session.add(session)
    db_session.flush()
    db_session.add(
        Conversation(
            session_id=session.id,
            role="assistant",
            content="here you go",
            meal_plan={
                "meal_name": "Prep",
                "foods": [
                    {"food_id": 1, "name": "Chicken", "grams": 500},
                    {"food_id": 2, "name": "Rice", "grams": 300},
                ],
            },
        )
    )
    db_session.commit()

    resp = client.post(
        "/api/v1/grocery_lists",
        json={"user_id": user_id, "session_id": session.id},
    )
    assert resp.status_code == 201, resp.text
    names = {i["food_name"].lower() for i in resp.json()["items"]}
    assert names == {"chicken", "rice"}


def test_create_list_requires_source(client):
    user_id = _make_user(client)
    resp = client.post("/api/v1/grocery_lists", json={"user_id": user_id})
    assert resp.status_code == 422


def test_create_list_unknown_user_404(client):
    resp = client.post(
        "/api/v1/grocery_lists",
        json={"user_id": 999999, "meal_plan": [{"name": "Rice", "grams": 100}]},
    )
    assert resp.status_code == 404


def test_get_unknown_list_404(client):
    assert client.get("/api/v1/grocery_lists/888888").status_code == 404


def test_chat_message_triggers_grocery_list(client):
    user_id = _make_user(client)
    client.post(
        "/foods",
        json={"name": "Chat Chicken", "protein_per_100g": 31, "carbs_per_100g": 0, "fat_per_100g": 3.6},
    )
    resp = client.post(
        "/api/v1/chat",
        json={"user_id": user_id, "message": "make me a grocery list for dinner"},
    )
    assert resp.status_code == 200, resp.text
    gid = resp.json()["grocery_list_id"]
    assert isinstance(gid, int)
    # The generated list is retrievable and non-empty.
    detail = client.get(f"/api/v1/grocery_lists/{gid}")
    assert detail.status_code == 200
    assert len(detail.json()["items"]) >= 1
