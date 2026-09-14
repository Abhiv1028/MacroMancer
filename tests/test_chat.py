"""Tests for the Phase-2 chat / meal-planning endpoint.

These pass whether or not a local Ollama server is running: with Ollama up the
response carries a parsed ``meal_plan``; without it the endpoint degrades to a
plain-text suggestion (``meal_plan is None``).
"""

from __future__ import annotations

import pytest

from backend.services.meal_type_detector import detect_meal_type


# --- meal_type_detector (pure unit tests) ------------------------------------
@pytest.mark.parametrize(
    "message,hour,expected",
    [
        ("What should I eat for breakfast?", 20, "breakfast"),  # keyword wins over clock
        ("Any lunch ideas?", 3, "lunch"),
        ("dinner please", 9, "dinner"),
        ("a quick snack", 13, "snack"),
        ("I'm hungry", 8, "breakfast"),   # 6-11
        ("I'm hungry", 12, "lunch"),      # 11-15
        ("I'm hungry", 18, "dinner"),     # 15-20
        ("I'm hungry", 23, "snack"),      # else
        ("I'm hungry", 5, "snack"),       # before 6
    ],
)
def test_detect_meal_type(message, hour, expected):
    assert detect_meal_type(message, hour) == expected


# --- helpers -----------------------------------------------------------------
def _seed_user_and_foods(client) -> int:
    user_id = client.post(
        "/users",
        json={
            "age": 30,
            "weight_kg": 75,
            "height_cm": 180,
            "activity_level": "moderate",
            "goal": "cut",
            "sex": "male",
        },
    ).json()["user"]["id"]

    for spec in [
        {"name": "Chat Chicken", "protein_per_100g": 31, "carbs_per_100g": 0, "fat_per_100g": 3.6},
        {"name": "Chat Rice", "protein_per_100g": 2.7, "carbs_per_100g": 28, "fat_per_100g": 0.3},
        {"name": "Chat Broccoli", "protein_per_100g": 2.8, "carbs_per_100g": 7, "fat_per_100g": 0.4},
    ]:
        food_id = client.post("/foods", json=spec).json()["id"]
        # Log one meal so today's eaten macros are non-zero.
        if spec["name"] == "Chat Chicken":
            client.post(
                "/meals",
                json={
                    "user_id": user_id,
                    "food_id": food_id,
                    "grams": 150,
                    "meal_type": "lunch",
                },
            )
    return user_id


# --- endpoint tests ----------------------------------------------------------
def test_chat_returns_plan_or_fallback(client):
    user_id = _seed_user_and_foods(client)

    resp = client.post(
        "/api/v1/chat",
        json={"user_id": user_id, "message": "What should I have for dinner?"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert isinstance(body["session_id"], int)
    assert isinstance(body["message"], str) and body["message"]

    if body["meal_plan"] is not None:
        # Ollama was reachable: validate best-effort structure.
        plan = body["meal_plan"]
        assert isinstance(plan, dict)
        if "foods" in plan:
            assert isinstance(plan["foods"], list)
        if "total_macros" in plan:
            assert isinstance(plan["total_macros"], dict)
    else:
        # Fallback mode: message explains Ollama is unavailable.
        assert "unavailable" in body["message"].lower()


def test_chat_session_continuity(client):
    user_id = _seed_user_and_foods(client)

    first = client.post(
        "/api/v1/chat",
        json={"user_id": user_id, "message": "breakfast ideas?"},
    ).json()
    session_id = first["session_id"]

    second = client.post(
        "/api/v1/chat",
        json={"user_id": user_id, "session_id": session_id, "message": "and lunch?"},
    )
    assert second.status_code == 200
    assert second.json()["session_id"] == session_id


def test_chat_unknown_user_404(client):
    resp = client.post(
        "/api/v1/chat",
        json={"user_id": 999999, "message": "hi"},
    )
    assert resp.status_code == 404


def test_chat_unknown_session_404(client):
    user_id = _seed_user_and_foods(client)
    resp = client.post(
        "/api/v1/chat",
        json={"user_id": user_id, "session_id": 888888, "message": "hi"},
    )
    assert resp.status_code == 404


# --- hosted LLM provider (Groq / OpenAI-compatible) --------------------------
def test_generate_meal_plan_uses_hosted_when_key_present(monkeypatch):
    """With a hosted key set, generate_meal_plan hits the chat-completions API
    (not Ollama) and parses the JSON object out of choices[0].message.content."""
    import asyncio
    import json as _json

    from backend.services import ollama_client as oc

    monkeypatch.setattr(oc.settings, "LLM_API_KEY", "test-key", raising=False)

    captured = {}

    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            content = _json.dumps(
                {
                    "meal_name": "Hosted Plan",
                    "foods": [{"food_id": 1, "name": "Chicken", "grams": 150}],
                    "total_macros": {"protein_g": 46, "carbs_g": 0, "fat_g": 5, "calories": 240},
                    "explanation": "ok",
                }
            )
            return {"choices": [{"message": {"content": content}}]}

    class _Client:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            pass

        async def post(self, url, json=None, headers=None):
            captured["url"] = url
            captured["headers"] = headers
            captured["model"] = json["model"]
            return _Resp()

    monkeypatch.setattr(oc.httpx, "AsyncClient", _Client)

    plan = asyncio.run(oc.generate_meal_plan("system prompt", "dinner please"))

    assert plan["meal_name"] == "Hosted Plan"
    assert plan["foods"][0]["name"] == "Chicken"
    assert "groq.com" in captured["url"]  # default hosted provider
    assert captured["headers"]["Authorization"] == "Bearer test-key"


def test_generate_meal_plan_hosted_bad_json_raises(monkeypatch):
    """Non-JSON hosted output surfaces as LLMUnavailableError for graceful fallback."""
    import asyncio

    from backend.services import ollama_client as oc

    monkeypatch.setattr(oc.settings, "LLM_API_KEY", "test-key", raising=False)

    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": "not json at all"}}]}

    class _Client:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            pass

        async def post(self, *a, **k):
            return _Resp()

    monkeypatch.setattr(oc.httpx, "AsyncClient", _Client)

    with pytest.raises(oc.LLMUnavailableError):
        asyncio.run(oc.generate_meal_plan("system", "hi"))
