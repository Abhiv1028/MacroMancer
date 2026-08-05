"""Tests for the feedback endpoint and feedback-driven score adjustment."""

from __future__ import annotations

from backend.models import Feedback
from backend.services import feedback_adjuster


def _make_user(client) -> int:
    return client.post(
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


def _make_food(client, name="FB Food") -> int:
    return client.post(
        "/foods",
        json={"name": name, "protein_per_100g": 20, "carbs_per_100g": 10, "fat_per_100g": 5},
    ).json()["id"]


def _log_meal(client, user_id, food_id) -> int:
    return client.post(
        "/meals",
        json={"user_id": user_id, "food_id": food_id, "grams": 100, "meal_type": "lunch"},
    ).json()["id"]


def test_create_feedback_for_meal(client):
    user_id = _make_user(client)
    food_id = _make_food(client)
    meal_id = _log_meal(client, user_id, food_id)

    resp = client.post(
        "/api/v1/feedback",
        json={
            "user_id": user_id,
            "meal_log_id": meal_id,
            "enjoyment": 5,
            "satiety": 4,
            "energy": 4,
            "workout_performance": 3,
            "notes": "great",
        },
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["status"] == "ok"
    assert isinstance(body["feedback_id"], int)


def test_create_feedback_without_meal(client):
    user_id = _make_user(client)
    resp = client.post(
        "/api/v1/feedback",
        json={"user_id": user_id, "enjoyment": 3, "satiety": 3, "energy": 3},
    )
    assert resp.status_code == 201


def test_feedback_unknown_user_404(client):
    resp = client.post(
        "/api/v1/feedback",
        json={"user_id": 999999, "enjoyment": 3, "satiety": 3, "energy": 3},
    )
    assert resp.status_code == 404


def test_feedback_bad_meal_log_422(client):
    user_id = _make_user(client)
    resp = client.post(
        "/api/v1/feedback",
        json={"user_id": user_id, "meal_log_id": 777777, "enjoyment": 3, "satiety": 3, "energy": 3},
    )
    assert resp.status_code == 422


def test_feedback_out_of_range_rejected(client):
    user_id = _make_user(client)
    resp = client.post(
        "/api/v1/feedback",
        json={"user_id": user_id, "enjoyment": 9, "satiety": 3, "energy": 3},
    )
    assert resp.status_code == 422


# --- feedback_adjuster unit tests --------------------------------------------
def test_modifier_defaults_neutral_without_feedback(client, db_session):
    user_id = _make_user(client)
    food_id = _make_food(client, "Neutral Food")
    assert feedback_adjuster.adjust_scores_with_feedback(db_session, food_id, user_id) == 1.0
    assert feedback_adjuster.avg_feedback_score(db_session, user_id, food_id) == 3.0


def test_modifier_downweights_disliked_food(client, db_session):
    user_id = _make_user(client)
    food_id = _make_food(client, "Disliked Food")
    # Three low-enjoyment feedbacks on meals of this food.
    for _ in range(3):
        meal_id = _log_meal(client, user_id, food_id)
        db_session.add(
            Feedback(
                user_id=user_id, meal_log_id=meal_id, enjoyment=1, satiety=2, energy=2
            )
        )
    db_session.commit()

    assert feedback_adjuster.adjust_scores_with_feedback(db_session, food_id, user_id) == 0.9
    assert feedback_adjuster.avg_feedback_score(db_session, user_id, food_id) < 3.0


def test_modifier_upweights_loved_food(client, db_session):
    user_id = _make_user(client)
    food_id = _make_food(client, "Loved Food")
    for _ in range(3):
        meal_id = _log_meal(client, user_id, food_id)
        db_session.add(
            Feedback(
                user_id=user_id, meal_log_id=meal_id, enjoyment=5, satiety=5, energy=5
            )
        )
    db_session.commit()

    assert feedback_adjuster.adjust_scores_with_feedback(db_session, food_id, user_id) == 1.1
    assert feedback_adjuster.avg_feedback_score(db_session, user_id, food_id) > 4.0
