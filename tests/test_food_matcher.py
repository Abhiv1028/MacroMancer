"""Tests for fuzzy dish -> food matching."""

from __future__ import annotations

from backend.services import cache
from backend.services.food_matcher import (
    MATCH_THRESHOLD,
    match_food_name,
    match_food_to_usda,
)

CHOICES = {
    1: "Grilled Chicken Breast",
    2: "White Rice, cooked",
    3: "Broccoli, steamed",
    4: "Atlantic Salmon Fillet",
    5: "Greek Yogurt, nonfat",
}


def setup_function(_):
    # Isolate cache between tests so results don't leak.
    cache.clear_food_match()


def test_matches_close_name():
    food_id, score = match_food_name("chicken breast", CHOICES)
    assert food_id == 1
    assert score >= MATCH_THRESHOLD


def test_matches_salmon():
    food_id, score = match_food_name("salmon", CHOICES)
    assert food_id == 4


def test_no_match_below_threshold():
    food_id, score = match_food_name("xylophone tacos", CHOICES)
    assert food_id is None
    assert score == 0.0


def test_empty_inputs():
    assert match_food_name("", CHOICES) == (None, 0.0)
    assert match_food_name("chicken", {}) == (None, 0.0)


def test_cache_is_used():
    # Prime the cache with a confident match, then ensure a changed choices set
    # is ignored (the cached result is returned regardless of new choices).
    food_id, _ = match_food_name("chicken breast", CHOICES)
    assert food_id == 1
    assert cache.get_food_match("chicken breast") is not None
    assert match_food_name("chicken breast", {9: "Totally Different"})[0] == 1


def test_match_food_to_usda_with_db(client, db_session):
    from backend.models import Food, Nutrient

    cache.clear_food_match()
    food = Food(name="Grilled Chicken Breast", default_grams=150, is_custom=True)
    food.nutrients.append(Nutrient(nutrient_name="Protein", amount_per_100g=31, unit="g"))
    db_session.add(food)
    db_session.commit()

    matched, score = match_food_to_usda("chicken breast", db_session)
    assert matched is not None
    assert matched.id == food.id
    assert score >= MATCH_THRESHOLD
