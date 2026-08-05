"""Fuzzy-match parsed dish names to foods in the database (rapidfuzz).

Match results are cached per dish name for a day (the food DB changes rarely);
the cache is cleared when a food is created (see ``routes/foods.py``).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from rapidfuzz import fuzz, process
from sqlalchemy.orm import Session

from backend.models import Food
from backend.services import cache

# Minimum rapidfuzz score (0-100) to accept a match.
MATCH_THRESHOLD = 70.0


def get_food_choices(db: Session) -> Dict[int, str]:
    """Return a ``{food_id: name}`` map of all foods for fuzzy matching."""
    return {fid: name for fid, name in db.query(Food.id, Food.name).all()}


def match_food_name(
    dish_name: str, choices: Dict[int, str]
) -> Tuple[Optional[int], float]:
    """Fuzzy-match ``dish_name`` against ``choices`` (pure; cache-backed).

    Returns ``(food_id, score)`` for the best match at/above
    :data:`MATCH_THRESHOLD`, else ``(None, 0.0)``.
    """
    if not dish_name or not dish_name.strip() or not choices:
        return (None, 0.0)

    cached = cache.get_food_match(dish_name)
    if cached is not None:
        return cached

    # process.extractOne over a dict returns (matched_value, score, key).
    result = process.extractOne(
        dish_name, choices, scorer=fuzz.WRatio, score_cutoff=MATCH_THRESHOLD
    )
    match: Tuple[Optional[int], float]
    if result is None:
        match = (None, 0.0)
    else:
        _, score, food_id = result
        match = (int(food_id), float(score))

    cache.set_food_match(dish_name, match)
    return match


def match_food_to_usda(
    dish_name: str, db: Session
) -> Tuple[Optional[Food], float]:
    """Fuzzy-match a dish name against the Food table.

    Returns ``(Food, score)`` for a confident match, else ``(None, 0.0)``.
    """
    choices = get_food_choices(db)
    food_id, score = match_food_name(dish_name, choices)
    if food_id is None:
        return (None, 0.0)
    return (db.get(Food, food_id), score)


def match_dishes(
    dish_names: List[str], choices: Dict[int, str]
) -> List[Tuple[Optional[int], float]]:
    """Match many dishes against a pre-fetched choices map (thread-safe: no DB).

    Suitable for running under ``asyncio.to_thread`` from an async route.
    """
    return [match_food_name(name, choices) for name in dish_names]
