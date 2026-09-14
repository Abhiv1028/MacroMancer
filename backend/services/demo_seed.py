"""Seed a small demo dataset so the live/Docker demo is never blank.

Only runs when ``SEED_DEMO`` is set and the relevant table is empty, so it's a
no-op for normal use and for the test suite (which leaves ``SEED_DEMO`` unset).
"""

from __future__ import annotations

import json
from typing import List

from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.config import settings
from backend.models import Food, Nutrient, User
from backend.services.macro_calculator import get_or_create_targets

_CAL = {"protein": 4.0, "carbs": 4.0, "fat": 9.0}


def _load_seed_foods() -> List[dict]:
    path = settings.DATA_DIR / "seed_foods.json"
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh).get("foods", [])
    except (OSError, json.JSONDecodeError) as exc:  # pragma: no cover
        print(f"[seed] Could not read {path}: {exc}")
        return []


def seed_foods(db: Session) -> int:
    """Insert the curated demo foods if the foods table is empty. Returns count."""
    if db.query(func.count(Food.id)).scalar():
        return 0
    created = 0
    for item in _load_seed_foods():
        p = float(item.get("protein_per_100g", 0.0))
        c = float(item.get("carbs_per_100g", 0.0))
        f = float(item.get("fat_per_100g", 0.0))
        calories = p * _CAL["protein"] + c * _CAL["carbs"] + f * _CAL["fat"]
        food = Food(
            name=item["name"],
            category=item.get("category"),
            default_grams=100.0,
            is_custom=True,
        )
        food.nutrients.extend(
            [
                Nutrient(nutrient_name="Protein", amount_per_100g=p, unit="g"),
                Nutrient(nutrient_name="Carbohydrate", amount_per_100g=c, unit="g"),
                Nutrient(nutrient_name="Fat", amount_per_100g=f, unit="g"),
                Nutrient(nutrient_name="Energy", amount_per_100g=round(calories, 1), unit="kcal"),
            ]
        )
        db.add(food)
        created += 1
    db.commit()
    return created


def seed_demo_user(db: Session) -> bool:
    """Create a demo user (id 1) + today's targets if there are no users."""
    if db.query(func.count(User.id)).scalar():
        return False
    user = User(
        age=30, weight_kg=80.0, height_cm=180.0,
        activity_level="moderate", goal="cut", sex="male",
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    get_or_create_targets(db, user)
    return True


def seed_demo(db: Session) -> None:
    """Idempotently seed demo foods + a demo user (only when SEED_DEMO is on)."""
    if not settings.SEED_DEMO:
        return
    n_foods = seed_foods(db)
    made_user = seed_demo_user(db)
    if n_foods or made_user:
        print(f"[seed] Demo seed: {n_foods} foods, demo_user={made_user}.")
