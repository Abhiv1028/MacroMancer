"""Suggest healthier food substitutions for a given dish.

Finds alternatives with a similar role (same category / macro profile) that are
lower in calories or fat, or higher in protein, then ranks them with the
optimizer (falling back to a simple heuristic if the model is unavailable).
"""

from __future__ import annotations

from typing import List

from sqlalchemy.orm import Session

from backend.models import Food, User
from backend.services import food_utils, optimizer


def _is_better(candidate: dict, original: dict) -> bool:
    """A candidate is a useful swap if it's leaner or more protein-dense."""
    return (
        candidate["calories"] <= original["calories"]
        or candidate["fat_g"] <= original["fat_g"]
        or candidate["protein_g"] >= original["protein_g"]
    )


def suggest_substitutions(
    db: Session, original_food: Food, user_id: int, limit: int = 3
) -> List[Food]:
    """Return up to ``limit`` healthier alternatives to ``original_food``.

    Candidates are drawn from the same category (falling back to all foods),
    filtered to those with a better macro profile, then ranked by the optimizer.
    Returns an empty list when there are no suitable alternatives.
    """
    user = db.get(User, user_id)
    if user is None:
        return []

    original = food_utils.macros_per_100g(original_food)

    query = db.query(Food).filter(Food.id != original_food.id)
    if original_food.category:
        same_cat = query.filter(Food.category == original_food.category).all()
    else:
        same_cat = []
    pool = same_cat or query.all()  # broaden if the category is empty/unset

    candidates = [
        f for f in pool if _is_better(food_utils.macros_per_100g(f), original)
    ]
    if not candidates:
        candidates = pool
    if not candidates:
        return []

    # Rank with the optimizer; fall back to a lean/high-protein heuristic.
    try:
        scored = optimizer.score_candidates(
            db=db,
            user=user,
            current_macros={"protein_g": 0.0, "carbs_g": 0.0, "fat_g": 0.0},
            meal_type="lunch",
            candidate_foods=candidates,
            top_n=limit,
        )
        ranked_ids = [r["food_id"] for r in scored]
        by_id = {f.id: f for f in candidates}
        return [by_id[i] for i in ranked_ids if i in by_id][:limit]
    except Exception as exc:  # e.g. XGBoost/OpenMP unavailable
        print(f"[substitute] Optimizer unavailable ({exc}); using heuristic.")
        candidates.sort(
            key=lambda f: (
                -food_utils.macros_per_100g(f)["protein_g"],
                food_utils.macros_per_100g(f)["calories"],
            )
        )
        return candidates[:limit]
