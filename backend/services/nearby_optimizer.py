"""Score nearby-restaurant menu items against a user's remaining macros.

Combines the XGBoost model score (via transient, non-persisted Food objects) with
a macro-fit measure to the target macros. Degrades to macro-fit only if the model
is unavailable, so it never fails the request.
"""

from __future__ import annotations

from typing import Dict, List

from sqlalchemy.orm import Session

from backend.models import Food, Nutrient, User
from backend.services import optimizer


def _item_to_food(item: Dict) -> Food:
    """Build a transient (unsaved) Food from a menu item's per-serving macros."""
    serving = float(item.get("serving_size_g") or 100.0) or 100.0
    factor = 100.0 / serving
    food = Food(name=item.get("name", "item"), default_grams=serving, is_custom=True)
    food.nutrients = [
        Nutrient(nutrient_name="Protein", amount_per_100g=item.get("protein_g", 0.0) * factor, unit="g"),
        Nutrient(nutrient_name="Carbohydrate", amount_per_100g=item.get("carbs_g", 0.0) * factor, unit="g"),
        Nutrient(nutrient_name="Fat", amount_per_100g=item.get("fat_g", 0.0) * factor, unit="g"),
        Nutrient(nutrient_name="Energy", amount_per_100g=item.get("calories", 0.0) * factor, unit="kcal"),
    ]
    return food


def score_nearby_items(
    db: Session,
    items: List[Dict],
    remaining_macros: Dict[str, float],
    user_id: int,
    meal_type: str = "lunch",
) -> List[Dict]:
    """Return ``items`` annotated with ``macro_fit`` and ``score``, best first.

    ``macro_fit`` measures how well an item's macro split matches the target
    (remaining) macros; ``score`` is the XGBoost probability (or macro_fit as a
    fallback). Items are sorted by a blend of the two.
    """
    if not items:
        return []
    user = db.get(User, user_id)
    if user is None:
        return []

    # XGBoost score per item (input order); fall back to macro-fit on any error.
    probs: List[float] = []
    try:
        foods = [_item_to_food(it) for it in items]
        # remaining_macros used as "consumed" proxy so the model reacts to the gap.
        scored = optimizer.score_candidates(
            db=db,
            user=user,
            current_macros=remaining_macros,
            meal_type=meal_type,
            candidate_foods=foods,
            sort=False,
        )
        probs = [r["predicted_score"] for r in scored]
    except Exception as exc:  # e.g. XGBoost/OpenMP unavailable
        print(f"[nearby] optimizer unavailable ({exc}); using macro-fit only.")
        probs = []

    results: List[Dict] = []
    for idx, item in enumerate(items):
        item_macros = {
            "protein_g": item.get("protein_g", 0.0),
            "carbs_g": item.get("carbs_g", 0.0),
            "fat_g": item.get("fat_g", 0.0),
        }
        macro_fit = optimizer._macro_fit(item_macros, remaining_macros)
        score = probs[idx] if idx < len(probs) else macro_fit
        enriched = dict(item)
        enriched["macro_fit"] = round(float(macro_fit), 2)
        enriched["score"] = round(float(score), 2)
        enriched["_combined"] = score * (0.5 + 0.5 * macro_fit)
        results.append(enriched)

    results.sort(key=lambda r: r["_combined"], reverse=True)
    for r in results:
        r.pop("_combined", None)
    return results
