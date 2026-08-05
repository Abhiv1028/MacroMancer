"""Aggregate meal-plan foods into a categorized, de-duplicated grocery list."""

from __future__ import annotations

from typing import Dict, List

from backend.models import GroceryItem
from backend.services.grocery_categorizer import categorize_food

# Keys we accept for a food's gram amount, in priority order.
_GRAM_KEYS = ("grams", "quantity", "grams_consumed", "suggested_grams")
# Above this many grams we present the quantity in kilograms.
_KG_THRESHOLD = 1000.0


def _extract_grams(food: dict) -> float:
    """Best-effort extraction of a food's gram amount from a plan dict."""
    for key in _GRAM_KEYS:
        value = food.get(key)
        if value is not None:
            try:
                return max(float(value), 0.0)
            except (TypeError, ValueError):
                continue
    return 0.0


def generate_grocery_items(
    foods: List[dict], meal_prep_days: int = 1
) -> List[GroceryItem]:
    """Turn a list of meal-plan foods into aggregated :class:`GroceryItem` rows.

    - Duplicates (by case-insensitive name) are summed.
    - Totals are scaled by ``meal_prep_days``.
    - Quantities >1000 g are expressed in kg (grams remain the internal unit of
      aggregation); everything is rounded to 2 decimals.
    - Empty/invalid input yields an empty list rather than raising.

    The returned items are unsaved (no ``grocery_list_id``); the caller attaches
    them to a :class:`GroceryList`.
    """
    if not foods:
        return []
    days = max(int(meal_prep_days), 1)

    # Aggregate grams by normalized name, remembering a display name + category.
    aggregated: Dict[str, dict] = {}
    for food in foods:
        if not isinstance(food, dict):
            continue
        raw_name = str(food.get("name") or food.get("food_name") or "").strip()
        if not raw_name:
            continue
        key = raw_name.lower()
        grams = _extract_grams(food) * days
        if key not in aggregated:
            aggregated[key] = {
                "name": raw_name,
                "grams": 0.0,
                "category": categorize_food(raw_name),
            }
        aggregated[key]["grams"] += grams

    items: List[GroceryItem] = []
    for entry in aggregated.values():
        grams_total = entry["grams"]
        if grams_total > _KG_THRESHOLD:
            quantity = round(grams_total / 1000.0, 2)
            unit = "kg"
        else:
            quantity = round(grams_total, 2)
            unit = "g"
        items.append(
            GroceryItem(
                food_name=entry["name"],
                category=entry["category"],
                quantity=quantity,
                unit=unit,
                checked=False,
            )
        )
    # Stable, shopper-friendly ordering: by category then name.
    items.sort(key=lambda it: (it.category, it.food_name.lower()))
    return items
