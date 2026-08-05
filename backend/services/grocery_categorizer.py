"""Map a food name to a grocery-aisle category.

Keywords are externalized in ``data/food_categories.json`` for easy updates.
Matching is case-insensitive substring; the **longest** matching keyword wins so
specific terms override generic ones (``"sweet potato"`` -> produce beats
``"potato"``; ``"garlic powder"`` -> spices beats ``"garlic"``).
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import List, Tuple

from backend.config import settings

VALID_CATEGORIES = {
    "produce",
    "dairy",
    "meat",
    "seafood",
    "pantry",
    "frozen",
    "spices",
    "other",
}
DEFAULT_CATEGORY = "other"


@lru_cache(maxsize=1)
def _load_keywords() -> List[Tuple[str, str]]:
    """Load (keyword, category) pairs sorted by keyword length descending.

    Sorting longest-first lets the first substring match be the most specific.
    Returns an empty list (not an error) if the file is missing/malformed.
    """
    path = settings.DATA_DIR / "food_categories.json"
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        mapping = data.get("categories", {})
    except (OSError, json.JSONDecodeError) as exc:  # pragma: no cover - config error
        print(f"[grocery] Could not load {path}: {exc}; defaulting all to 'other'.")
        return []

    pairs = [
        (kw.lower(), cat)
        for kw, cat in mapping.items()
        if cat in VALID_CATEGORIES
    ]
    pairs.sort(key=lambda p: len(p[0]), reverse=True)
    return pairs


def categorize_food(food_name: str) -> str:
    """Return the grocery category for ``food_name``.

    Falls back to ``"other"`` for empty input or an unrecognized name.
    """
    if not food_name or not food_name.strip():
        return DEFAULT_CATEGORY
    name = food_name.lower()
    for keyword, category in _load_keywords():
        if keyword in name:
            return category
    return DEFAULT_CATEGORY
