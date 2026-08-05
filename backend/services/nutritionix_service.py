"""Restaurant menu nutrition via the free Nutritionix API tier.

Register (no credit card) at https://www.nutritionix.com/ and set
``NUTRITIONIX_APP_ID`` / ``NUTRITIONIX_API_KEY``. When unconfigured,
:func:`fetch_restaurant_menu` raises :class:`NutritionixUnavailableError`, which
the route maps to HTTP 503 with setup instructions.
"""

from __future__ import annotations

from typing import Dict, List

import httpx

from backend.config import settings

SETUP_HELP = (
    "Nutritionix is not configured. Register (free, no card) at "
    "https://www.nutritionix.com/business/api and set NUTRITIONIX_APP_ID and "
    "NUTRITIONIX_API_KEY environment variables, then retry."
)


class NutritionixUnavailableError(RuntimeError):
    """Raised when Nutritionix credentials are missing."""


def is_configured() -> bool:
    """Whether Nutritionix credentials are present."""
    return bool(settings.NUTRITIONIX_APP_ID and settings.NUTRITIONIX_API_KEY)


def _parse_item(raw: Dict) -> Dict:
    """Normalize one Nutritionix branded item into our menu-item shape."""
    return {
        "name": raw.get("food_name") or raw.get("item_name") or "Unknown item",
        "calories": float(raw.get("nf_calories") or 0.0),
        "protein_g": float(raw.get("nf_protein") or 0.0),
        "carbs_g": float(raw.get("nf_total_carbohydrate") or 0.0),
        "fat_g": float(raw.get("nf_total_fat") or 0.0),
        "serving_size_g": float(raw.get("nf_serving_weight_grams") or 0.0) or 100.0,
    }


async def fetch_restaurant_menu(restaurant_name: str) -> List[Dict]:
    """Search Nutritionix for branded menu items of a restaurant.

    Returns a list of ``{name, calories, protein_g, carbs_g, fat_g,
    serving_size_g}``. Network errors yield an empty list (so a search over many
    restaurants degrades gracefully); a missing key raises.
    """
    if not is_configured():
        raise NutritionixUnavailableError(SETUP_HELP)

    headers = {
        "x-app-id": settings.NUTRITIONIX_APP_ID,
        "x-app-key": settings.NUTRITIONIX_API_KEY,
        "Content-Type": "application/json",
    }
    params = {"query": restaurant_name, "branded": "true", "common": "false"}
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                settings.NUTRITIONIX_BASE_URL, headers=headers, params=params
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPError:
        return []

    branded = data.get("branded", data.get("hits", []))
    target = restaurant_name.lower()
    items: List[Dict] = []
    for raw in branded:
        brand = (raw.get("brand_name") or "").lower()
        # Keep items whose brand matches the restaurant when brand info exists.
        if brand and target not in brand and brand not in target:
            continue
        items.append(_parse_item(raw))
    return items
