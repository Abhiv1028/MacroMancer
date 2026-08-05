"""Helpers for deriving macro/calorie values from a Food's nutrient profile."""

from __future__ import annotations

from typing import Dict, Iterable

from backend.models import Food, Nutrient

# Canonical nutrient labels stored by the USDA loader / custom-food creation.
PROTEIN = "Protein"
CARBS = "Carbohydrate"
FAT = "Fat"
ENERGY = "Energy"

CAL_P, CAL_C, CAL_F = 4.0, 4.0, 9.0


def nutrient_map(nutrients: Iterable[Nutrient]) -> Dict[str, float]:
    """Return a {nutrient_name: amount_per_100g} dict for quick lookup."""
    return {n.nutrient_name: n.amount_per_100g for n in nutrients}


def macros_per_100g(food: Food) -> Dict[str, float]:
    """Extract protein/carbs/fat/calories per 100 g for a food.

    Falls back to computing calories from macros (4/4/9) when an explicit
    Energy value is missing or zero.
    """
    nm = nutrient_map(food.nutrients)
    protein = float(nm.get(PROTEIN, 0.0) or 0.0)
    carbs = float(nm.get(CARBS, 0.0) or 0.0)
    fat = float(nm.get(FAT, 0.0) or 0.0)
    calories = float(nm.get(ENERGY, 0.0) or 0.0)
    if calories <= 0:
        calories = protein * CAL_P + carbs * CAL_C + fat * CAL_F
    return {
        "protein_g": protein,
        "carbs_g": carbs,
        "fat_g": fat,
        "calories": calories,
    }


def macros_for_grams(food: Food, grams: float) -> Dict[str, float]:
    """Scale a food's per-100g macros to an arbitrary gram amount."""
    base = macros_per_100g(food)
    factor = grams / 100.0
    return {
        "protein_g": round(base["protein_g"] * factor, 2),
        "carbs_g": round(base["carbs_g"] * factor, 2),
        "fat_g": round(base["fat_g"] * factor, 2),
        "calories": round(base["calories"] * factor, 2),
    }
