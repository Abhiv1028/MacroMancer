"""Macro target calculation using the Mifflin-St Jeor equation.

Pure functions (no DB access) so they are trivially unit-testable, plus a thin
helper that upserts a :class:`DailyMacroTarget` row for a user/date.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as date_cls
from typing import Optional

from sqlalchemy.orm import Session

from backend.models import DailyMacroTarget, User

# Activity multipliers applied to BMR to obtain TDEE.
ACTIVITY_FACTORS = {
    "sedentary": 1.2,
    "light": 1.375,
    "moderate": 1.55,
    "very": 1.725,
    "extra": 1.9,
}

# Calorie adjustment applied to TDEE based on the user's goal.
GOAL_FACTORS = {
    "cut": 0.85,
    "maintain": 1.0,
    "bulk": 1.15,
}

# Macro rules.
PROTEIN_G_PER_KG = 2.2
PROTEIN_CAP_G = 250.0
FAT_G_PER_KG = 0.8
FAT_MIN_G = 40.0

CALORIES_PER_G_PROTEIN = 4.0
CALORIES_PER_G_CARB = 4.0
CALORIES_PER_G_FAT = 9.0

# Output safety bounds (kept in sync with the schema Field constraints) so a bad
# or extreme input can never produce a nonsensical/overflowing stored value.
CALORIE_BOUNDS = (0.0, 10000.0)
PROTEIN_BOUNDS = (0.0, 1000.0)
CARB_BOUNDS = (0.0, 1000.0)
FAT_BOUNDS = (0.0, 500.0)
TDEE_BOUNDS = (500.0, 10000.0)


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Constrain ``value`` to the inclusive range [min_val, max_val].

    NaN/None-safe: non-finite inputs collapse to ``min_val``.
    """
    try:
        value = float(value)
    except (TypeError, ValueError):
        return min_val
    if value != value:  # NaN
        return min_val
    return max(min_val, min(value, max_val))


@dataclass
class MacroTargets:
    """Computed daily macro targets."""

    calories: float
    protein_g: float
    carbs_g: float
    fat_g: float


def calculate_bmr(
    weight_kg: float, height_cm: float, age: int, sex: str = "female"
) -> float:
    """Mifflin-St Jeor Basal Metabolic Rate.

    BMR = 10*weight_kg + 6.25*height_cm - 5*age + s
    where s = +5 for male, -161 for female (default).
    """
    s = 5 if str(sex).lower() == "male" else -161
    return 10.0 * weight_kg + 6.25 * height_cm - 5.0 * age + s


def calculate_tdee(bmr: float, activity_level: str) -> float:
    """Total Daily Energy Expenditure = BMR * activity factor (clamped)."""
    factor = ACTIVITY_FACTORS.get(str(activity_level).lower(), 1.2)
    return clamp(bmr * factor, *TDEE_BOUNDS)


def split_macros(goal_calories: float, weight_kg: float) -> MacroTargets:
    """Split a (goal-adjusted) calorie total into protein/fat/carb targets.

    - Protein: 2.2 g/kg, capped at 250 g.
    - Fat: 0.8 g/kg, floored at 40 g.
    - Carbs: remaining calories / 4 (never negative).
    """
    protein_g = min(PROTEIN_G_PER_KG * weight_kg, PROTEIN_CAP_G)
    fat_g = max(FAT_G_PER_KG * weight_kg, FAT_MIN_G)
    remaining_calories = (
        goal_calories - protein_g * CALORIES_PER_G_PROTEIN - fat_g * CALORIES_PER_G_FAT
    )
    carbs_g = max(remaining_calories / CALORIES_PER_G_CARB, 0.0)
    # Clamp every output so extreme inputs can't overflow downstream storage.
    return MacroTargets(
        calories=round(clamp(goal_calories, *CALORIE_BOUNDS), 1),
        protein_g=round(clamp(protein_g, *PROTEIN_BOUNDS), 1),
        carbs_g=round(clamp(carbs_g, *CARB_BOUNDS), 1),
        fat_g=round(clamp(fat_g, *FAT_BOUNDS), 1),
    )


def calculate_macro_targets(
    weight_kg: float,
    height_cm: float,
    age: int,
    activity_level: str,
    goal: str,
    sex: str = "female",
) -> MacroTargets:
    """Compute calorie and macronutrient targets from a profile (Mifflin path)."""
    bmr = calculate_bmr(weight_kg, height_cm, age, sex)
    tdee = calculate_tdee(bmr, activity_level)
    calories = tdee * GOAL_FACTORS.get(str(goal).lower(), 1.0)
    return split_macros(calories, weight_kg)


def calculate_macro_targets_from_tdee(
    tdee: float, weight_kg: float, goal: str
) -> MacroTargets:
    """Compute targets from an already-computed TDEE (e.g. adaptive TDEE).

    The TDEE already accounts for activity, so only the goal factor is applied.
    """
    calories = tdee * GOAL_FACTORS.get(str(goal).lower(), 1.0)
    return split_macros(calories, weight_kg)


def store_targets(
    db: Session, user: User, target_date: date_cls, targets: MacroTargets
) -> DailyMacroTarget:
    """Upsert a user's macro targets for a date with the given values."""
    row = (
        db.query(DailyMacroTarget)
        .filter(
            DailyMacroTarget.user_id == user.id,
            DailyMacroTarget.date == target_date,
        )
        .one_or_none()
    )
    if row is None:
        row = DailyMacroTarget(user_id=user.id, date=target_date)
        db.add(row)
    row.calories = targets.calories
    row.protein_g = targets.protein_g
    row.carbs_g = targets.carbs_g
    row.fat_g = targets.fat_g
    db.commit()
    db.refresh(row)
    return row


def recalculate_targets(
    db: Session,
    user: User,
    target_date: Optional[date_cls] = None,
    tdee: Optional[float] = None,
) -> DailyMacroTarget:
    """Recompute and persist a user's targets, overwriting any existing row.

    If ``tdee`` is provided, targets are derived from it (adaptive path);
    otherwise the static Mifflin-St Jeor calculation is used.
    """
    if target_date is None:
        target_date = date_cls.today()
    if tdee is not None:
        targets = calculate_macro_targets_from_tdee(tdee, user.weight_kg, user.goal)
    else:
        targets = calculate_macro_targets(
            weight_kg=user.weight_kg,
            height_cm=user.height_cm,
            age=user.age,
            activity_level=user.activity_level,
            goal=user.goal,
            sex=user.sex,
        )
    return store_targets(db, user, target_date, targets)


def get_or_create_targets(
    db: Session, user: User, target_date: Optional[date_cls] = None
) -> DailyMacroTarget:
    """Return the stored targets for ``user`` on ``target_date``.

    If none exist, compute and persist them. Defaults to today's date.
    """
    if target_date is None:
        target_date = date_cls.today()

    existing = (
        db.query(DailyMacroTarget)
        .filter(
            DailyMacroTarget.user_id == user.id,
            DailyMacroTarget.date == target_date,
        )
        .one_or_none()
    )
    if existing is not None:
        return existing

    targets = calculate_macro_targets(
        weight_kg=user.weight_kg,
        height_cm=user.height_cm,
        age=user.age,
        activity_level=user.activity_level,
        goal=user.goal,
        sex=user.sex,
    )
    row = DailyMacroTarget(
        user_id=user.id,
        date=target_date,
        calories=targets.calories,
        protein_g=targets.protein_g,
        carbs_g=targets.carbs_g,
        fat_g=targets.fat_g,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row
