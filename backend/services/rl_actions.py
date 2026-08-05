"""Fixed macro-strategy action space for the RL layer.

Each action defines a target macronutrient split (fractions of total calories,
summing to 1.0). Later phases use these to bias recommendations; Part 1 only
defines them and seeds the ``rl_actions`` table.
"""

from __future__ import annotations

from datetime import date as date_cls
from typing import Dict, List, Optional

from sqlalchemy.orm import Session

from backend.models import RLAction, User

# name -> (description, {protein, carbs, fat} as fractions of calories, summing 1.0)
ACTIONS: List[Dict] = [
    {
        "name": "high_protein",
        "description": "Protein >= 35% of calories; supports muscle retention on a cut.",
        "split": {"protein": 0.40, "carbs": 0.35, "fat": 0.25},
    },
    {
        "name": "balanced",
        "description": "Even split: 25% protein, 45% carbs, 30% fat.",
        "split": {"protein": 0.25, "carbs": 0.45, "fat": 0.30},
    },
    {
        "name": "low_carb",
        "description": "Carbs < 30% of calories; higher protein and fat.",
        "split": {"protein": 0.35, "carbs": 0.25, "fat": 0.40},
    },
    {
        "name": "high_fat",
        "description": "Fat >= 35% of calories; keto-leaning.",
        "split": {"protein": 0.25, "carbs": 0.35, "fat": 0.40},
    },
]

# Canonical order (index == action index used by the bandit later).
ACTION_NAMES: List[str] = [a["name"] for a in ACTIONS]
N_ACTIONS: int = len(ACTIONS)

# Calories per gram of each macro (for split -> grams conversion).
_CAL_PER_G = {"protein": 4.0, "carbs": 4.0, "fat": 9.0}


def get_action_split(name: str) -> Dict[str, float]:
    """Return the {protein, carbs, fat} calorie-fraction split for an action."""
    for action in ACTIONS:
        if action["name"] == name:
            return dict(action["split"])
    raise KeyError(f"Unknown RL action: {name!r}")


def split_to_grams(name: str, calories: float) -> Dict[str, float]:
    """Convert an action's calorie-fraction split into gram targets."""
    split = get_action_split(name)
    return {
        macro: round((calories * frac) / _CAL_PER_G[macro], 1)
        for macro, frac in split.items()
    }


def seed_actions(db: Session) -> int:
    """Idempotently insert the fixed action set. Returns rows created."""
    existing = {name for (name,) in db.query(RLAction.name).all()}
    created = 0
    for action in ACTIONS:
        if action["name"] in existing:
            continue
        db.add(RLAction(name=action["name"], description=action["description"]))
        created += 1
    if created:
        db.commit()
    return created


def get_action_by_name(db: Session, name: str) -> Optional[RLAction]:
    """Fetch an :class:`RLAction` row by name (None if absent)."""
    return db.query(RLAction).filter(RLAction.name == name).one_or_none()


# Minimum remaining calories before we fall back to a single-meal target.
_MIN_REMAINING_CAL = 200.0


def apply_action(
    db: Session, action_name: str, user_id: int, day: date_cls
) -> Dict[str, float]:
    """Map an RL action to concrete macro targets for a user's next meal.

    Splits the user's **remaining** calories for ``day`` (daily target minus
    what's been eaten) according to the action's macro strategy. If little/no
    budget remains, falls back to one meal's worth (a third of the daily target)
    so a suggestion is still produced.

    Returns ``{protein_g, carbs_g, fat_g, calories}``.
    """
    # Imported here to avoid a module-level import cycle at package load time.
    from backend.services.macro_calculator import get_or_create_targets
    from backend.services.rl_reward import _sum_day_macros

    user = db.get(User, user_id)
    if user is None:
        raise ValueError(f"User {user_id} not found")

    targets = get_or_create_targets(db, user, day)
    eaten = _sum_day_macros(db, user_id, day)
    eaten_cal = (
        eaten["protein_g"] * 4.0 + eaten["carbs_g"] * 4.0 + eaten["fat_g"] * 9.0
    )
    remaining_cal = targets.calories - eaten_cal
    if remaining_cal < _MIN_REMAINING_CAL:
        remaining_cal = targets.calories / 3.0

    grams = split_to_grams(action_name, remaining_cal)
    return {
        "protein_g": grams["protein"],
        "carbs_g": grams["carbs"],
        "fat_g": grams["fat"],
        "calories": round(remaining_cal, 1),
    }
