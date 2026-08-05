"""Feedback-driven adjustments to recommendation scores.

Two integration points into the optimizer:
    1. :func:`avg_feedback_score` -- a per-user/food composite rating fed to the
       XGBoost model as a feature (so retraining can learn feedback preferences).
    2. :func:`adjust_scores_with_feedback` -- a multiplicative modifier applied to
       the model probability at inference time (no retraining needed).

Both degrade to neutral defaults when feedback is sparse.
"""

from __future__ import annotations

from typing import List, Optional

from sqlalchemy.orm import Session

from backend.models import Feedback, MealLog

# Neutral composite rating (midpoint of the 1-5 scale) used when no data.
NEUTRAL_FEEDBACK = 3.0

# Modifier bounds and thresholds.
MODIFIER_MIN, MODIFIER_MAX = 0.8, 1.2
MIN_OCCASIONS = 3
LOW_ENJOYMENT = 3  # enjoyment < 3 is "low"
HIGH_ENJOYMENT = 4  # enjoyment > 4 is "high"


def _feedbacks_for_food(db: Session, user_id: int, food_id: int) -> List[Feedback]:
    """All feedback rows a user left on meals of a given food."""
    return (
        db.query(Feedback)
        .join(MealLog, Feedback.meal_log_id == MealLog.id)
        .filter(Feedback.user_id == user_id, MealLog.food_id == food_id)
        .all()
    )


def avg_feedback_score(db: Session, user_id: int, food_id: int) -> float:
    """Mean composite rating ((enjoyment+satiety+energy)/3) for a user/food.

    Returns :data:`NEUTRAL_FEEDBACK` (3.0) when the user has no feedback on the
    food -- keeping the training/inference feature well-defined for new foods.
    """
    rows = _feedbacks_for_food(db, user_id, food_id)
    if not rows:
        return NEUTRAL_FEEDBACK
    composites = [(r.enjoyment + r.satiety + r.energy) / 3.0 for r in rows]
    return round(sum(composites) / len(composites), 3)


def adjust_scores_with_feedback(
    db: Session, food_id: int, user_id: int, meal_log_id: Optional[int] = None
) -> float:
    """Return a multiplicative score modifier in ``[0.8, 1.2]`` for a food.

    Per-user heuristic based on how the user has historically rated this food:
        - low enjoyment (<3) on >= 3 occasions -> 0.9 (downweight)
        - high enjoyment (>4) on >= 3 occasions -> 1.1 (upweight)
        - otherwise -> 1.0 (neutral / insufficient data)

    ``meal_log_id`` is accepted for API symmetry but not required by the
    per-food heuristic.
    """
    rows = _feedbacks_for_food(db, user_id, food_id)
    if not rows:
        return 1.0

    low = sum(1 for r in rows if r.enjoyment < LOW_ENJOYMENT)
    high = sum(1 for r in rows if r.enjoyment > HIGH_ENJOYMENT)

    if low >= MIN_OCCASIONS:
        modifier = 0.9
    elif high >= MIN_OCCASIONS:
        modifier = 1.1
    else:
        modifier = 1.0
    return max(MODIFIER_MIN, min(modifier, MODIFIER_MAX))
