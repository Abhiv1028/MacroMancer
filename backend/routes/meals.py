"""Meal endpoint: log a meal with macros/calories derived from the food."""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.db import get_db, session_scope
from backend.models import Food, MealLog, Portion, User
from backend.schemas import MealCreate, MealOut
from backend.services import cache, food_utils, rl_online
from backend.services.daily_summary import update_daily_summary

router = APIRouter(prefix="/meals", tags=["meals"])

VALID_MEAL_TYPES = {"breakfast", "lunch", "dinner", "snack"}


def _summary_task(user_id: int, day) -> None:
    """Background task: refresh the user's DailySummary for ``day``."""
    try:
        with session_scope() as db:
            update_daily_summary(db, user_id, day)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[meals] Daily-summary update failed: {exc}")


@router.post("", response_model=MealOut, status_code=201)
def log_meal(
    payload: MealCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> MealOut:
    """Log a meal. Macros/calories are computed from the food's nutrients.

    The insert itself is unchanged; a background task refreshes the day's
    :class:`DailySummary` (Phase 3) without affecting the response.
    """
    if payload.meal_type.lower() not in VALID_MEAL_TYPES:
        raise HTTPException(422, f"meal_type must be one of {sorted(VALID_MEAL_TYPES)}")

    user = db.get(User, payload.user_id)
    if user is None:
        raise HTTPException(404, "User not found")

    food = db.get(Food, payload.food_id)
    if food is None:
        raise HTTPException(404, "Food not found")

    if payload.portion_id is not None:
        portion = db.get(Portion, payload.portion_id)
        if portion is None or portion.food_id != food.id:
            raise HTTPException(422, "portion_id does not belong to this food")

    macros = food_utils.macros_for_grams(food, payload.grams)

    meal = MealLog(
        user_id=user.id,
        food_id=food.id,
        portion_id=payload.portion_id,
        grams_consumed=payload.grams,
        meal_type=payload.meal_type.lower(),
        timestamp=payload.timestamp or datetime.now(),
        protein_g=macros["protein_g"],
        carbs_g=macros["carbs_g"],
        fat_g=macros["fat_g"],
        calories=macros["calories"],
    )
    db.add(meal)
    db.commit()
    db.refresh(meal)

    # Logging changes remaining macros -> stale optimize cache for this user.
    cache.invalidate_user_optimize(user.id)
    day = meal.timestamp.date()
    background_tasks.add_task(_summary_task, user.id, day)
    # Phase 6: recompute the day's RL reward, and learn online once the day
    # looks complete (dinner logged or evening) -- see rl_online.
    day_complete = meal.meal_type == "dinner" or datetime.now().hour >= 20
    background_tasks.add_task(
        rl_online.reward_and_update_task, user.id, day, day_complete
    )
    return MealOut.model_validate(meal)
