"""Meal endpoint: log a meal with macros/calories derived from the food."""

from __future__ import annotations

from datetime import date as date_cls, datetime, timedelta
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from backend.db import get_db, session_scope
from backend.models import Food, MealLog, Portion, User
from backend.schemas import MealCreate, MealHistoryItem, MealOut
from backend.services import cache, food_utils, rl_online
from backend.services.daily_summary import update_daily_summary

router = APIRouter(prefix="/meals", tags=["meals"])

VALID_MEAL_TYPES = {"breakfast", "lunch", "dinner", "snack"}


@router.get("", response_model=List[MealHistoryItem])
def list_meals(
    user_id: int = Query(...),
    date: Optional[str] = Query(default=None, description="YYYY-MM-DD (defaults to all)"),
    db: Session = Depends(get_db),
) -> List[MealHistoryItem]:
    """List a user's logged meals, most recent first, optionally for one date."""
    user = db.get(User, user_id)
    if user is None:
        raise HTTPException(404, "User not found")
    query = db.query(MealLog).filter(MealLog.user_id == user_id)
    if date:
        try:
            day = date_cls.fromisoformat(date)
        except ValueError:
            raise HTTPException(422, "date must be YYYY-MM-DD")
        start = datetime(day.year, day.month, day.day)
        query = query.filter(
            MealLog.timestamp >= start, MealLog.timestamp < start + timedelta(days=1)
        )
    rows = query.order_by(MealLog.timestamp.desc()).all()
    return [
        MealHistoryItem(
            id=r.id, food_id=r.food_id,
            food_name=r.food.name if r.food else "Food",
            grams_consumed=r.grams_consumed, meal_type=r.meal_type,
            timestamp=r.timestamp, protein_g=r.protein_g, carbs_g=r.carbs_g,
            fat_g=r.fat_g, calories=r.calories,
        )
        for r in rows
    ]


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


@router.delete("/{meal_id}")
def delete_meal(
    meal_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> dict:
    """Delete a logged meal and refresh that day's summary/cache."""
    meal = db.get(MealLog, meal_id)
    if meal is None:
        raise HTTPException(404, "Meal not found")
    user_id, day = meal.user_id, meal.timestamp.date()
    db.delete(meal)
    db.commit()
    cache.invalidate_user_optimize(user_id)
    background_tasks.add_task(_summary_task, user_id, day)
    return {"status": "deleted", "meal_id": meal_id}
