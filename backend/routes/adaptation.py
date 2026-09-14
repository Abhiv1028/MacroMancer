"""Phase 3 endpoints: feedback, body composition, adaptive TDEE, and goals.

All routes are mounted under ``/api/v1`` (see ``backend/main.py``). Long-running
recomputations use FastAPI ``BackgroundTasks`` so responses stay fast.
"""

from __future__ import annotations

from datetime import date as date_cls

from typing import List

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from backend.db import get_db, session_scope
from backend.rate_limit import limiter
from backend.models import BodyComposition, Feedback, MealLog, User
from backend.schemas import (
    BodyCompCreate,
    BodyCompEntry,
    BodyCompResponse,
    FeedbackCreate,
    FeedbackResponse,
    GoalsResponse,
    GoalsUpdate,
    MacroTargetOut,
    TDEEInfo,
)
from backend.services import rl_online, tdee_calculator
from backend.services.daily_summary import update_daily_summary
from backend.services.macro_calculator import (
    calculate_bmr,
    calculate_tdee,
    recalculate_targets,
)

router = APIRouter(tags=["adaptation"])

VALID_GOALS = {"cut", "maintain", "bulk"}

# Targets are re-derived from adaptive TDEE only when it diverges from the
# static estimate by more than this fraction.
ADAPTIVE_TARGET_THRESHOLD = 0.05


def _static_tdee(user: User) -> float:
    """Static Mifflin-St Jeor maintenance TDEE for the user's current profile."""
    bmr = calculate_bmr(user.weight_kg, user.height_cm, user.age, user.sex)
    return round(calculate_tdee(bmr, user.activity_level), 1)


def _summary_task(user_id: int, day: date_cls) -> None:
    """Background task: refresh a user's DailySummary in its own session."""
    try:
        with session_scope() as db:
            update_daily_summary(db, user_id, day)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[adaptation] Daily-summary update failed: {exc}")


@router.post("/feedback", response_model=FeedbackResponse, status_code=201)
@limiter.limit("60/minute")
def create_feedback(
    request: Request,
    payload: FeedbackCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> FeedbackResponse:
    """Record subjective feedback and refresh that day's summary in the background."""
    user = db.get(User, payload.user_id)
    if user is None:
        raise HTTPException(404, "User not found")

    day = date_cls.today()
    if payload.meal_log_id is not None:
        meal = db.get(MealLog, payload.meal_log_id)
        if meal is None or meal.user_id != user.id:
            raise HTTPException(422, "meal_log_id does not belong to this user")
        day = meal.timestamp.date()

    feedback = Feedback(
        user_id=user.id,
        meal_log_id=payload.meal_log_id,
        enjoyment=payload.enjoyment,
        satiety=payload.satiety,
        energy=payload.energy,
        workout_performance=payload.workout_performance,
        notes=payload.notes,
    )
    db.add(feedback)
    db.commit()
    db.refresh(feedback)

    background_tasks.add_task(_summary_task, user.id, day)
    # Phase 6: feedback refines the day's reward and is a strong "day done"
    # signal -> recompute reward and learn online in the background.
    background_tasks.add_task(
        rl_online.reward_and_update_task, user.id, day, True
    )
    return FeedbackResponse(status="ok", feedback_id=feedback.id)


@router.post("/body_composition", response_model=BodyCompResponse, status_code=201)
@limiter.limit("60/minute")
def create_body_composition(
    request: Request,
    payload: BodyCompCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> BodyCompResponse:
    """Record a body-composition entry, update the user's weight, and recalc targets."""
    user = db.get(User, payload.user_id)
    if user is None:
        raise HTTPException(404, "User not found")

    entry_date = payload.date or date_cls.today()
    comp = BodyComposition(
        user_id=user.id,
        date=entry_date,
        weight_kg=payload.weight_kg,
        body_fat_percent=payload.body_fat_percent,
        lean_mass_kg=payload.lean_mass_kg,
    )
    db.add(comp)

    # Keep the user's canonical weight in sync and recompute targets.
    user.weight_kg = payload.weight_kg
    db.commit()
    db.refresh(comp)

    targets = recalculate_targets(db, user, entry_date)
    background_tasks.add_task(_summary_task, user.id, entry_date)

    return BodyCompResponse(
        id=comp.id,
        user_id=user.id,
        date=comp.date,
        weight_kg=comp.weight_kg,
        body_fat_percent=comp.body_fat_percent,
        lean_mass_kg=comp.lean_mass_kg,
        updated_targets=MacroTargetOut.model_validate(targets),
    )


@router.get("/users/{user_id}/tdee", response_model=TDEEInfo)
def get_tdee(user_id: int, db: Session = Depends(get_db)) -> TDEEInfo:
    """Return static vs adaptive TDEE, updating targets if they diverge >5%."""
    user = db.get(User, user_id)
    if user is None:
        raise HTTPException(404, "User not found")

    current = _static_tdee(user)
    result = tdee_calculator.calculate_adaptive_tdee(db, user_id)

    adaptive_value = result.tdee if result.method_used == "adaptive" else None
    if (
        adaptive_value is not None
        and current > 0
        and abs(adaptive_value - current) / current > ADAPTIVE_TARGET_THRESHOLD
    ):
        recalculate_targets(db, user, date_cls.today(), tdee=adaptive_value)

    return TDEEInfo(
        current_tdee=current,
        adaptive_tdee=adaptive_value,
        last_updated=result.last_updated,
        method_used=result.method_used,
    )


@router.put("/users/{user_id}/goals", response_model=GoalsResponse)
def update_goals(
    user_id: int, payload: GoalsUpdate, db: Session = Depends(get_db)
) -> GoalsResponse:
    """Update a user's goal and recompute targets using the latest (adaptive) TDEE."""
    user = db.get(User, user_id)
    if user is None:
        raise HTTPException(404, "User not found")
    if payload.goal.lower() not in VALID_GOALS:
        raise HTTPException(422, f"goal must be one of {sorted(VALID_GOALS)}")

    user.goal = payload.goal.lower()
    db.commit()

    result = tdee_calculator.calculate_adaptive_tdee(db, user_id)
    tdee = result.tdee if result.method_used == "adaptive" else None
    targets = recalculate_targets(db, user, date_cls.today(), tdee=tdee)

    return GoalsResponse(
        user_id=user.id,
        goal=user.goal,
        method_used=result.method_used,
        targets=MacroTargetOut.model_validate(targets),
    )


@router.get("/body_composition/history/{user_id}", response_model=List[BodyCompEntry])
def body_composition_history(
    user_id: int, db: Session = Depends(get_db)
) -> List[BodyCompEntry]:
    """Return a user's body-composition measurements, oldest first (for charts)."""
    user = db.get(User, user_id)
    if user is None:
        raise HTTPException(404, "User not found")
    rows = (
        db.query(BodyComposition)
        .filter(BodyComposition.user_id == user_id)
        .order_by(BodyComposition.date.asc(), BodyComposition.id.asc())
        .all()
    )
    return [BodyCompEntry.model_validate(r) for r in rows]
