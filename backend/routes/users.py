"""User endpoints: creation and daily macro targets."""

from __future__ import annotations

from datetime import date as date_cls, timedelta
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from backend.db import get_db
from backend.models import User
from backend.schemas import (
    MacroTargetOut,
    UserCreate,
    UserCreateResponse,
    UserOut,
)
from backend.services.macro_calculator import get_or_create_targets

router = APIRouter(prefix="/users", tags=["users"])

VALID_ACTIVITY = {"sedentary", "light", "moderate", "very", "extra"}
VALID_GOALS = {"cut", "maintain", "bulk"}


@router.post("", response_model=UserCreateResponse, status_code=201)
def create_user(payload: UserCreate, db: Session = Depends(get_db)) -> UserCreateResponse:
    """Create a user and compute+store today's macro targets."""
    if payload.activity_level.lower() not in VALID_ACTIVITY:
        raise HTTPException(422, f"activity_level must be one of {sorted(VALID_ACTIVITY)}")
    if payload.goal.lower() not in VALID_GOALS:
        raise HTTPException(422, f"goal must be one of {sorted(VALID_GOALS)}")

    user = User(
        age=payload.age,
        weight_kg=payload.weight_kg,
        height_cm=payload.height_cm,
        activity_level=payload.activity_level.lower(),
        goal=payload.goal.lower(),
        sex=(payload.sex or "female").lower(),
        dietary_restrictions=payload.dietary_restrictions,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    targets = get_or_create_targets(db, user, date_cls.today())
    return UserCreateResponse(
        user=UserOut.model_validate(user),
        targets=MacroTargetOut.model_validate(targets),
    )


@router.get("/{user_id}", response_model=UserOut)
def get_user(user_id: int, db: Session = Depends(get_db)) -> UserOut:
    """Fetch a single user."""
    user = db.get(User, user_id)
    if user is None:
        raise HTTPException(404, "User not found")
    return UserOut.model_validate(user)


@router.get("/{user_id}/targets", response_model=List[MacroTargetOut])
def get_targets(
    user_id: int,
    date: Optional[str] = Query(
        default=None, description="Start date as YYYY-MM-DD (defaults to today)."
    ),
    days: int = Query(
        default=1,
        ge=1,
        le=31,
        description="Number of consecutive days to (pre-)generate, starting at `date`.",
    ),
    db: Session = Depends(get_db),
) -> List[MacroTargetOut]:
    """Get macro targets starting at `date`, auto-generating any that are missing.

    Returns a list of length `days` (defaults to a single day). Use `days=7` to
    pre-generate a week's worth of targets in one call.
    """
    user = db.get(User, user_id)
    if user is None:
        raise HTTPException(404, "User not found")

    start_date = date_cls.today()
    if date:
        try:
            start_date = date_cls.fromisoformat(date)
        except ValueError:
            raise HTTPException(422, "date must be in YYYY-MM-DD format")

    results = []
    for offset in range(days):
        target = get_or_create_targets(db, user, start_date + timedelta(days=offset))
        results.append(MacroTargetOut.model_validate(target))
    return results
