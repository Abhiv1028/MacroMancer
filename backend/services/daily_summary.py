"""Maintain the denormalized :class:`DailySummary` roll-up table.

A summary row aggregates a user's meal-log totals and latest known weight for a
given calendar day. It is refreshed (idempotently) whenever a meal, feedback, or
body-composition entry is logged, keeping adaptive-TDEE math cheap.
"""

from __future__ import annotations

from datetime import date as date_cls, datetime, timedelta
from typing import Optional

from sqlalchemy.orm import Session

from backend.models import BodyComposition, DailySummary, MealLog, User
from backend.services.macro_calculator import calculate_bmr, calculate_tdee


def _latest_weight_on_or_before(
    db: Session, user_id: int, day: date_cls
) -> Optional[float]:
    """Most recent recorded body weight on or before ``day``, if any."""
    row = (
        db.query(BodyComposition)
        .filter(
            BodyComposition.user_id == user_id,
            BodyComposition.date <= day,
        )
        .order_by(BodyComposition.date.desc(), BodyComposition.id.desc())
        .first()
    )
    return row.weight_kg if row is not None else None


def update_daily_summary(
    db: Session, user_id: int, day: Optional[date_cls] = None
) -> Optional[DailySummary]:
    """Recompute the summary row for ``user_id`` on ``day`` from source tables.

    Returns the upserted :class:`DailySummary`, or ``None`` if the user is gone.
    Safe to call repeatedly and from background tasks (commits its own work).
    """
    if day is None:
        day = date_cls.today()

    user = db.get(User, user_id)
    if user is None:
        return None

    start = datetime(day.year, day.month, day.day)
    end = start + timedelta(days=1)
    logs = (
        db.query(MealLog)
        .filter(
            MealLog.user_id == user_id,
            MealLog.timestamp >= start,
            MealLog.timestamp < end,
        )
        .all()
    )

    totals = {"calories": 0.0, "protein": 0.0, "carbs": 0.0, "fat": 0.0}
    for log in logs:
        totals["calories"] += log.calories
        totals["protein"] += log.protein_g
        totals["carbs"] += log.carbs_g
        totals["fat"] += log.fat_g

    weight = _latest_weight_on_or_before(db, user_id, day)

    # A cheap static TDEE estimate for reference (adaptive TDEE is computed
    # separately from the summary series to avoid circular dependencies).
    tdee_estimate = None
    effective_weight = weight if weight is not None else user.weight_kg
    if effective_weight:
        bmr = calculate_bmr(effective_weight, user.height_cm, user.age, user.sex)
        tdee_estimate = round(calculate_tdee(bmr, user.activity_level), 1)

    summary = (
        db.query(DailySummary)
        .filter(DailySummary.user_id == user_id, DailySummary.date == day)
        .one_or_none()
    )
    if summary is None:
        summary = DailySummary(user_id=user_id, date=day)
        db.add(summary)

    summary.total_calories = round(totals["calories"], 2)
    summary.total_protein = round(totals["protein"], 2)
    summary.total_carbs = round(totals["carbs"], 2)
    summary.total_fat = round(totals["fat"], 2)
    summary.weight_kg = weight
    summary.tdee_estimate = tdee_estimate

    db.commit()
    db.refresh(summary)
    return summary
