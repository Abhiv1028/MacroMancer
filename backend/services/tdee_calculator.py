"""Adaptive TDEE estimation from logged intake and body-weight trend.

The energy-balance identity says that, over a window, TDEE equals average intake
plus the average daily energy deficit/surplus implied by weight change:

    TDEE = (total_intake + weight_change_kg * KCAL_PER_KG) / days

where ``weight_change_kg`` is start-minus-end weight (positive when losing).
Falls back to static Mifflin-St Jeor when there isn't enough data.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as date_cls, timedelta
from typing import List, Optional, Tuple

from sqlalchemy.orm import Session

from backend.models import BodyComposition, DailySummary, User
from backend.services.macro_calculator import calculate_bmr, calculate_tdee

# Approximate energy content of 1 kg of body-weight change.
KCAL_PER_KG = 7700.0
# Minimum days of intake data required to trust the adaptive estimate.
MIN_DAYS_FOR_ADAPTIVE = 7
# A physiologically implausible rate of change -> treat the data as corrupt.
MAX_KG_CHANGE_PER_DAY = 5.0


@dataclass
class TDEEResult:
    """Outcome of a TDEE computation."""

    tdee: float
    method_used: str  # "adaptive" | "mifflin"
    last_updated: Optional[date_cls]


def _static_tdee(user: User) -> float:
    """Static Mifflin-St Jeor maintenance TDEE for a user's current profile."""
    bmr = calculate_bmr(user.weight_kg, user.height_cm, user.age, user.sex)
    return round(calculate_tdee(bmr, user.activity_level), 1)


def _weight_series(
    db: Session, user_id: int, start: date_cls, end: date_cls
) -> List[Tuple[date_cls, float]]:
    """Sorted (date, weight) points in [start, end], preferring body comps.

    Combines explicit BodyComposition entries with any weights carried on
    DailySummary rows, de-duplicated by date (body-comp wins).
    """
    weights: dict = {}
    summaries = (
        db.query(DailySummary)
        .filter(
            DailySummary.user_id == user_id,
            DailySummary.date >= start,
            DailySummary.date <= end,
            DailySummary.weight_kg.isnot(None),
        )
        .all()
    )
    for s in summaries:
        weights[s.date] = s.weight_kg
    comps = (
        db.query(BodyComposition)
        .filter(
            BodyComposition.user_id == user_id,
            BodyComposition.date >= start,
            BodyComposition.date <= end,
        )
        .all()
    )
    for c in comps:
        weights[c.date] = c.weight_kg  # body-comp overrides summary weight
    return sorted(weights.items())


def _edge_weights(weights: List[float]) -> Tuple[float, float]:
    """Representative start/end weights for a series.

    With >= 6 points, average the first three and last three (noise reduction);
    with fewer, use the single first and last points so the endpoints don't
    overlap and cancel the measured change.
    """
    if len(weights) >= 6:
        first = sum(weights[:3]) / 3.0
        last = sum(weights[-3:]) / 3.0
    else:
        first = weights[0]
        last = weights[-1]
    return first, last


def calculate_adaptive_tdee(
    db: Session, user_id: int, days: int = 14
) -> TDEEResult:
    """Estimate adaptive TDEE for a user over the last ``days`` days.

    Args:
        db: Active database session.
        user_id: Target user.
        days: Look-back window length in days.

    Returns:
        A :class:`TDEEResult`. ``method_used`` is ``"adaptive"`` when enough
        intake + weight data exists, else ``"mifflin"`` (static fallback).
    """
    user = db.get(User, user_id)
    if user is None:
        raise ValueError(f"User {user_id} not found")

    end = date_cls.today()
    start = end - timedelta(days=days - 1)

    summaries = (
        db.query(DailySummary)
        .filter(
            DailySummary.user_id == user_id,
            DailySummary.date >= start,
            DailySummary.date <= end,
        )
        .order_by(DailySummary.date.asc())
        .all()
    )
    intake_days = [s for s in summaries if s.total_calories > 0]

    weight_points = _weight_series(db, user_id, start, end)

    # Guard: need enough intake days and at least two weight measurements.
    if len(intake_days) < MIN_DAYS_FOR_ADAPTIVE or len(weight_points) < 2:
        print(
            f"[tdee] Insufficient data for user {user_id} "
            f"(intake_days={len(intake_days)}, weight_points={len(weight_points)}); "
            "falling back to Mifflin-St Jeor."
        )
        return TDEEResult(
            tdee=_static_tdee(user), method_used="mifflin", last_updated=None
        )

    weights = [w for _, w in weight_points]
    weight_first, weight_last = _edge_weights(weights)
    weight_change = weight_first - weight_last  # positive => weight lost

    total_intake = sum(s.total_calories for s in intake_days)
    n_days = len(intake_days)

    # Guard: division by zero, and physiologically impossible weight swings that
    # signal bad data (e.g. a fat-fingered weigh-in). Fall back to static.
    if n_days <= 0 or abs(weight_change) / n_days > MAX_KG_CHANGE_PER_DAY:
        print(
            f"[tdee] Rejecting adaptive estimate for user {user_id} "
            f"(n_days={n_days}, weight_change={weight_change:.2f}kg); "
            "falling back to Mifflin-St Jeor."
        )
        return TDEEResult(
            tdee=_static_tdee(user), method_used="mifflin", last_updated=None
        )

    tdee = (total_intake + weight_change * KCAL_PER_KG) / n_days
    # Clamp to a sane physiological range to absorb noisy short windows.
    tdee = max(1000.0, min(tdee, 6000.0))

    return TDEEResult(
        tdee=round(tdee, 1),
        method_used="adaptive",
        last_updated=intake_days[-1].date,
    )
