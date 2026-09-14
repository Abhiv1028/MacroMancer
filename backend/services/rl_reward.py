"""RL reward computation and context construction (Phase 6, Part 1).

- :func:`compute_reward` scores a user's day (macro adherence + feedback) and
  upserts an :class:`RLReward` row (idempotent per user/day).
- :func:`build_context` builds the normalized decision-context feature vector and
  upserts an :class:`RLState` row.

Both are pure w.r.t. the DB session passed in; a background-task wrapper
(:func:`recompute_day_task`) opens its own session for use with FastAPI
``BackgroundTasks``.
"""

from __future__ import annotations

from datetime import date as date_cls, datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
from sqlalchemy import func, or_
from sqlalchemy.orm import Session

from backend.db import session_scope
from backend.models import Feedback, MealLog, RLReward, RLState, User
from backend.services.macro_calculator import clamp, get_or_create_targets

# --- encodings ------------------------------------------------------------- #
GOAL_ENCODING = {"cut": 0, "maintain": 1, "bulk": 2}
GOAL_MAX = 2
ACTIVITY_ORDER = ["sedentary", "light", "moderate", "very", "extra"]
ACTIVITY_MAX = len(ACTIVITY_ORDER) - 1

# Ordered context features (index == position in the returned numpy vector).
CONTEXT_FEATURES: List[str] = [
    "protein_remaining_ratio",
    "carbs_remaining_ratio",
    "fat_remaining_ratio",
    "hour",
    "day_of_week",
    "goal_encoded",
    "activity_level_encoded",
]
CONTEXT_DIM = len(CONTEXT_FEATURES)

# Reward weights.
W_MACRO = 0.5
W_FEEDBACK = 0.3
W_SATIETY = 0.2
FEEDBACK_MAX = 5.0  # 1-5 Likert scale


def _sum_day_macros(db: Session, user_id: int, day: date_cls) -> Dict[str, float]:
    """Sum protein/carbs/fat consumed by a user on ``day`` from meal logs."""
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
    totals = {"protein_g": 0.0, "carbs_g": 0.0, "fat_g": 0.0}
    for log in logs:
        totals["protein_g"] += log.protein_g
        totals["carbs_g"] += log.carbs_g
        totals["fat_g"] += log.fat_g
    return totals


def _day_feedbacks(db: Session, user_id: int, day: date_cls) -> List[Feedback]:
    """Feedback tied to a user's meals on ``day`` or created on ``day``."""
    start = datetime(day.year, day.month, day.day)
    end = start + timedelta(days=1)
    day_meal_ids = [
        mid
        for (mid,) in db.query(MealLog.id).filter(
            MealLog.user_id == user_id,
            MealLog.timestamp >= start,
            MealLog.timestamp < end,
        )
    ]
    return (
        db.query(Feedback)
        .filter(
            Feedback.user_id == user_id,
            or_(
                Feedback.meal_log_id.in_(day_meal_ids) if day_meal_ids else False,
                func.date(Feedback.created_at) == day.isoformat(),
            ),
        )
        .all()
    )


def _macro_adherence(targets, eaten: Dict[str, float]) -> float:
    """1 - mean(|target-eaten|/target) across macros, clamped to [0, 1]."""
    errors = []
    for macro, target_val in (
        ("protein_g", targets.protein_g),
        ("carbs_g", targets.carbs_g),
        ("fat_g", targets.fat_g),
    ):
        if target_val and target_val > 0:
            errors.append(abs(target_val - eaten.get(macro, 0.0)) / target_val)
    if not errors:
        return 0.0
    return clamp(1.0 - (sum(errors) / len(errors)), 0.0, 1.0)


def compute_reward(db: Session, user_id: int, day: date_cls) -> float:
    """Compute and persist the day's scalar reward for a user (idempotent).

    ``reward = 0.5*macro_adherence + 0.3*feedback_avg + 0.2*satiety_norm`` where
    ``feedback_avg`` = mean(enjoyment, energy)/5 (satiety is excluded here since
    it has its own term) and ``satiety_norm`` = mean(satiety)/5. Every component
    is in [0, 1]; feedback components default to 0 when no feedback was logged.
    """
    user = db.get(User, user_id)
    if user is None:
        raise ValueError(f"User {user_id} not found")

    targets = get_or_create_targets(db, user, day)
    eaten = _sum_day_macros(db, user_id, day)
    macro_adherence = _macro_adherence(targets, eaten)

    feedbacks = _day_feedbacks(db, user_id, day)
    if feedbacks:
        # Satiety is scored on its own (W_SATIETY term), so it's excluded here to
        # avoid double-counting -- feedback_avg uses enjoyment + energy only.
        feedback_avg = float(
            np.mean([(f.enjoyment + f.energy) / 2.0 for f in feedbacks]) / FEEDBACK_MAX
        )
        satiety_norm = float(np.mean([f.satiety for f in feedbacks]) / FEEDBACK_MAX)
    else:
        feedback_avg = 0.0
        satiety_norm = 0.0

    reward = round(
        W_MACRO * macro_adherence + W_FEEDBACK * feedback_avg + W_SATIETY * satiety_norm,
        4,
    )
    components = {
        "macro_adherence": round(macro_adherence, 4),
        "feedback_score": round(feedback_avg, 4),
        "satiety": round(satiety_norm, 4),
    }

    row = (
        db.query(RLReward)
        .filter(RLReward.user_id == user_id, RLReward.date == day)
        .one_or_none()
    )
    if row is None:
        row = RLReward(user_id=user_id, date=day)
        db.add(row)
    row.reward = reward
    row.components = components
    db.commit()
    return reward


def build_context(db: Session, user_id: int, day: date_cls) -> np.ndarray:
    """Build and persist the normalized context vector for a user/day.

    Returns a numpy array ordered per :data:`CONTEXT_FEATURES`; each feature is
    normalized to [0, 1]. Also upserts the :class:`RLState` row.
    """
    user = db.get(User, user_id)
    if user is None:
        raise ValueError(f"User {user_id} not found")

    targets = get_or_create_targets(db, user, day)
    eaten = _sum_day_macros(db, user_id, day)

    def _remaining_ratio(target_val: float, eaten_val: float) -> float:
        if not target_val or target_val <= 0:
            return 0.0
        return clamp(max(target_val - eaten_val, 0.0) / target_val, 0.0, 1.0)

    hour = datetime.now().hour if day == date_cls.today() else 12
    goal_enc = GOAL_ENCODING.get(str(user.goal).lower(), 1)
    try:
        activity_enc = ACTIVITY_ORDER.index(str(user.activity_level).lower())
    except ValueError:
        activity_enc = 2  # moderate

    values = {
        "protein_remaining_ratio": _remaining_ratio(targets.protein_g, eaten["protein_g"]),
        "carbs_remaining_ratio": _remaining_ratio(targets.carbs_g, eaten["carbs_g"]),
        "fat_remaining_ratio": _remaining_ratio(targets.fat_g, eaten["fat_g"]),
        "hour": hour / 23.0,
        "day_of_week": day.weekday() / 6.0,
        "goal_encoded": goal_enc / GOAL_MAX,
        "activity_level_encoded": activity_enc / ACTIVITY_MAX,
    }
    vector = np.array([values[name] for name in CONTEXT_FEATURES], dtype=float)

    row = (
        db.query(RLState)
        .filter(RLState.user_id == user_id, RLState.date == day)
        .one_or_none()
    )
    if row is None:
        row = RLState(user_id=user_id, date=day)
        db.add(row)
    row.context_vector = {k: round(v, 6) for k, v in values.items()}
    db.commit()
    return vector


def context_dict_to_vector(context: Dict[str, float]) -> np.ndarray:
    """Convert a stored context dict into the ordered numpy feature vector.

    Missing features default to 0.0 so older/partial records stay usable.
    """
    return np.array(
        [float(context.get(name, 0.0)) for name in CONTEXT_FEATURES], dtype=float
    )


def recompute_day(db: Session, user_id: int, day: date_cls) -> Tuple[np.ndarray, float]:
    """Rebuild the day's context and reward together. Returns (context, reward)."""
    context = build_context(db, user_id, day)
    reward = compute_reward(db, user_id, day)
    return context, reward


def recompute_day_task(user_id: int, day: date_cls) -> None:
    """Background-task wrapper: recompute context+reward in a fresh session."""
    try:
        with session_scope() as db:
            recompute_day(db, user_id, day)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[rl] Reward recompute failed for user {user_id} on {day}: {exc}")
