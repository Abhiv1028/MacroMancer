"""RL vs XGBoost evaluation and monitoring (Phase 6, Part 4).

Compares average daily reward on **RL days** (a bandit action was logged) vs
**XGBoost days** (no action logged -> the default optimizer was used). Drives the
fallback decision in ``POST /api/v1/rl/recommend``.
"""

from __future__ import annotations

from datetime import date as date_cls, timedelta
from typing import Dict

import numpy as np
from sqlalchemy.orm import Session

from backend.models import RLActionLog, RLDailyPerformance, RLReward

# Minimum labeled days required in *each* group to trust the comparison.
MIN_DAYS_PER_GROUP = 5


def evaluate_rl_vs_xgboost(
    db: Session, user_id: int, days: int = 14
) -> Dict:
    """Compare mean reward on RL days vs XGBoost days over a look-back window.

    Returns a dict with ``rl_avg_reward``, ``xgboost_avg_reward``,
    ``improvement_percent`` (None when the XGBoost baseline is 0/absent),
    ``recommendation`` (``"use_rl"`` | ``"use_xgboost"`` | ``"insufficient_data"``),
    and the ``rl_days`` / ``xgboost_days`` counts.
    """
    end = date_cls.today()
    start = end - timedelta(days=days - 1)

    rl_dates = {
        d
        for (d,) in db.query(RLActionLog.date)
        .filter(
            RLActionLog.user_id == user_id,
            RLActionLog.date >= start,
            RLActionLog.date <= end,
        )
        .distinct()
    }

    rewards = (
        db.query(RLReward)
        .filter(
            RLReward.user_id == user_id,
            RLReward.date >= start,
            RLReward.date <= end,
        )
        .all()
    )
    rl_rewards = [r.reward for r in rewards if r.date in rl_dates]
    xgb_rewards = [r.reward for r in rewards if r.date not in rl_dates]

    rl_avg = float(np.mean(rl_rewards)) if rl_rewards else None
    xgb_avg = float(np.mean(xgb_rewards)) if xgb_rewards else None

    result: Dict = {
        "rl_avg_reward": round(rl_avg, 4) if rl_avg is not None else None,
        "xgboost_avg_reward": round(xgb_avg, 4) if xgb_avg is not None else None,
        "improvement_percent": None,
        "recommendation": "insufficient_data",
        "rl_days": len(rl_rewards),
        "xgboost_days": len(xgb_rewards),
    }

    if len(rl_rewards) < MIN_DAYS_PER_GROUP or len(xgb_rewards) < MIN_DAYS_PER_GROUP:
        return result

    if xgb_avg and xgb_avg > 0:
        result["improvement_percent"] = round((rl_avg - xgb_avg) / xgb_avg * 100, 2)
    result["recommendation"] = "use_rl" if rl_avg >= xgb_avg else "use_xgboost"
    return result


def record_performance(
    db: Session, user_id: int, day: date_cls, evaluation: Dict
) -> RLDailyPerformance:
    """Upsert a daily RL-vs-XGBoost performance summary (for monitoring/cron)."""
    row = (
        db.query(RLDailyPerformance)
        .filter(RLDailyPerformance.user_id == user_id, RLDailyPerformance.date == day)
        .one_or_none()
    )
    if row is None:
        row = RLDailyPerformance(user_id=user_id, date=day)
        db.add(row)
    row.rl_reward_avg = evaluation.get("rl_avg_reward")
    row.xgboost_reward_avg = evaluation.get("xgboost_avg_reward")
    row.improvement = evaluation.get("improvement_percent")
    db.commit()
    db.refresh(row)
    return row


def resolve_use_rl(
    db: Session, user_id: int, days: int = 14
) -> tuple:
    """Decide whether to use RL for a user, honoring manual overrides.

    Returns ``(use_rl: bool, reason: Optional[str], evaluation: Optional[dict])``.
    A manual :class:`RLOverride` wins; otherwise RL is used unless the evaluation
    recommends ``use_xgboost``.
    """
    from backend.models import RLOverride

    override = (
        db.query(RLOverride).filter(RLOverride.user_id == user_id).one_or_none()
    )
    if override is not None:
        if override.forced_rl:
            return (True, None, None)
        return (False, "manual_override", None)

    evaluation = evaluate_rl_vs_xgboost(db, user_id, days=days)
    if evaluation["recommendation"] == "use_xgboost":
        return (False, "rl_underperforming", evaluation)
    return (True, None, evaluation)
