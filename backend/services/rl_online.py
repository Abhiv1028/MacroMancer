"""Online learning for the RL bandit (Phase 6, Part 3).

After a day's action + reward exist, update the LinUCB bandit with the observed
transition and persist an :class:`RLTransition`. Idempotent per day (keyed on the
day's reward), so it can be triggered repeatedly by background tasks without
double-counting.
"""

from __future__ import annotations

from datetime import date as date_cls, timedelta
from typing import Dict

from sqlalchemy.orm import Session

from backend.db import session_scope
from backend.models import RLActionLog, RLReward, RLState, RLTransition
from backend.services import rl_bandit, rl_reward
from backend.services.rl_actions import ACTION_NAMES


def _latest_action_log(db: Session, user_id: int, day: date_cls):
    """Most recent RLActionLog for a user on a day (None if absent)."""
    return (
        db.query(RLActionLog)
        .filter(RLActionLog.user_id == user_id, RLActionLog.date == day)
        .order_by(RLActionLog.created_at.desc(), RLActionLog.id.desc())
        .first()
    )


def update_bandit_from_day(db: Session, user_id: int, day: date_cls) -> Dict:
    """Update the bandit from a day's (context, action, reward) and log it.

    Returns a status dict. Skips (no update) when the action log or reward is
    missing, when the action is unknown, or when a transition already exists for
    the day's reward (idempotency).
    """
    action_log = _latest_action_log(db, user_id, day)
    if action_log is None:
        return {"status": "skipped", "reason": "no_action_log"}

    reward = (
        db.query(RLReward)
        .filter(RLReward.user_id == user_id, RLReward.date == day)
        .one_or_none()
    )
    if reward is None:
        return {"status": "skipped", "reason": "no_reward"}

    # Idempotency: one transition per day's reward.
    existing = (
        db.query(RLTransition)
        .filter(RLTransition.reward_id == reward.id)
        .first()
    )
    if existing is not None:
        return {
            "status": "skipped",
            "reason": "already_updated",
            "transition_id": existing.id,
        }

    action_name = action_log.action.name if action_log.action else None
    if action_name not in ACTION_NAMES:
        return {"status": "skipped", "reason": "unknown_action"}
    action_index = ACTION_NAMES.index(action_name)

    # Context used at selection time (fall back to the day's RLState).
    context_source = action_log.context_json
    if not context_source:
        state = (
            db.query(RLState)
            .filter(RLState.user_id == user_id, RLState.date == day)
            .one_or_none()
        )
        context_source = state.context_vector if state else {}
    context_vec = rl_reward.context_dict_to_vector(context_source)

    # Next state (next day), if it exists -- stored for future (SARSA-style) use.
    next_state = (
        db.query(RLState)
        .filter(RLState.user_id == user_id, RLState.date == day + timedelta(days=1))
        .one_or_none()
    )

    # Standard LinUCB update uses (action, context, reward); next_state is stored
    # on the transition for future extensions but not consumed by the update.
    bandit = rl_bandit.get_bandit()
    bandit.update(action_index, context_vec, float(reward.reward))
    rl_bandit.save_bandit()

    transition = RLTransition(
        user_id=user_id,
        action_log_id=action_log.id,
        reward_id=reward.id,
        next_state_id=next_state.id if next_state else None,
    )
    db.add(transition)
    db.commit()
    db.refresh(transition)

    return {
        "status": "updated",
        "action": action_name,
        "reward": float(reward.reward),
        "update_count": bandit.update_count,
        "transition_id": transition.id,
    }


def reward_and_update_task(
    user_id: int, day: date_cls, day_complete: bool = False
) -> None:
    """Background task: recompute the day's reward, then (optionally) learn.

    Always refreshes reward + context. Only performs the bandit update when
    ``day_complete`` is set (e.g. dinner logged / evening / feedback given), so
    the once-per-day online update tends to use a fuller day's reward.
    """
    try:
        with session_scope() as db:
            rl_reward.recompute_day(db, user_id, day)
            if day_complete:
                update_bandit_from_day(db, user_id, day)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[rl] reward_and_update failed for user {user_id} on {day}: {exc}")
