"""Phase 6 RL endpoints (Part 1): reward computation and reward history.

Mounted under ``/api/v1`` (see ``backend/main.py``) as ``/api/v1/rl/...``.
"""

from __future__ import annotations

from datetime import date as date_cls, datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from backend.db import get_db
from backend.models import RLActionLog, RLFallbackLog, RLOverride, RLReward, RLState, User
from backend.schemas import (
    RewardComputeRequest,
    RewardComputeResponse,
    RewardResponse,
    RLEvalResponse,
    RLRecommendedFood,
    RLRecommendRequest,
    RLRecommendResponse,
    RLStatusResponse,
    RLSwitchRequest,
    RLSwitchResponse,
    RLUpdateRequest,
    RLUpdateResponse,
)
from backend.services import (
    optimizer,
    rl_actions,
    rl_bandit,
    rl_eval,
    rl_online,
    rl_reward,
)
from backend.config import settings
from backend.services.macro_calculator import get_or_create_targets
from backend.services.meal_type_detector import detect_meal_type

router = APIRouter(prefix="/rl", tags=["rl"])


@router.post("/compute_reward", response_model=RewardComputeResponse)
def compute_reward(
    payload: RewardComputeRequest, db: Session = Depends(get_db)
) -> RewardComputeResponse:
    """Manually (re)compute the reward + context for a user/day (backfill-safe)."""
    user = db.get(User, payload.user_id)
    if user is None:
        raise HTTPException(404, "User not found")

    day = payload.date or date_cls.today()
    _, reward = rl_reward.recompute_day(db, user.id, day)

    reward_row = (
        db.query(RLReward)
        .filter(RLReward.user_id == user.id, RLReward.date == day)
        .one_or_none()
    )
    state_row = (
        db.query(RLState)
        .filter(RLState.user_id == user.id, RLState.date == day)
        .one_or_none()
    )
    return RewardComputeResponse(
        user_id=user.id,
        date=day,
        reward=reward,
        components=reward_row.components if reward_row else {},
        context=state_row.context_vector if state_row else {},
    )


@router.get("/rewards/{user_id}", response_model=List[RewardResponse])
def list_rewards(
    user_id: int,
    start_date: Optional[str] = Query(default=None, description="YYYY-MM-DD"),
    end_date: Optional[str] = Query(default=None, description="YYYY-MM-DD"),
    db: Session = Depends(get_db),
) -> List[RewardResponse]:
    """List a user's daily rewards, optionally bounded by a date range."""
    user = db.get(User, user_id)
    if user is None:
        raise HTTPException(404, "User not found")

    query = db.query(RLReward).filter(RLReward.user_id == user_id)
    if start_date:
        try:
            query = query.filter(RLReward.date >= date_cls.fromisoformat(start_date))
        except ValueError:
            raise HTTPException(422, "start_date must be YYYY-MM-DD")
    if end_date:
        try:
            query = query.filter(RLReward.date <= date_cls.fromisoformat(end_date))
        except ValueError:
            raise HTTPException(422, "end_date must be YYYY-MM-DD")

    rows = query.order_by(RLReward.date.asc()).all()
    return [RewardResponse.model_validate(r) for r in rows]


@router.post("/recommend", response_model=RLRecommendResponse)
def recommend(
    payload: RLRecommendRequest, db: Session = Depends(get_db)
) -> RLRecommendResponse:
    """RL-boosted recommendations: pick a macro strategy, then fit foods to it.

    Builds today's context, selects an action with the LinUCB bandit
    (epsilon-greedy; fully exploratory on cold start), maps the action to macro
    targets, logs the selection, and returns the top-3 foods for that strategy.
    """
    user = db.get(User, payload.user_id)
    if user is None:
        raise HTTPException(404, "User not found")

    day = date_cls.today()
    context = rl_reward.build_context(db, user.id, day)
    eaten = rl_reward._sum_day_macros(db, user.id, day)
    meal_type = payload.meal_type or detect_meal_type("", datetime.now().hour)

    # --- Fallback decision: use RL, or bypass to the default optimizer? ---
    use_rl, fallback_reason, _ = rl_eval.resolve_use_rl(db, user.id, days=14)
    if not use_rl:
        db.add(RLFallbackLog(user_id=user.id, date=day, reason=fallback_reason))
        db.commit()
        try:
            foods = optimizer.get_top_recommendations(
                db=db, user=user, current_macros=eaten, meal_type=meal_type, limit=3
            )
        except Exception as exc:  # pragma: no cover - optimizer unavailable
            print(f"[rl] fallback: optimizer unavailable ({exc}); no foods.")
            foods = []
        targets = get_or_create_targets(db, user, day)
        return RLRecommendResponse(
            action_name="xgboost",
            recommended_macros={
                "protein_g": targets.protein_g,
                "carbs_g": targets.carbs_g,
                "fat_g": targets.fat_g,
                "calories": targets.calories,
            },
            foods=[RLRecommendedFood(**f) for f in foods],
            used_rl=False,
            fallback_reason=fallback_reason,
        )

    bandit = rl_bandit.get_bandit()
    action_index = bandit.select_action(context)
    action_name = rl_actions.ACTION_NAMES[action_index]

    target_macros = rl_actions.apply_action(db, action_name, user.id, day)

    # Rank foods by the strategy; degrade to empty list if the optimizer is down.
    try:
        foods = optimizer.get_top_recommendations(
            db=db,
            user=user,
            current_macros=eaten,
            meal_type=meal_type,
            limit=3,
            target_macros=target_macros,
        )
    except Exception as exc:  # e.g. XGBoost/OpenMP unavailable
        print(f"[rl] recommend: optimizer unavailable ({exc}); returning no foods.")
        foods = []

    # Log the action selection for later online training (Part 3).
    action_row = rl_actions.get_action_by_name(db, action_name)
    if action_row is not None:
        state_row = (
            db.query(RLState)
            .filter(RLState.user_id == user.id, RLState.date == day)
            .one_or_none()
        )
        db.add(
            RLActionLog(
                user_id=user.id,
                date=day,
                action_id=action_row.id,
                context_json=state_row.context_vector if state_row else {},
            )
        )
        db.commit()

    return RLRecommendResponse(
        action_name=action_name,
        recommended_macros=target_macros,
        foods=[RLRecommendedFood(**f) for f in foods],
        used_rl=True,
    )


@router.post("/update", response_model=RLUpdateResponse)
def update(payload: RLUpdateRequest, db: Session = Depends(get_db)) -> RLUpdateResponse:
    """Manually trigger an online bandit update for a user/day (idempotent)."""
    user = db.get(User, payload.user_id)
    if user is None:
        raise HTTPException(404, "User not found")
    day = payload.date or date_cls.today()
    result = rl_online.update_bandit_from_day(db, user.id, day)
    return RLUpdateResponse(**result)


@router.get("/status", response_model=RLStatusResponse)
def status() -> RLStatusResponse:
    """Report the current bandit state (update count, exploration rate, dims)."""
    bandit = rl_bandit.get_bandit()
    return RLStatusResponse(
        update_count=bandit.update_count,
        epsilon=round(bandit.epsilon(), 6),
        n_actions=bandit.n_actions,
        context_dim=bandit.context_dim,
        alpha=bandit.alpha,
        weights_persisted=settings.RL_BANDIT_PATH.exists(),
    )


@router.get("/evaluate/{user_id}", response_model=RLEvalResponse)
def evaluate(
    user_id: int,
    days: int = Query(default=14, ge=1, le=365),
    db: Session = Depends(get_db),
) -> RLEvalResponse:
    """RL-vs-XGBoost reward comparison over the last ``days`` days."""
    user = db.get(User, user_id)
    if user is None:
        raise HTTPException(404, "User not found")
    return RLEvalResponse(**rl_eval.evaluate_rl_vs_xgboost(db, user_id, days=days))


@router.post("/switch", response_model=RLSwitchResponse)
def switch(
    payload: RLSwitchRequest, db: Session = Depends(get_db)
) -> RLSwitchResponse:
    """Manually force RL on/off for a user (admin/debug), upserting the override."""
    user = db.get(User, payload.user_id)
    if user is None:
        raise HTTPException(404, "User not found")

    override = (
        db.query(RLOverride).filter(RLOverride.user_id == payload.user_id).one_or_none()
    )
    if override is None:
        override = RLOverride(user_id=payload.user_id)
        db.add(override)
    override.forced_rl = payload.force_rl
    db.commit()
    db.refresh(override)
    return RLSwitchResponse(user_id=override.user_id, forced_rl=override.forced_rl)
