"""Optimize + train endpoints backed by the XGBoost optimizer."""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from backend.db import get_db, session_scope
from backend.models import Food, User
from backend.rate_limit import limiter
from backend.schemas import (
    OptimizeRequest,
    OptimizeResponse,
    Recommendation,
    TrainResponse,
)
from backend.services import cache, optimizer

router = APIRouter(tags=["optimize"])


@router.post("/optimize", response_model=OptimizeResponse)
@limiter.limit("30/minute")
def optimize(
    request: Request, payload: OptimizeRequest, db: Session = Depends(get_db)
) -> OptimizeResponse:
    """Recommend the top foods for the user's remaining macro budget.

    Rate limited to 30/min per IP. Results are cached for 60s per user +
    rounded macros + meal_type + candidate set, so refresh-spamming is cheap.
    """
    user = db.get(User, payload.user_id)
    if user is None:
        raise HTTPException(404, "User not found")

    current = payload.current_macros.model_dump()
    cache_key = cache.optimize_key(
        payload.user_id, payload.meal_type, current, payload.available_food_ids
    )
    cached = cache.get_optimize(cache_key)
    if cached is not None:
        return OptimizeResponse(recommendations=[Recommendation(**r) for r in cached])

    if payload.available_food_ids:
        candidates = (
            db.query(Food).filter(Food.id.in_(payload.available_food_ids)).all()
        )
        if not candidates:
            raise HTTPException(404, "None of the available_food_ids were found")
    else:
        candidates = optimizer.default_candidate_foods(db)

    if not candidates:
        raise HTTPException(
            409, "No candidate foods available. Load USDA data or add custom foods."
        )

    recs = optimizer.score_candidates(
        db=db,
        user=user,
        current_macros=current,
        meal_type=payload.meal_type,
        candidate_foods=candidates,
    )
    cache.set_optimize(cache_key, recs)
    return OptimizeResponse(
        recommendations=[Recommendation(**r) for r in recs]
    )


def _retrain_job() -> None:
    """Background job: retrain from logs (own session) and refresh cache."""
    try:
        with session_scope() as db:
            optimizer.retrain_from_logs(db)
        optimizer.invalidate_cache()
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[train] Background retrain failed: {exc}")


@router.post("/train", response_model=TrainResponse)
def train(
    background_tasks: BackgroundTasks,
    run_async: bool = True,
    db: Session = Depends(get_db),
) -> TrainResponse:
    """Trigger model (re)training from existing MealLog data.

    By default the retrain runs as a background task; pass ``run_async=false``
    to train synchronously and block until done.
    """
    if run_async:
        background_tasks.add_task(_retrain_job)
        return TrainResponse(
            status="scheduled",
            detail="Retraining started in the background from existing logs.",
        )

    optimizer.retrain_from_logs(db)
    optimizer.invalidate_cache()
    return TrainResponse(status="completed", detail="Model retrained synchronously.")
