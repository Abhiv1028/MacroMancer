"""Macromancer FastAPI application factory and lifespan wiring."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from sqlalchemy import func

from backend.config import settings
from backend.db import init_db, session_scope
from backend.models import Food
from backend.rate_limit import limiter
from backend.routes import (
    adaptation,
    chat,
    foods,
    grocery,
    meals,
    nearby,
    optimize,
    restaurant,
    rl,
    users,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the DB and ensure a model exists on startup."""
    init_db()

    with session_scope() as db:
        food_count = db.query(func.count(Food.id)).scalar() or 0
        # Seed the fixed RL action space (idempotent).
        from backend.services.rl_actions import seed_actions

        seeded = seed_actions(db)
        if seeded:
            print(f"[startup] Seeded {seeded} RL actions.")
    print(f"[startup] Database ready. Foods loaded: {food_count}")
    if food_count == 0:
        print(
            "[startup] No foods found. Run `python -m data.download_usda` to load "
            "USDA foods, or add custom foods via POST /foods."
        )

    # Train the model on synthetic data if none exists yet (non-fatal on error).
    if settings.SKIP_MODEL_INIT:
        print("[startup] MACROMANCER_SKIP_MODEL_INIT set; skipping model init.")
    else:
        try:
            from backend.services import optimizer

            if not settings.MODEL_PATH.exists():
                print("[startup] No model found; training on synthetic data ...")
                optimizer.train_on_synthetic()
            else:
                optimizer.load_model()
                print(f"[startup] Loaded model from {settings.MODEL_PATH}")
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[startup] Model init skipped due to error: {exc}")

    # Phase 6: load persisted RL bandit weights (or start fresh).
    try:
        from backend.services.rl_bandit import get_bandit

        bandit = get_bandit()
        print(f"[startup] RL bandit ready ({bandit.update_count} updates).")
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[startup] RL bandit init skipped due to error: {exc}")

    yield


app = FastAPI(
    title="Macromancer",
    version="1.0.0",
    description="Personal AI-driven nutrition optimizer (Phases 1-4).",
    lifespan=lifespan,
)

# Rate limiting (slowapi). Disable with DISABLE_RATELIMIT=1.
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.include_router(users.router)
app.include_router(foods.router)
app.include_router(meals.router)
app.include_router(optimize.router)
# Phase 2: conversational meal planning, namespaced under /api/v1.
app.include_router(chat.router, prefix="/api/v1")
# Phase 3: feedback, body composition, adaptive TDEE, goals.
app.include_router(adaptation.router, prefix="/api/v1")
# Phase 4: grocery lists / meal-prep planner.
app.include_router(grocery.router, prefix="/api/v1")
# Phase 5: restaurant mode (OCR receipts, matching, substitutions).
app.include_router(restaurant.router, prefix="/api/v1")
# Phase 6: reinforcement learning (reward + context, Part 1).
app.include_router(rl.router, prefix="/api/v1")
# Phase 7: nearby restaurants (OSM + Nutritionix).
app.include_router(nearby.router, prefix="/api/v1")


@app.get("/", tags=["health"])
def root() -> dict:
    """Health/info endpoint."""
    return {
        "name": "Macromancer",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "ok",
    }


@app.get("/health", tags=["health"])
def health() -> dict:
    """Liveness probe."""
    return {"status": "ok"}
