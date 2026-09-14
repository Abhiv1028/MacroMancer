"""Conversational meal-planning endpoint backed by a local Ollama LLM.

Flow for ``POST /chat``:
    1. Resolve the user and today's macro targets.
    2. Sum today's meal logs -> macros already eaten; derive remaining budget.
    3. Ask the Phase-1 optimizer for the top candidate foods.
    4. Prompt Ollama to assemble a meal plan using only those foods.
    5. Persist the turn (user + assistant) and return the plan.

If Ollama is unavailable the endpoint degrades to a plain-text suggestion built
from the optimizer's top recommendation (``meal_plan=None``).
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from backend.db import get_db
from backend.models import Conversation, ConversationSession, MealLog, User
from backend.rate_limit import limiter
from backend.schemas import ChatRequest, ChatResponse
from backend.services import food_utils, optimizer
from backend.services.macro_calculator import get_or_create_targets
from backend.services.meal_type_detector import detect_meal_type
from backend.services.ollama_client import LLMUnavailableError, generate_meal_plan

router = APIRouter(prefix="/chat", tags=["chat"])

_MACROS = ("protein_g", "carbs_g", "fat_g")
# Phrases that signal the user also wants a grocery/shopping list built.
_GROCERY_KEYWORDS = ("grocery list", "shopping list", "meal prep")


def _wants_grocery_list(message: str) -> bool:
    text = (message or "").lower()
    return any(kw in text for kw in _GROCERY_KEYWORDS)


def _sum_todays_macros(db: Session, user_id: int, day: date) -> Dict[str, float]:
    """Sum protein/carbs/fat/calories across a user's meal logs for ``day``."""
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
    totals = {"protein_g": 0.0, "carbs_g": 0.0, "fat_g": 0.0, "calories": 0.0}
    for log in logs:
        totals["protein_g"] += log.protein_g
        totals["carbs_g"] += log.carbs_g
        totals["fat_g"] += log.fat_g
        totals["calories"] += log.calories
    return totals


def _heuristic_recommendations(
    db: Session, remaining: Dict[str, float], limit: int
) -> List[dict]:
    """ML-free fallback ranking used when the optimizer/XGBoost is unavailable.

    Ranks candidate foods by how much a standard serving reduces the absolute
    remaining macro gap, so ``/chat`` still works without the OpenMP runtime.
    """
    recs = []
    for food in optimizer.default_candidate_foods(db):
        grams = optimizer._suggested_grams(food)
        serving = food_utils.macros_for_grams(food, grams)
        pre = sum(abs(remaining.get(m, 0.0)) for m in _MACROS)
        post = sum(
            abs(remaining.get(m, 0.0) - serving[m]) for m in _MACROS
        )
        recs.append(
            {
                "food_id": food.id,
                "name": food.name,
                "protein_g": serving["protein_g"],
                "carbs_g": serving["carbs_g"],
                "fat_g": serving["fat_g"],
                "calories": serving["calories"],
                "suggested_grams": grams,
                "predicted_score": round(max(pre - post, 0.0), 2),
            }
        )
    recs.sort(key=lambda r: r["predicted_score"], reverse=True)
    return recs[:limit]


def _gather_recommendations(
    db: Session,
    user: User,
    eaten: Dict[str, float],
    remaining: Dict[str, float],
    meal_type: str,
    limit: int,
) -> List[dict]:
    """Get optimizer recommendations, degrading to a heuristic on ML failure.

    ``eaten`` (consumed-so-far) is what the model expects for its ratios; the
    heuristic fallback uses ``remaining`` directly.
    """
    try:
        return optimizer.get_top_recommendations(
            db=db,
            user=user,
            current_macros=eaten,
            meal_type=meal_type,
            limit=limit,
        )
    except Exception as exc:  # e.g. missing OpenMP runtime for XGBoost
        print(f"[chat] Optimizer unavailable ({exc}); using heuristic fallback.")
        return _heuristic_recommendations(db, remaining, limit)


def _build_system_prompt(
    user: User, remaining: Dict[str, float], recommendations: List[dict], message: str
) -> str:
    """Assemble the system prompt instructing the LLM to build a meal plan."""
    food_lines = "\n".join(
        f"- {r['name']} (food_id {r['food_id']}, ~{r['suggested_grams']}g): "
        f"{r['protein_g']}g protein, {r['carbs_g']}g carbs, "
        f"{r['fat_g']}g fat, {r['calories']} kcal"
        for r in recommendations
    )
    return (
        "You are a nutrition assistant. "
        f"User stats: age={user.age}, weight={user.weight_kg}kg, goal={user.goal}.\n"
        f"Remaining macros for today: {remaining['protein_g']}g protein, "
        f"{remaining['carbs_g']}g carbs, {remaining['fat_g']}g fat, "
        f"{remaining['calories']} kcal.\n"
        "Available foods (with macros per suggested grams):\n"
        f"{food_lines}\n"
        f'Generate a meal plan for the user\'s request: "{message}".\n'
        "Use ONLY the available foods listed.\n"
        "Return valid JSON with fields: meal_name (string), "
        "foods (list of {food_id: int, name: string, grams: float}), "
        "total_macros (object with protein_g, carbs_g, fat_g, calories), "
        "explanation (string, max 2 sentences).\n"
        "Do not include any other text outside the JSON."
    )


def _resolve_session(
    db: Session, user: User, session_id: Optional[int], message: str
) -> ConversationSession:
    """Fetch an existing session (validating ownership) or create a new one."""
    if session_id is not None:
        session = db.get(ConversationSession, session_id)
        if session is None or session.user_id != user.id:
            raise HTTPException(404, "Conversation session not found for this user")
        return session

    session = ConversationSession(user_id=user.id, title=message[:50])
    db.add(session)
    db.flush()  # assign session.id for the FK on conversation rows
    return session


@router.post("", response_model=ChatResponse)
@limiter.limit("10/minute")
async def chat(
    request: Request, payload: ChatRequest, db: Session = Depends(get_db)
) -> ChatResponse:
    """Generate a meal plan for a natural-language request via the local LLM.

    Rate limited to 10/min per IP (Ollama is slow). The Ollama call is awaited
    via ``httpx.AsyncClient`` so it never blocks the event loop. If the message
    mentions a grocery/shopping/meal-prep list, one is generated and its id is
    returned in ``grocery_list_id``.
    """
    user = db.get(User, payload.user_id)
    if user is None:
        raise HTTPException(404, "User not found")

    today = date.today()
    targets = get_or_create_targets(db, user, today)
    eaten = _sum_todays_macros(db, user.id, today)

    remaining = {
        "protein_g": round(max(targets.protein_g - eaten["protein_g"], 0.0), 1),
        "carbs_g": round(max(targets.carbs_g - eaten["carbs_g"], 0.0), 1),
        "fat_g": round(max(targets.fat_g - eaten["fat_g"], 0.0), 1),
        "calories": round(max(targets.calories - eaten["calories"], 0.0), 1),
    }

    meal_type = detect_meal_type(payload.message, datetime.now().hour)
    recommendations = _gather_recommendations(
        db, user, eaten, remaining, meal_type, limit=5
    )

    # Persist the session and the user's turn up front.
    session = _resolve_session(db, user, payload.session_id, payload.message)
    db.add(
        Conversation(session_id=session.id, role="user", content=payload.message)
    )
    db.flush()

    # --- LLM call with graceful fallback ---------------------------------
    system_prompt = _build_system_prompt(user, remaining, recommendations, payload.message)
    try:
        meal_plan = await generate_meal_plan(system_prompt, payload.message)
        assistant_message = meal_plan.get("explanation") or json.dumps(meal_plan)
    except LLMUnavailableError as exc:
        print(f"[chat] {exc}")
        from backend.config import settings

        llm_name = "The AI model" if settings.LLM_HOSTED else "Ollama"
        if recommendations:
            top = recommendations[0]
            assistant_message = (
                f"{llm_name} is unavailable right now, but based on your remaining "
                f"macros the top pick is {top['name']} — about {top['suggested_grams']}g."
            )
        else:
            assistant_message = (
                f"{llm_name} is unavailable and no food suggestions are available. "
                "Add custom foods or load USDA data, then try again."
            )
        meal_plan = None

    db.add(
        Conversation(
            session_id=session.id,
            role="assistant",
            content=assistant_message,
            meal_plan=meal_plan,
        )
    )
    db.commit()

    # Optional grocery-list generation when the user asks for one.
    grocery_list_id = None
    if _wants_grocery_list(payload.message):
        grocery_list_id = _maybe_build_grocery_list(
            db, user, session.id, meal_plan, recommendations
        )

    return ChatResponse(
        message=assistant_message,
        meal_plan=meal_plan,
        session_id=session.id,
        grocery_list_id=grocery_list_id,
    )


def _maybe_build_grocery_list(
    db: Session,
    user: User,
    session_id: int,
    meal_plan: Optional[dict],
    recommendations: List[dict],
) -> Optional[int]:
    """Build a grocery list from the meal plan (or recommendations) foods.

    Returns the new list's id, or ``None`` if there were no foods to add.
    """
    from backend.routes.grocery import create_grocery_list, extract_foods_from_meal_plan

    foods = extract_foods_from_meal_plan(meal_plan)
    if not foods:
        # Fall back to the optimizer's suggested foods/servings.
        foods = [
            {"name": r["name"], "grams": r.get("suggested_grams", 0)}
            for r in recommendations
        ]
    if not foods:
        return None
    grocery_list = create_grocery_list(
        db, user, foods, name="Meal Prep List", session_id=session_id
    )
    return grocery_list.id
