"""Phase 5 Restaurant Mode endpoints (mounted under /api/v1/restaurant).

Flow: photograph a receipt/menu -> OCR -> parse dishes -> fuzzy-match to foods
(provisional, no logging) -> user confirms -> log_meal persists a RestaurantLog
and mirrors each item into MealLog so daily macros stay accurate.
"""

from __future__ import annotations

import asyncio
from datetime import date as date_cls, datetime
from typing import List

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)
from sqlalchemy.orm import Session

from backend.db import get_db, session_scope
from backend.models import Food, MealLog, RestaurantItem, RestaurantLog, User
from backend.rate_limit import limiter
from backend.schemas import (
    FoodSearchResult,
    MatchedFood,
    ParsedReceiptItem,
    ReceiptUploadResponse,
    RestaurantLogCreate,
    RestaurantLogResponse,
    SubstituteRequest,
    SubstituteResponse,
)
from backend.services import cache, food_utils
from backend.services.food_matcher import get_food_choices, match_dishes
from backend.services.menu_parser import parse_menu_items
from backend.services.meal_type_detector import detect_meal_type
from backend.services.ocr_service import (
    InvalidImageError,
    OCRUnavailableError,
    extract_text_from_image,
)
from backend.services.substitution_suggester import suggest_substitutions

router = APIRouter(prefix="/restaurant", tags=["restaurant"])


def _summary_task(user_id: int, day) -> None:
    """Background task: refresh the user's DailySummary for ``day``."""
    from backend.services.daily_summary import update_daily_summary

    try:
        with session_scope() as db:
            update_daily_summary(db, user_id, day)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[restaurant] Daily-summary update failed: {exc}")


@router.post("/upload_receipt", response_model=ReceiptUploadResponse)
@limiter.limit("10/minute")
async def upload_receipt(
    request: Request,
    user_id: int = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> ReceiptUploadResponse:
    """OCR a receipt/menu image, parse dishes, and fuzzy-match to foods.

    Provisional only -- nothing is logged. Rate limited to 10/min (OCR is slow).
    Returns 503 with install instructions if OCR is unavailable, 400 for a
    non-image upload.
    """
    user = db.get(User, user_id)
    if user is None:
        raise HTTPException(404, "User not found")

    image_bytes = await file.read()
    try:
        receipt_text = await extract_text_from_image(image_bytes)
    except OCRUnavailableError as exc:
        raise HTTPException(503, str(exc))
    except InvalidImageError as exc:
        raise HTTPException(400, str(exc))

    dishes = parse_menu_items(receipt_text)
    choices = get_food_choices(db)
    names = [d["name"] for d in dishes]
    # Fuzzy matching is CPU-bound: run it off the event loop.
    matches = await asyncio.to_thread(match_dishes, names, choices)

    parsed_items: List[ParsedReceiptItem] = []
    for dish, (food_id, score) in zip(dishes, matches):
        matched_food = None
        suggested_grams = None
        warning = None
        if food_id is not None:
            food = db.get(Food, food_id)
            if food is not None:
                matched_food = MatchedFood(id=food.id, name=food.name)
                suggested_grams = food.default_grams
        if matched_food is None:
            warning = "No confident food match; log manually or pick a food."
        parsed_items.append(
            ParsedReceiptItem(
                dish_name=dish["name"],
                price=dish.get("price"),
                matched_food=matched_food,
                confidence=round(score, 1),
                suggested_grams=suggested_grams,
                warning=warning,
            )
        )

    return ReceiptUploadResponse(receipt_text=receipt_text, parsed_items=parsed_items)


@router.post("/log_meal", response_model=RestaurantLogResponse, status_code=201)
def log_restaurant_meal(
    payload: RestaurantLogCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> RestaurantLogResponse:
    """Persist a restaurant log and mirror each item into MealLog for tracking."""
    user = db.get(User, payload.user_id)
    if user is None:
        raise HTTPException(404, "User not found")

    entry_date = payload.date or date_cls.today()

    if payload.restaurant_log_id is not None:
        rlog = db.get(RestaurantLog, payload.restaurant_log_id)
        if rlog is None or rlog.user_id != user.id:
            raise HTTPException(404, "Restaurant log not found for this user")
        if payload.restaurant_name is not None:
            rlog.restaurant_name = payload.restaurant_name
        if payload.notes is not None:
            rlog.notes = payload.notes
    else:
        rlog = RestaurantLog(
            user_id=user.id,
            date=entry_date,
            restaurant_name=payload.restaurant_name,
            notes=payload.notes,
        )
        db.add(rlog)
        db.flush()

    now = datetime.now()
    meal_type = detect_meal_type("", now.hour)
    for item in payload.items:
        food = db.get(Food, item.food_id)
        if food is None:
            raise HTTPException(404, f"Food {item.food_id} not found")
        macros = food_utils.macros_for_grams(food, item.grams)

        # Mirror into MealLog so the day's macros include the restaurant meal.
        db.add(
            MealLog(
                user_id=user.id,
                food_id=food.id,
                grams_consumed=item.grams,
                meal_type=meal_type,
                timestamp=now,
                protein_g=macros["protein_g"],
                carbs_g=macros["carbs_g"],
                fat_g=macros["fat_g"],
                calories=macros["calories"],
            )
        )
        rlog.items.append(
            RestaurantItem(
                food_name=item.food_name or food.name,
                matched_food_id=food.id,
                serving_grams=item.grams,
                calories_estimate=macros["calories"],
                is_substituted=item.is_substituted,
            )
        )

    db.commit()
    db.refresh(rlog)

    cache.invalidate_user_optimize(user.id)
    background_tasks.add_task(_summary_task, user.id, entry_date)
    return RestaurantLogResponse.model_validate(rlog)


@router.get("/logs/{user_id}", response_model=List[RestaurantLogResponse])
def list_restaurant_logs(
    user_id: int, db: Session = Depends(get_db)
) -> List[RestaurantLogResponse]:
    """List all restaurant logs for a user (most recent first)."""
    user = db.get(User, user_id)
    if user is None:
        raise HTTPException(404, "User not found")
    logs = (
        db.query(RestaurantLog)
        .filter(RestaurantLog.user_id == user_id)
        .order_by(RestaurantLog.date.desc(), RestaurantLog.id.desc())
        .all()
    )
    return [RestaurantLogResponse.model_validate(log) for log in logs]


@router.post("/substitute", response_model=SubstituteResponse)
def substitute(
    payload: SubstituteRequest, db: Session = Depends(get_db)
) -> SubstituteResponse:
    """Suggest healthier substitutions for a food."""
    user = db.get(User, payload.user_id)
    if user is None:
        raise HTTPException(404, "User not found")
    food = db.get(Food, payload.food_id)
    if food is None:
        raise HTTPException(404, "Food not found")

    subs = suggest_substitutions(db, food, payload.user_id, limit=payload.limit)

    results: List[FoodSearchResult] = []
    for f in subs:
        m = food_utils.macros_per_100g(f)
        results.append(
            FoodSearchResult(
                id=f.id,
                name=f.name,
                category=f.category,
                is_custom=f.is_custom,
                protein_per_100g=round(m["protein_g"], 2),
                carbs_per_100g=round(m["carbs_g"], 2),
                fat_per_100g=round(m["fat_g"], 2),
                calories_per_100g=round(m["calories"], 2),
            )
        )
    return SubstituteResponse(original_food_id=food.id, substitutions=results)
