"""Phase 7 Nearby Restaurants endpoints (free OSM + Nutritionix stack).

``GET /api/v1/nearby/search`` finds nearby restaurants, pulls their menus, and
ranks items against the user's remaining macros (RL-biased when active).
``POST /api/v1/nearby/log`` logs a chosen menu item as a meal.
"""

from __future__ import annotations

from datetime import date as date_cls, datetime, timedelta
from typing import Dict, List, Optional

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
    Query,
    Request,
)
from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.config import settings
from backend.db import get_db, session_scope
from backend.models import (
    CachedMenuItem,
    Food,
    MealLog,
    NearbySearchCache,
    Nutrient,
    RestaurantCache,
    RestaurantMealLog,
    RLActionLog,
    RLState,
    User,
)
from backend.rate_limit import limiter
from backend.schemas import (
    NearbyLogRequest,
    NearbyLogResponse,
    NearbyMenuItem,
    NearbyRestaurant,
    NearbySearchResponse,
)
from backend.services import (
    cache,
    food_utils,
    geo_service,
    nearby_optimizer,
    nutritionix_service,
    osm_service,
    rl_actions,
    rl_bandit,
    rl_eval,
    rl_online,
    rl_reward,
)
from backend.services.macro_calculator import get_or_create_targets
from backend.services.meal_type_detector import detect_meal_type

router = APIRouter(prefix="/nearby", tags=["nearby"])


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _remaining_macros(db: Session, user: User, day: date_cls) -> Dict[str, float]:
    """Macros still available today = daily target - eaten (floored at 0)."""
    targets = get_or_create_targets(db, user, day)
    eaten = rl_reward._sum_day_macros(db, user.id, day)
    return {
        "protein_g": max(targets.protein_g - eaten["protein_g"], 0.0),
        "carbs_g": max(targets.carbs_g - eaten["carbs_g"], 0.0),
        "fat_g": max(targets.fat_g - eaten["fat_g"], 0.0),
    }


def _rl_target(
    db: Session, user: User, day: date_cls, remaining: Dict[str, float]
) -> tuple:
    """Pick the scoring target: an RL strategy's macros, or plain remaining.

    Returns ``(target_macros, action_name, used_rl)``. Fully defensive -- any RL
    error falls back to the remaining macros with ``used_rl=False``.
    """
    try:
        use_rl, _reason, _ = rl_eval.resolve_use_rl(db, user.id)
        if not use_rl:
            return remaining, None, False
        context = rl_reward.build_context(db, user.id, day)
        bandit = rl_bandit.get_bandit()
        action_name = rl_actions.ACTION_NAMES[bandit.select_action(context)]
        strat = rl_actions.apply_action(db, action_name, user.id, day)
        # Log the selection (for online learning / evaluation).
        action_row = rl_actions.get_action_by_name(db, action_name)
        if action_row is not None:
            state = (
                db.query(RLState)
                .filter(RLState.user_id == user.id, RLState.date == day)
                .one_or_none()
            )
            db.add(
                RLActionLog(
                    user_id=user.id,
                    date=day,
                    action_id=action_row.id,
                    context_json=state.context_vector if state else {},
                )
            )
            db.commit()
        target = {
            "protein_g": strat["protein_g"],
            "carbs_g": strat["carbs_g"],
            "fat_g": strat["fat_g"],
        }
        return target, action_name, True
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[nearby] RL strategy unavailable ({exc}); using remaining macros.")
        return remaining, None, False


async def _get_or_fetch_menu(db: Session, restaurant: Dict) -> List[Dict]:
    """Return a restaurant's menu items, using RestaurantCache (7-day TTL)."""
    osm_id = restaurant["osm_id"]
    cutoff = datetime.now() - timedelta(seconds=settings.RESTAURANT_MENU_TTL)
    row = (
        db.query(RestaurantCache).filter(RestaurantCache.osm_id == osm_id).one_or_none()
    )
    if row is not None and row.last_fetched >= cutoff and row.menu_items is not None:
        return list(row.menu_items)

    items = await nutritionix_service.fetch_restaurant_menu(restaurant["name"])

    if row is None:
        row = RestaurantCache(osm_id=osm_id, name=restaurant["name"])
        db.add(row)
    row.name = restaurant["name"]
    row.address = restaurant.get("address")
    row.lat = restaurant.get("lat")
    row.lng = restaurant.get("lon")
    row.cuisine = restaurant.get("cuisine")
    row.menu_items = items
    row.last_fetched = datetime.now()
    # Replace structured menu-item rows.
    for existing in list(row.items):
        db.delete(existing)
    for it in items:
        row.items.append(
            CachedMenuItem(
                name=it["name"],
                calories=it.get("calories", 0.0),
                protein_g=it.get("protein_g", 0.0),
                carbs_g=it.get("carbs_g", 0.0),
                fat_g=it.get("fat_g", 0.0),
                serving_size_g=it.get("serving_size_g", 100.0),
            )
        )
    db.commit()
    return items


def _reward_task(user_id: int, day: date_cls, day_complete: bool) -> None:
    """Background: refresh reward/summary after a restaurant meal is logged."""
    try:
        with session_scope() as inner:
            from backend.services.daily_summary import update_daily_summary

            update_daily_summary(inner, user_id, day)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[nearby] summary update failed: {exc}")
    rl_online.reward_and_update_task(user_id, day, day_complete)


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@router.get("/search", response_model=NearbySearchResponse)
@limiter.limit("5/minute")
async def search(
    request: Request,
    user_id: int = Query(...),
    lat: Optional[float] = Query(default=None),
    lon: Optional[float] = Query(default=None),
    radius: Optional[int] = Query(default=None, ge=100, le=20000),
    db: Session = Depends(get_db),
) -> NearbySearchResponse:
    """Find nearby restaurants and rank menu items against remaining macros.

    Uses IP geolocation when lat/lon are omitted. Returns 503 if Nutritionix is
    not configured. Rate limited to 5/min per IP (OSM fair-use).
    """
    user = db.get(User, user_id)
    if user is None:
        raise HTTPException(404, "User not found")
    if not nutritionix_service.is_configured():
        raise HTTPException(503, nutritionix_service.SETUP_HELP)

    radius_m = radius or settings.NEARBY_SEARCH_RADIUS_DEFAULT

    # Resolve location.
    if lat is not None and lon is not None:
        location_label = f"{round(lat, 4)}, {round(lon, 4)}"
    else:
        client_ip = request.client.host if request.client else None
        loc = await geo_service.get_location_from_ip(client_ip)
        lat, lon = loc["lat"], loc["lon"]
        location_label = geo_service.location_label(loc)

    lat_r, lon_r = round(lat, 4), round(lon, 4)

    # Serve from the nearby-search cache when fresh.
    cutoff = datetime.now() - timedelta(seconds=settings.NEARBY_SEARCH_TTL)
    cached = (
        db.query(NearbySearchCache)
        .filter(
            NearbySearchCache.user_id == user_id,
            NearbySearchCache.latitude == lat_r,
            NearbySearchCache.longitude == lon_r,
            NearbySearchCache.radius_m == radius_m,
            NearbySearchCache.created_at >= cutoff,
        )
        .order_by(NearbySearchCache.created_at.desc())
        .first()
    )
    if cached is not None:
        return NearbySearchResponse(**cached.results)

    restaurants = await osm_service.search_nearby_restaurants(lat, lon, radius_m)
    restaurants = restaurants[: settings.NEARBY_MAX_RESTAURANTS]

    day = date_cls.today()
    remaining = _remaining_macros(db, user, day)
    target_macros, action_name, used_rl = _rl_target(db, user, day, remaining)

    out_restaurants: List[NearbyRestaurant] = []
    for r in restaurants:
        items = await _get_or_fetch_menu(db, r)
        if not items:
            continue
        scored = nearby_optimizer.score_nearby_items(
            db, items, target_macros, user_id
        )[: settings.NEARBY_MAX_ITEMS_PER_RESTAURANT]
        out_restaurants.append(
            NearbyRestaurant(
                name=r["name"],
                address=r.get("address"),
                cuisine=r.get("cuisine"),
                menu_items=[NearbyMenuItem(**it) for it in scored],
            )
        )

    response = NearbySearchResponse(
        location=location_label,
        used_rl=used_rl,
        action_name=action_name,
        restaurants=out_restaurants,
    )
    db.add(
        NearbySearchCache(
            user_id=user_id,
            latitude=lat_r,
            longitude=lon_r,
            radius_m=radius_m,
            results=response.model_dump(),
            # Use the app clock so the TTL check matches (avoid UTC/local skew).
            created_at=datetime.now(),
        )
    )
    db.commit()
    return response


@router.post("/log", response_model=NearbyLogResponse, status_code=201)
def log_nearby_meal(
    payload: NearbyLogRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> NearbyLogResponse:
    """Log a nearby-restaurant menu item as a meal (macros from the cached item)."""
    user = db.get(User, payload.user_id)
    if user is None:
        raise HTTPException(404, "User not found")

    # Look up the cached menu item's per-serving macros.
    item = (
        db.query(CachedMenuItem)
        .join(RestaurantCache, CachedMenuItem.restaurant_cache_id == RestaurantCache.id)
        .filter(
            func.lower(RestaurantCache.name) == payload.restaurant_name.lower(),
            func.lower(CachedMenuItem.name) == payload.item_name.lower(),
        )
        .order_by(CachedMenuItem.id.desc())
        .first()
    )
    if item is None:
        raise HTTPException(
            404,
            "Menu item not found in cache. Run GET /api/v1/nearby/search first.",
        )

    # Get-or-create a custom Food for this menu item (per-100g nutrients).
    food_name = f"{payload.restaurant_name} - {payload.item_name}"
    food = db.query(Food).filter(Food.name == food_name).first()
    if food is None:
        serving = item.serving_size_g or 100.0
        factor = 100.0 / serving
        food = Food(name=food_name, category="restaurant", default_grams=serving, is_custom=True)
        food.nutrients.extend(
            [
                Nutrient(nutrient_name="Protein", amount_per_100g=item.protein_g * factor, unit="g"),
                Nutrient(nutrient_name="Carbohydrate", amount_per_100g=item.carbs_g * factor, unit="g"),
                Nutrient(nutrient_name="Fat", amount_per_100g=item.fat_g * factor, unit="g"),
                Nutrient(nutrient_name="Energy", amount_per_100g=item.calories * factor, unit="kcal"),
            ]
        )
        db.add(food)
        db.flush()

    macros = food_utils.macros_for_grams(food, payload.grams)
    now = datetime.now()
    meal_type = detect_meal_type("", now.hour)
    meal = MealLog(
        user_id=user.id,
        food_id=food.id,
        grams_consumed=payload.grams,
        meal_type=meal_type,
        timestamp=now,
        protein_g=macros["protein_g"],
        carbs_g=macros["carbs_g"],
        fat_g=macros["fat_g"],
        calories=macros["calories"],
    )
    db.add(meal)
    db.flush()

    rlog = RestaurantMealLog(
        user_id=user.id,
        restaurant_name=payload.restaurant_name,
        item_name=payload.item_name,
        grams=payload.grams,
        meal_log_id=meal.id,
    )
    db.add(rlog)
    db.commit()
    db.refresh(rlog)
    db.refresh(meal)

    cache.invalidate_user_optimize(user.id)
    day = now.date()
    day_complete = meal_type == "dinner" or now.hour >= 20
    background_tasks.add_task(_reward_task, user.id, day, day_complete)

    return NearbyLogResponse(
        restaurant_meal_log_id=rlog.id,
        meal_log_id=meal.id,
        macros=macros,
        feedback_prompt=(
            f"How was the {payload.item_name}? Rate it via "
            f"POST /api/v1/feedback (meal_log_id={meal.id})."
        ),
    )
