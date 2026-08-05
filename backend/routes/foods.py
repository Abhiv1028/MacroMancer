"""Food endpoints: create custom foods and search USDA + custom foods."""

from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import or_
from sqlalchemy.orm import Session

from backend.db import get_db
from backend.models import Food, Nutrient, Portion
from backend.schemas import FoodCreate, FoodOut, FoodSearchResult
from backend.services import cache, food_utils

router = APIRouter(prefix="/foods", tags=["foods"])


@router.post("", response_model=FoodOut, status_code=201)
def create_food(payload: FoodCreate, db: Session = Depends(get_db)) -> FoodOut:
    """Add a custom food defined by its per-100g macros and optional portions."""
    calories = payload.calories_per_100g
    if calories is None:
        calories = (
            payload.protein_per_100g * 4
            + payload.carbs_per_100g * 4
            + payload.fat_per_100g * 9
        )

    food = Food(
        usda_food_id=None,
        name=payload.name,
        category=payload.category,
        default_grams=payload.default_grams,
        is_custom=True,
    )
    food.nutrients.extend(
        [
            Nutrient(nutrient_name="Protein", amount_per_100g=payload.protein_per_100g, unit="g"),
            Nutrient(nutrient_name="Carbohydrate", amount_per_100g=payload.carbs_per_100g, unit="g"),
            Nutrient(nutrient_name="Fat", amount_per_100g=payload.fat_per_100g, unit="g"),
            Nutrient(nutrient_name="Energy", amount_per_100g=calories, unit="kcal"),
        ]
    )
    if payload.portions:
        for p in payload.portions:
            food.portions.append(
                Portion(description=p.description, gram_weight=p.gram_weight)
            )

    db.add(food)
    db.commit()
    db.refresh(food)
    # New/changed foods invalidate cached search and fuzzy-match results.
    cache.clear_food_search()
    cache.clear_food_match()
    return FoodOut.model_validate(food)


@router.get("", response_model=List[FoodSearchResult])
def search_foods(
    search: str = Query(default="", description="Case-insensitive name substring."),
    skip: int = Query(default=0, ge=0, description="Number of results to skip."),
    limit: int = Query(default=25, ge=1, le=200, description="Max results to return."),
    db: Session = Depends(get_db),
) -> List[FoodSearchResult]:
    """Search across USDA and custom foods by name substring, with pagination.

    Results are cached for 5 minutes (per search/skip/limit); the cache is
    cleared whenever a food is created.
    """
    key = cache.food_search_key(search, skip, limit)
    cached = cache.get_food_search(key)
    if cached is not None:
        return [FoodSearchResult(**r) for r in cached]

    query = db.query(Food)
    if search:
        pattern = f"%{search.strip()}%"
        query = query.filter(or_(Food.name.ilike(pattern), Food.category.ilike(pattern)))
    foods = (
        query.order_by(Food.is_custom.desc(), Food.name.asc())
        .offset(skip)
        .limit(limit)
        .all()
    )

    results: List[FoodSearchResult] = []
    for food in foods:
        m = food_utils.macros_per_100g(food)
        results.append(
            FoodSearchResult(
                id=food.id,
                name=food.name,
                category=food.category,
                is_custom=food.is_custom,
                protein_per_100g=round(m["protein_g"], 2),
                carbs_per_100g=round(m["carbs_g"], 2),
                fat_per_100g=round(m["fat_g"], 2),
                calories_per_100g=round(m["calories"], 2),
            )
        )
    cache.set_food_search(key, [r.model_dump() for r in results])
    return results


@router.get("/{food_id}", response_model=FoodOut)
def get_food(food_id: int, db: Session = Depends(get_db)) -> FoodOut:
    """Fetch a single food with its full nutrient and portion detail."""
    food = db.get(Food, food_id)
    if food is None:
        raise HTTPException(404, "Food not found")
    return FoodOut.model_validate(food)
