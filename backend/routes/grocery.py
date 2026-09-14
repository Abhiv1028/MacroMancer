"""Phase 4 grocery-list / meal-prep endpoints (mounted under /api/v1)."""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.db import get_db
from backend.models import Conversation, ConversationSession, GroceryList, GroceryItem, User
from backend.schemas import (
    GroceryListSummary,
    GroceryItemResponse,
    GroceryListCreate,
    GroceryListDetailResponse,
    GroceryListResponse,
)
from backend.services.grocery_generator import generate_grocery_items

router = APIRouter(tags=["grocery"])


def extract_foods_from_meal_plan(meal_plan: Optional[dict]) -> List[dict]:
    """Pull the ``foods`` list out of a stored meal-plan JSON blob (safely)."""
    if not isinstance(meal_plan, dict):
        return []
    foods = meal_plan.get("foods")
    if not isinstance(foods, list):
        return []
    return [f for f in foods if isinstance(f, dict) and f.get("name")]


def latest_session_meal_plan_foods(db: Session, session_id: int) -> List[dict]:
    """Return foods from the most recent assistant meal plan in a chat session."""
    convo = (
        db.query(Conversation)
        .filter(
            Conversation.session_id == session_id,
            Conversation.role == "assistant",
            Conversation.meal_plan.isnot(None),
        )
        .order_by(Conversation.created_at.desc(), Conversation.id.desc())
        .first()
    )
    if convo is None:
        return []
    return extract_foods_from_meal_plan(convo.meal_plan)


def create_grocery_list(
    db: Session,
    user: User,
    foods: List[dict],
    name: Optional[str] = None,
    session_id: Optional[int] = None,
    meal_prep_days: int = 1,
) -> GroceryList:
    """Build and persist a grocery list from meal-plan foods (single transaction).

    Empty ``foods`` yields a list with no items rather than an error.
    """
    grocery_list = GroceryList(
        user_id=user.id,
        session_id=session_id,
        name=name or "Grocery List",
    )
    for item in generate_grocery_items(foods, meal_prep_days=meal_prep_days):
        grocery_list.items.append(item)

    db.add(grocery_list)
    db.commit()
    db.refresh(grocery_list)
    return grocery_list


@router.post("/grocery_lists", response_model=GroceryListResponse, status_code=201)
def create_list(
    payload: GroceryListCreate, db: Session = Depends(get_db)
) -> GroceryListResponse:
    """Create a grocery list from an explicit meal plan or a chat session."""
    user = db.get(User, payload.user_id)
    if user is None:
        raise HTTPException(404, "User not found")

    session_id: Optional[int] = None
    if payload.meal_plan is not None:
        foods = [item.model_dump() for item in payload.meal_plan]
    elif payload.session_id is not None:
        session = db.get(ConversationSession, payload.session_id)
        if session is None or session.user_id != user.id:
            raise HTTPException(404, "Conversation session not found for this user")
        session_id = session.id
        foods = latest_session_meal_plan_foods(db, session.id)
    else:
        raise HTTPException(422, "Provide either meal_plan or session_id")

    grocery_list = create_grocery_list(
        db,
        user,
        foods,
        name=payload.name,
        session_id=session_id,
        meal_prep_days=payload.meal_prep_days,
    )
    return GroceryListResponse(
        grocery_list_id=grocery_list.id,
        name=grocery_list.name,
        items=[GroceryItemResponse.model_validate(i) for i in grocery_list.items],
    )


@router.get("/grocery_lists", response_model=List[GroceryListSummary])
def list_grocery_lists(
    user_id: int = Query(...), db: Session = Depends(get_db)
) -> List[GroceryListSummary]:
    """List a user's grocery lists (most recent first) with item counts."""
    user = db.get(User, user_id)
    if user is None:
        raise HTTPException(404, "User not found")
    rows = (
        db.query(
            GroceryList.id, GroceryList.name, GroceryList.created_at,
            func.count(GroceryItem.id).label("item_count"),
        )
        .outerjoin(GroceryItem, GroceryItem.grocery_list_id == GroceryList.id)
        .filter(GroceryList.user_id == user_id)
        .group_by(GroceryList.id)
        .order_by(GroceryList.created_at.desc())
        .all()
    )
    return [
        GroceryListSummary(id=r.id, name=r.name, created_at=r.created_at, item_count=r.item_count)
        for r in rows
    ]


@router.get(
    "/grocery_lists/{grocery_list_id}", response_model=GroceryListDetailResponse
)
def get_list(
    grocery_list_id: int, db: Session = Depends(get_db)
) -> GroceryListDetailResponse:
    """Return a grocery list and all of its items."""
    grocery_list = db.get(GroceryList, grocery_list_id)
    if grocery_list is None:
        raise HTTPException(404, "Grocery list not found")
    return GroceryListDetailResponse.model_validate(grocery_list)


@router.put("/grocery_items/{item_id}", response_model=GroceryItemResponse)
def toggle_item(item_id: int, db: Session = Depends(get_db)) -> GroceryItemResponse:
    """Toggle an item's ``checked`` state."""
    item = db.get(GroceryItem, item_id)
    if item is None:
        raise HTTPException(404, "Grocery item not found")
    item.checked = not item.checked
    db.commit()
    db.refresh(item)
    return GroceryItemResponse.model_validate(item)


@router.delete("/grocery_lists/{grocery_list_id}")
def delete_list(grocery_list_id: int, db: Session = Depends(get_db)) -> dict:
    """Delete a grocery list and its items (cascade)."""
    grocery_list = db.get(GroceryList, grocery_list_id)
    if grocery_list is None:
        raise HTTPException(404, "Grocery list not found")
    db.delete(grocery_list)
    db.commit()
    return {"status": "deleted", "grocery_list_id": grocery_list_id}
