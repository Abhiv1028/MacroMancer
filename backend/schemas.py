"""Pydantic v2 request/response models for the Macromancer API."""

from __future__ import annotations

from datetime import date as date_cls, datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

ActivityLevel = str  # sedentary | light | moderate | very | extra
Goal = str  # cut | maintain | bulk
MealType = str  # breakfast | lunch | dinner | snack


# --------------------------------------------------------------------------- #
# Users
# --------------------------------------------------------------------------- #
class UserCreate(BaseModel):
    age: int = Field(..., ge=10, le=120)
    weight_kg: float = Field(..., ge=20, le=400)
    height_cm: float = Field(..., ge=50, le=300)
    activity_level: ActivityLevel = Field(..., examples=["moderate"])
    goal: Goal = Field(..., examples=["cut"])
    sex: str = Field(default="female", examples=["female", "male"])
    dietary_restrictions: Optional[List[str]] = None


class MacroTargetOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    date: date_cls
    calories: float
    protein_g: float
    carbs_g: float
    fat_g: float


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    age: int
    weight_kg: float
    height_cm: float
    activity_level: str
    goal: str
    sex: str
    dietary_restrictions: Optional[List[str]] = None
    created_at: datetime


class UserCreateResponse(BaseModel):
    user: UserOut
    targets: MacroTargetOut


# --------------------------------------------------------------------------- #
# Foods
# --------------------------------------------------------------------------- #
class PortionIn(BaseModel):
    description: str = Field(..., min_length=1, max_length=255)
    gram_weight: float = Field(..., gt=0, le=10000)


class PortionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    description: str
    gram_weight: float


class NutrientOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    nutrient_name: str
    amount_per_100g: float
    unit: str


class FoodCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=500)
    protein_per_100g: float = Field(..., ge=0, le=1000)
    carbs_per_100g: float = Field(..., ge=0, le=1000)
    fat_per_100g: float = Field(..., ge=0, le=500)
    calories_per_100g: Optional[float] = Field(
        default=None, ge=0, le=10000, description="Computed from macros if omitted."
    )
    category: Optional[str] = None
    default_grams: float = Field(default=100.0, gt=0, le=10000)
    portions: Optional[List[PortionIn]] = None


class FoodOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    usda_food_id: Optional[int] = None
    name: str
    category: Optional[str] = None
    default_grams: float
    is_custom: bool
    nutrients: List[NutrientOut] = []
    portions: List[PortionOut] = []


class FoodSearchResult(BaseModel):
    id: int
    name: str
    category: Optional[str] = None
    is_custom: bool
    protein_per_100g: float
    carbs_per_100g: float
    fat_per_100g: float
    calories_per_100g: float


# --------------------------------------------------------------------------- #
# Meals
# --------------------------------------------------------------------------- #
class MealCreate(BaseModel):
    user_id: int
    food_id: int
    grams: float = Field(..., gt=0, le=10000)
    meal_type: MealType = Field(..., examples=["lunch"])
    portion_id: Optional[int] = None
    timestamp: Optional[datetime] = None


class MealOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    food_id: int
    portion_id: Optional[int] = None
    grams_consumed: float
    meal_type: str
    timestamp: datetime
    protein_g: float
    carbs_g: float
    fat_g: float
    calories: float


class MealHistoryItem(BaseModel):
    """A logged meal enriched with the food's name (for GET /meals)."""

    id: int
    food_id: int
    food_name: str
    grams_consumed: float
    meal_type: str
    timestamp: datetime
    protein_g: float
    carbs_g: float
    fat_g: float
    calories: float


# --------------------------------------------------------------------------- #
# Optimize
# --------------------------------------------------------------------------- #
class CurrentMacros(BaseModel):
    protein_g: float = Field(default=0.0, ge=0, le=1000)
    carbs_g: float = Field(default=0.0, ge=0, le=1000)
    fat_g: float = Field(default=0.0, ge=0, le=500)


class OptimizeRequest(BaseModel):
    user_id: int
    current_macros: CurrentMacros
    meal_type: MealType = Field(..., examples=["dinner"])
    available_food_ids: Optional[List[int]] = None


class Recommendation(BaseModel):
    food_id: int
    name: str
    predicted_score: float
    macros_per_serving: Dict[str, float]
    suggested_grams: float


class OptimizeResponse(BaseModel):
    recommendations: List[Recommendation]


# --------------------------------------------------------------------------- #
# Train
# --------------------------------------------------------------------------- #
class TrainResponse(BaseModel):
    status: str
    detail: str


# --------------------------------------------------------------------------- #
# Chat / LLM meal planning
# --------------------------------------------------------------------------- #
class FoodSuggestion(BaseModel):
    """A single food entry within a generated meal plan."""

    food_id: int
    name: str
    grams: float


class MealPlanResponse(BaseModel):
    """Structured meal plan returned by the LLM (best-effort validated)."""

    meal_name: str
    foods: List[FoodSuggestion] = []
    total_macros: Dict[str, float] = {}
    explanation: str = ""


class ChatRequest(BaseModel):
    user_id: int
    session_id: Optional[int] = None
    message: str = Field(..., min_length=1)


class ChatResponse(BaseModel):
    message: str
    meal_plan: Optional[Dict[str, Any]] = None
    session_id: int
    # Populated when the user's message asks for a grocery/shopping/meal-prep list.
    grocery_list_id: Optional[int] = None


# --------------------------------------------------------------------------- #
# Phase 3: feedback, body composition, adaptive TDEE
# --------------------------------------------------------------------------- #
class FeedbackCreate(BaseModel):
    user_id: int
    meal_log_id: Optional[int] = None
    enjoyment: int = Field(..., ge=1, le=5)
    satiety: int = Field(..., ge=1, le=5)
    energy: int = Field(..., ge=1, le=5)
    workout_performance: Optional[int] = Field(default=None, ge=1, le=5)
    notes: Optional[str] = None


class FeedbackResponse(BaseModel):
    status: str
    feedback_id: int


class BodyCompCreate(BaseModel):
    user_id: int
    date: Optional[date_cls] = None
    weight_kg: float = Field(..., ge=20, le=400)
    body_fat_percent: Optional[float] = Field(default=None, ge=0, le=100)
    lean_mass_kg: Optional[float] = Field(default=None, gt=0, le=400)


class BodyCompResponse(BaseModel):
    id: int
    user_id: int
    date: date_cls
    weight_kg: float
    body_fat_percent: Optional[float] = None
    lean_mass_kg: Optional[float] = None
    updated_targets: MacroTargetOut


class TDEEInfo(BaseModel):
    current_tdee: float
    adaptive_tdee: Optional[float] = None
    last_updated: Optional[date_cls] = None
    method_used: str


class GoalsUpdate(BaseModel):
    goal: str = Field(..., examples=["cut", "maintain", "bulk"])


class GoalsResponse(BaseModel):
    user_id: int
    goal: str
    method_used: str
    targets: MacroTargetOut


# --------------------------------------------------------------------------- #
# Phase 4: grocery lists
# --------------------------------------------------------------------------- #
class GroceryFoodItem(BaseModel):
    """A meal-plan food supplied directly in a grocery-list request."""

    name: str = Field(..., min_length=1, max_length=255)
    grams: float = Field(..., ge=0, le=100000)


class GroceryListCreate(BaseModel):
    user_id: int
    session_id: Optional[int] = None
    meal_plan: Optional[List[GroceryFoodItem]] = None
    name: Optional[str] = Field(default=None, max_length=255)
    meal_prep_days: int = Field(default=1, ge=1, le=30)


class GroceryItemResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    food_name: str
    category: str
    quantity: float
    unit: str
    checked: bool


class GroceryListResponse(BaseModel):
    grocery_list_id: int
    name: str
    items: List[GroceryItemResponse]


class GroceryListDetailResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    session_id: Optional[int] = None
    name: str
    created_at: datetime
    items: List[GroceryItemResponse]


class GroceryListSummary(BaseModel):
    """Lightweight grocery-list row for listing a user's lists."""

    id: int
    name: str
    created_at: datetime
    item_count: int


class BodyCompEntry(BaseModel):
    """A single body-composition measurement (for history listings)."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    date: date_cls
    weight_kg: float
    body_fat_percent: Optional[float] = None
    lean_mass_kg: Optional[float] = None
    created_at: datetime


# --------------------------------------------------------------------------- #
# Phase 5: restaurant mode
# --------------------------------------------------------------------------- #
class MatchedFood(BaseModel):
    id: int
    name: str


class ParsedReceiptItem(BaseModel):
    dish_name: str
    price: Optional[float] = None
    matched_food: Optional[MatchedFood] = None
    confidence: float = 0.0
    suggested_grams: Optional[float] = None
    warning: Optional[str] = None


class ReceiptUploadResponse(BaseModel):
    receipt_text: str
    parsed_items: List[ParsedReceiptItem]


class RestaurantItemIn(BaseModel):
    food_id: int
    grams: float = Field(..., gt=0, le=10000)
    food_name: Optional[str] = Field(default=None, max_length=255)
    is_substituted: bool = False


class RestaurantLogCreate(BaseModel):
    user_id: int
    restaurant_log_id: Optional[int] = None
    items: List[RestaurantItemIn] = Field(default_factory=list)
    restaurant_name: Optional[str] = Field(default=None, max_length=255)
    date: Optional[date_cls] = None
    notes: Optional[str] = None


class RestaurantItemResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    food_name: str
    matched_food_id: Optional[int] = None
    serving_grams: Optional[float] = None
    calories_estimate: Optional[float] = None
    is_substituted: bool


class RestaurantLogResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    date: date_cls
    restaurant_name: Optional[str] = None
    image_path: Optional[str] = None
    notes: Optional[str] = None
    created_at: datetime
    items: List[RestaurantItemResponse]


class SubstituteRequest(BaseModel):
    user_id: int
    food_id: int
    limit: int = Field(default=3, ge=1, le=10)


class SubstituteResponse(BaseModel):
    original_food_id: int
    substitutions: List[FoodSearchResult]


# --------------------------------------------------------------------------- #
# Phase 6: reinforcement learning (Part 1)
# --------------------------------------------------------------------------- #
class RewardComputeRequest(BaseModel):
    user_id: int
    date: Optional[date_cls] = None


class RewardResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    user_id: int
    date: date_cls
    reward: float
    components: Dict[str, float] = {}


class RewardComputeResponse(BaseModel):
    user_id: int
    date: date_cls
    reward: float
    components: Dict[str, float] = {}
    context: Dict[str, float] = {}


class RLRecommendRequest(BaseModel):
    user_id: int
    meal_type: Optional[str] = None


class RLRecommendedFood(BaseModel):
    food_id: int
    name: str
    protein_g: float
    carbs_g: float
    fat_g: float
    calories: float
    suggested_grams: float
    predicted_score: float
    strategy_fit: Optional[float] = None


class RLRecommendResponse(BaseModel):
    action_name: str
    recommended_macros: Dict[str, float]
    foods: List[RLRecommendedFood]
    used_rl: bool = True
    fallback_reason: Optional[str] = None


class RLUpdateRequest(BaseModel):
    user_id: int
    date: Optional[date_cls] = None


class RLUpdateResponse(BaseModel):
    status: str  # "updated" | "skipped"
    reason: Optional[str] = None
    action: Optional[str] = None
    reward: Optional[float] = None
    update_count: Optional[int] = None
    transition_id: Optional[int] = None


class RLStatusResponse(BaseModel):
    update_count: int
    epsilon: float
    n_actions: int
    context_dim: int
    alpha: float
    weights_persisted: bool


class RLEvalResponse(BaseModel):
    rl_avg_reward: Optional[float] = None
    xgboost_avg_reward: Optional[float] = None
    improvement_percent: Optional[float] = None
    recommendation: str
    rl_days: int
    xgboost_days: int


class RLSwitchRequest(BaseModel):
    user_id: int
    force_rl: bool


class RLSwitchResponse(BaseModel):
    user_id: int
    forced_rl: bool


# --------------------------------------------------------------------------- #
# Phase 7: nearby restaurants
# --------------------------------------------------------------------------- #
class NearbyMenuItem(BaseModel):
    name: str
    calories: float
    protein_g: float
    carbs_g: float
    fat_g: float
    serving_size_g: float = 100.0
    macro_fit: float = 0.0
    score: float = 0.0


class NearbyRestaurant(BaseModel):
    name: str
    address: Optional[str] = None
    cuisine: Optional[str] = None
    menu_items: List[NearbyMenuItem] = []


class NearbySearchResponse(BaseModel):
    location: str
    used_rl: bool = False
    action_name: Optional[str] = None
    restaurants: List[NearbyRestaurant] = []


class NearbyLogRequest(BaseModel):
    user_id: int
    restaurant_name: str = Field(..., min_length=1, max_length=255)
    item_name: str = Field(..., min_length=1, max_length=255)
    grams: float = Field(..., gt=0, le=10000)


class NearbyLogResponse(BaseModel):
    restaurant_meal_log_id: int
    meal_log_id: int
    macros: Dict[str, float]
    feedback_prompt: str
