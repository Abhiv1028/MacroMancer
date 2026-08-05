"""SQLAlchemy ORM models for Macromancer.

Schema overview:
    User               -> profile + goals used to compute macro targets
    Food               -> USDA or custom food item
    Nutrient           -> per-100g nutrient value belonging to a Food
    Portion            -> a named serving size (grams) belonging to a Food
    MealLog            -> a single logged meal with derived macros/calories
    DailyMacroTarget   -> a user's macro targets for a given date
"""

from __future__ import annotations

from datetime import date, datetime
from typing import List, Optional

from sqlalchemy import (
    JSON,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.db import Base


class User(Base):
    """A person whose nutrition we optimize."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    age: Mapped[int] = mapped_column(Integer, nullable=False)
    weight_kg: Mapped[float] = mapped_column(Float, nullable=False)
    height_cm: Mapped[float] = mapped_column(Float, nullable=False)
    # sedentary / light / moderate / very / extra
    activity_level: Mapped[str] = mapped_column(String(20), nullable=False)
    # cut / maintain / bulk
    goal: Mapped[str] = mapped_column(String(20), nullable=False)
    # "male" / "female"; defaults to female per spec if unspecified.
    sex: Mapped[str] = mapped_column(String(10), nullable=False, default="female")
    # Optional JSON list of dietary restrictions, e.g. ["vegan", "gluten_free"].
    dietary_restrictions: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )

    meal_logs: Mapped[List["MealLog"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    macro_targets: Mapped[List["DailyMacroTarget"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    conversation_sessions: Mapped[List["ConversationSession"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    feedbacks: Mapped[List["Feedback"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    body_compositions: Mapped[List["BodyComposition"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    daily_summaries: Mapped[List["DailySummary"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    grocery_lists: Mapped[List["GroceryList"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    restaurant_logs: Mapped[List["RestaurantLog"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    rl_states: Mapped[List["RLState"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    rl_rewards: Mapped[List["RLReward"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    rl_action_logs: Mapped[List["RLActionLog"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    rl_transitions: Mapped[List["RLTransition"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    rl_fallback_logs: Mapped[List["RLFallbackLog"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    rl_override: Mapped[Optional["RLOverride"]] = relationship(
        back_populates="user", cascade="all, delete-orphan", uselist=False
    )
    rl_daily_performance: Mapped[List["RLDailyPerformance"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    nearby_searches: Mapped[List["NearbySearchCache"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    restaurant_meal_logs: Mapped[List["RestaurantMealLog"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )


class Food(Base):
    """A food item, either sourced from USDA or added as a custom food."""

    __tablename__ = "foods"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    usda_food_id: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True, index=True
    )
    name: Mapped[str] = mapped_column(String(500), nullable=False, index=True)
    category: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    # Default serving size in grams (used as a fallback suggested portion).
    default_grams: Mapped[float] = mapped_column(Float, nullable=False, default=100.0)
    is_custom: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    nutrients: Mapped[List["Nutrient"]] = relationship(
        back_populates="food", cascade="all, delete-orphan"
    )
    portions: Mapped[List["Portion"]] = relationship(
        back_populates="food", cascade="all, delete-orphan"
    )


class Nutrient(Base):
    """A single nutrient value for a food, normalized to a per-100g amount."""

    __tablename__ = "nutrients"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    food_id: Mapped[int] = mapped_column(
        ForeignKey("foods.id", ondelete="CASCADE"), nullable=False, index=True
    )
    nutrient_name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    amount_per_100g: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    unit: Mapped[str] = mapped_column(String(50), nullable=False, default="g")

    food: Mapped["Food"] = relationship(back_populates="nutrients")


class Portion(Base):
    """A named serving size for a food (e.g. '1 cup' -> 240 g)."""

    __tablename__ = "portions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    food_id: Mapped[int] = mapped_column(
        ForeignKey("foods.id", ondelete="CASCADE"), nullable=False, index=True
    )
    description: Mapped[str] = mapped_column(String(255), nullable=False)
    gram_weight: Mapped[float] = mapped_column(Float, nullable=False)

    food: Mapped["Food"] = relationship(back_populates="portions")


class MealLog(Base):
    """A logged meal. Macros/calories are derived at insert time and stored."""

    __tablename__ = "meal_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    food_id: Mapped[int] = mapped_column(
        ForeignKey("foods.id", ondelete="CASCADE"), nullable=False, index=True
    )
    portion_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("portions.id", ondelete="SET NULL"), nullable=True
    )
    grams_consumed: Mapped[float] = mapped_column(Float, nullable=False)
    # breakfast / lunch / dinner / snack
    meal_type: Mapped[str] = mapped_column(String(20), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now(), index=True
    )

    protein_g: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    carbs_g: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    fat_g: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    calories: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    user: Mapped["User"] = relationship(back_populates="meal_logs")
    food: Mapped["Food"] = relationship()


class DailyMacroTarget(Base):
    """A user's calorie/macro targets for a specific calendar date."""

    __tablename__ = "daily_macro_targets"
    __table_args__ = (
        UniqueConstraint("user_id", "date", name="uq_user_date_target"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    calories: Mapped[float] = mapped_column(Float, nullable=False)
    protein_g: Mapped[float] = mapped_column(Float, nullable=False)
    carbs_g: Mapped[float] = mapped_column(Float, nullable=False)
    fat_g: Mapped[float] = mapped_column(Float, nullable=False)

    user: Mapped["User"] = relationship(back_populates="macro_targets")


class ConversationSession(Base):
    """A chat session grouping a sequence of conversation turns for a user."""

    __tablename__ = "conversation_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    title: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )

    user: Mapped["User"] = relationship(back_populates="conversation_sessions")
    conversations: Mapped[List["Conversation"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="Conversation.created_at",
    )


class Conversation(Base):
    """A single turn (user or assistant) within a conversation session."""

    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(
        ForeignKey("conversation_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    # "user" or "assistant"
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    # Parsed meal-plan JSON attached to assistant turns (nullable). none_as_null
    # stores a Python None as SQL NULL rather than the JSON string "null".
    meal_plan: Mapped[Optional[dict]] = mapped_column(
        JSON(none_as_null=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )

    session: Mapped["ConversationSession"] = relationship(
        back_populates="conversations"
    )


class Feedback(Base):
    """Subjective feedback for a user, optionally tied to a specific meal."""

    __tablename__ = "feedbacks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    meal_log_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("meal_logs.id", ondelete="SET NULL"), nullable=True, index=True
    )
    enjoyment: Mapped[int] = mapped_column(Integer, nullable=False)  # 1-5
    satiety: Mapped[int] = mapped_column(Integer, nullable=False)  # 1-5
    energy: Mapped[int] = mapped_column(Integer, nullable=False)  # 1-5
    workout_performance: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True
    )  # 1-5
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )

    user: Mapped["User"] = relationship(back_populates="feedbacks")
    meal_log: Mapped[Optional["MealLog"]] = relationship()


class BodyComposition(Base):
    """A dated body-composition measurement for a user."""

    __tablename__ = "body_compositions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    weight_kg: Mapped[float] = mapped_column(Float, nullable=False)
    body_fat_percent: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    lean_mass_kg: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )

    user: Mapped["User"] = relationship(back_populates="body_compositions")


class DailySummary(Base):
    """Denormalized per-user/day roll-up used for fast adaptive-TDEE math."""

    __tablename__ = "daily_summaries"
    __table_args__ = (
        UniqueConstraint("user_id", "date", name="uq_user_date_summary"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    total_calories: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    total_protein: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    total_carbs: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    total_fat: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    weight_kg: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    tdee_estimate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    user: Mapped["User"] = relationship(back_populates="daily_summaries")


class GroceryList(Base):
    """A named shopping list generated from a meal plan for a user."""

    __tablename__ = "grocery_lists"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    session_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("conversation_sessions.id", ondelete="SET NULL"), nullable=True
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False, default="Grocery List")
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )

    user: Mapped["User"] = relationship(back_populates="grocery_lists")
    items: Mapped[List["GroceryItem"]] = relationship(
        back_populates="grocery_list",
        cascade="all, delete-orphan",
        order_by="GroceryItem.category, GroceryItem.food_name",
    )


class RestaurantLog(Base):
    """A record of a restaurant meal, optionally from a scanned receipt/menu."""

    __tablename__ = "restaurant_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    restaurant_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    image_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )

    user: Mapped["User"] = relationship(back_populates="restaurant_logs")
    items: Mapped[List["RestaurantItem"]] = relationship(
        back_populates="restaurant_log", cascade="all, delete-orphan"
    )


class RestaurantItem(Base):
    """A single dish within a restaurant log, matched to a USDA/custom food."""

    __tablename__ = "restaurant_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    restaurant_log_id: Mapped[int] = mapped_column(
        ForeignKey("restaurant_logs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    food_name: Mapped[str] = mapped_column(String(255), nullable=False)
    matched_food_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("foods.id", ondelete="SET NULL"), nullable=True
    )
    serving_grams: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    calories_estimate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    is_substituted: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    restaurant_log: Mapped["RestaurantLog"] = relationship(back_populates="items")
    matched_food: Mapped[Optional["Food"]] = relationship()


class GroceryItem(Base):
    """A single line item on a grocery list."""

    __tablename__ = "grocery_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    grocery_list_id: Mapped[int] = mapped_column(
        ForeignKey("grocery_lists.id", ondelete="CASCADE"), nullable=False, index=True
    )
    food_name: Mapped[str] = mapped_column(String(255), nullable=False)
    category: Mapped[str] = mapped_column(String(50), nullable=False, default="other")
    quantity: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    unit: Mapped[str] = mapped_column(String(20), nullable=False, default="g")
    checked: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    grocery_list: Mapped["GroceryList"] = relationship(back_populates="items")


# --------------------------------------------------------------------------- #
# Phase 6: Reinforcement Learning layer
# --------------------------------------------------------------------------- #
class RLAction(Base):
    """A fixed macro-strategy the RL layer can choose (e.g. 'high_protein')."""

    __tablename__ = "rl_actions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(50), nullable=False, unique=True, index=True)
    description: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)


class RLState(Base):
    """A snapshot of a user's decision context for a given day (one per day)."""

    __tablename__ = "rl_states"
    __table_args__ = (
        UniqueConstraint("user_id", "date", name="uq_rl_state_user_date"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    # Named feature dict, e.g. {"protein_remaining_ratio": 0.4, "hour": 0.5, ...}.
    context_vector: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )

    user: Mapped["User"] = relationship(back_populates="rl_states")


class RLReward(Base):
    """The scalar reward earned by a user on a given day (one per day)."""

    __tablename__ = "rl_rewards"
    __table_args__ = (
        UniqueConstraint("user_id", "date", name="uq_rl_reward_user_date"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    reward: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    # Breakdown, e.g. {"macro_adherence": 0.8, "feedback_score": 0.9, "satiety": 0.7}.
    components: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )

    user: Mapped["User"] = relationship(back_populates="rl_rewards")


class RLActionLog(Base):
    """A record of which action the bandit chose for a user on a day."""

    __tablename__ = "rl_action_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    action_id: Mapped[int] = mapped_column(
        ForeignKey("rl_actions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    context_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )

    user: Mapped["User"] = relationship(back_populates="rl_action_logs")
    action: Mapped["RLAction"] = relationship()


class RLTransition(Base):
    """A learned experience: (state+action) -> reward -> next_state.

    Joins an :class:`RLActionLog` (context + chosen action) to the day's
    :class:`RLReward`, optionally pointing at the next day's :class:`RLState`.
    One transition per day is expected (idempotency keyed on ``reward_id``).
    """

    __tablename__ = "rl_transitions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    action_log_id: Mapped[int] = mapped_column(
        ForeignKey("rl_action_logs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    reward_id: Mapped[int] = mapped_column(
        ForeignKey("rl_rewards.id", ondelete="CASCADE"), nullable=False, index=True
    )
    next_state_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("rl_states.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )

    user: Mapped["User"] = relationship(back_populates="rl_transitions")
    action_log: Mapped["RLActionLog"] = relationship()
    reward: Mapped["RLReward"] = relationship()
    next_state: Mapped[Optional["RLState"]] = relationship()


class RLFallbackLog(Base):
    """Records when a recommendation fell back from RL to the default optimizer."""

    __tablename__ = "rl_fallback_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    # e.g. "rl_underperforming" | "manual_override"
    reason: Mapped[str] = mapped_column(String(50), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )

    user: Mapped["User"] = relationship(back_populates="rl_fallback_logs")


class RLOverride(Base):
    """A manual per-user override forcing RL on or off (admin/debug)."""

    __tablename__ = "rl_overrides"
    __table_args__ = (
        UniqueConstraint("user_id", name="uq_rl_override_user"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    forced_rl: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )

    user: Mapped["User"] = relationship(back_populates="rl_override")


class RLDailyPerformance(Base):
    """Optional daily RL-vs-XGBoost reward summary (for monitoring/cron)."""

    __tablename__ = "rl_daily_performance"
    __table_args__ = (
        UniqueConstraint("user_id", "date", name="uq_rl_perf_user_date"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    rl_reward_avg: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    xgboost_reward_avg: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    improvement: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )

    user: Mapped["User"] = relationship(back_populates="rl_daily_performance")


# --------------------------------------------------------------------------- #
# Phase 7: Nearby restaurants (OpenStreetMap + Nutritionix)
# --------------------------------------------------------------------------- #
class NearbySearchCache(Base):
    """Cached nearby-search results for a user/location (TTL ~1 hour)."""

    __tablename__ = "nearby_search_caches"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    latitude: Mapped[float] = mapped_column(Float, nullable=False)
    longitude: Mapped[float] = mapped_column(Float, nullable=False)
    radius_m: Mapped[int] = mapped_column(Integer, nullable=False)
    results: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False, index=True
    )

    user: Mapped["User"] = relationship(back_populates="nearby_searches")


class RestaurantCache(Base):
    """Cached OSM restaurant + its Nutritionix menu (TTL ~7 days)."""

    __tablename__ = "restaurant_caches"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    osm_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    address: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    lat: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    lng: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    cuisine: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    menu_items: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    last_fetched: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )

    items: Mapped[List["CachedMenuItem"]] = relationship(
        back_populates="restaurant_cache", cascade="all, delete-orphan"
    )


class CachedMenuItem(Base):
    """A structured menu item belonging to a :class:`RestaurantCache`."""

    __tablename__ = "cached_menu_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    restaurant_cache_id: Mapped[int] = mapped_column(
        ForeignKey("restaurant_caches.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    calories: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    protein_g: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    carbs_g: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    fat_g: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    serving_size_g: Mapped[float] = mapped_column(Float, nullable=False, default=100.0)

    restaurant_cache: Mapped["RestaurantCache"] = relationship(back_populates="items")


class RestaurantMealLog(Base):
    """A meal logged from a nearby-restaurant menu item."""

    __tablename__ = "restaurant_meal_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    restaurant_name: Mapped[str] = mapped_column(String(255), nullable=False)
    item_name: Mapped[str] = mapped_column(String(255), nullable=False)
    grams: Mapped[float] = mapped_column(Float, nullable=False)
    meal_log_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("meal_logs.id", ondelete="SET NULL"), nullable=True
    )
    logged_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )

    user: Mapped["User"] = relationship(back_populates="restaurant_meal_logs")
