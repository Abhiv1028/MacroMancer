"""Synthetic training-data generator for the XGBoost macro optimizer.

Simulates 50 users over 30 days (configurable), each logging meals drawn from a
pool of plausible foods with random deviation from their macro targets. Produces
a feature/label DataFrame matching :data:`optimizer.FEATURE_COLUMNS`.

The food pool is generated in-memory so training never depends on the USDA
download having succeeded; the feature schema is identical to inference time.
"""

from __future__ import annotations

import random
from typing import Dict, List

import pandas as pd

from backend.config import settings
from backend.services.macro_calculator import calculate_macro_targets
from backend.services.optimizer import (
    FEATURE_COLUMNS,
    LABEL_COLUMN,
    encode_goal,
    encode_meal_type,
    label_meal,
)

ACTIVITY_LEVELS = ["sedentary", "light", "moderate", "very", "extra"]
GOALS = ["cut", "maintain", "bulk"]
SEXES = ["male", "female"]
MEAL_SEQUENCE = ["breakfast", "lunch", "dinner", "snack"]
MEAL_HOURS = {"breakfast": 8, "lunch": 13, "dinner": 19, "snack": 16}


def _make_food_pool(rng: random.Random, n: int = 60) -> List[Dict[str, float]]:
    """Create a pool of plausible foods with per-100g macros."""
    pool: List[Dict[str, float]] = []
    for _ in range(n):
        protein = round(rng.uniform(0, 30), 1)
        carbs = round(rng.uniform(0, 70), 1)
        fat = round(rng.uniform(0, 40), 1)
        calories = round(protein * 4 + carbs * 4 + fat * 9 + rng.uniform(-10, 10), 1)
        pool.append(
            {
                "protein_g": protein,
                "carbs_g": carbs,
                "fat_g": fat,
                "calories": max(calories, 1.0),
            }
        )
    return pool


def _random_user(rng: random.Random) -> Dict:
    """Generate a random user profile with computed macro targets."""
    weight = round(rng.uniform(50, 110), 1)
    height = round(rng.uniform(150, 195), 1)
    age = rng.randint(18, 65)
    sex = rng.choice(SEXES)
    activity = rng.choice(ACTIVITY_LEVELS)
    goal = rng.choice(GOALS)
    targets = calculate_macro_targets(weight, height, age, activity, goal, sex)
    return {
        "weight": weight,
        "height": height,
        "age": age,
        "sex": sex,
        "activity": activity,
        "goal": goal,
        "target": {
            "protein_g": targets.protein_g,
            "carbs_g": targets.carbs_g,
            "fat_g": targets.fat_g,
        },
    }


def generate_training_dataframe(
    n_users: int = None, n_days: int = None, seed: int = None
) -> pd.DataFrame:
    """Simulate meal logs and emit a feature/label DataFrame for training."""
    n_users = n_users if n_users is not None else settings.SYNTH_USERS
    n_days = n_days if n_days is not None else settings.SYNTH_DAYS
    seed = seed if seed is not None else settings.RANDOM_SEED

    rng = random.Random(seed)
    food_pool = _make_food_pool(rng)
    rows: List[dict] = []

    for _ in range(n_users):
        user = _random_user(rng)
        target = user["target"]

        n_meals = len(MEAL_SEQUENCE)
        for day in range(n_days):
            day_of_week = day % 7
            running = {"protein_g": 0.0, "carbs_g": 0.0, "fat_g": 0.0}

            for meal_type in MEAL_SEQUENCE:
                # Simulate a meal as roughly one meal's share of the daily target
                # for each macro, with an independent random deviation, so that
                # cumulative intake tracks the target as a real day would. Mixing
                # in an occasional random pool food adds off-target noise, giving
                # the "reduces remaining error without overshooting" label a
                # healthy balance of good and bad examples.
                if rng.random() < 0.25:
                    pf = rng.choice(food_pool)
                    grams = max(rng.gauss(150, 50), 20.0)
                    factor = grams / 100.0
                    meal = {
                        "protein_g": pf["protein_g"] * factor,
                        "carbs_g": pf["carbs_g"] * factor,
                        "fat_g": pf["fat_g"] * factor,
                    }
                else:
                    meal = {
                        macro: (target[macro] / n_meals) * rng.uniform(0.4, 1.6)
                        for macro in ("protein_g", "carbs_g", "fat_g")
                    }
                # Represent the meal as a per-100g "food" (serving = 100 g).
                food = {
                    "protein_g": round(meal["protein_g"], 2),
                    "carbs_g": round(meal["carbs_g"], 2),
                    "fat_g": round(meal["fat_g"], 2),
                    "calories": round(
                        meal["protein_g"] * 4 + meal["carbs_g"] * 4 + meal["fat_g"] * 9,
                        2,
                    ),
                }
                hour = MEAL_HOURS[meal_type] + rng.randint(-1, 1)
                label = label_meal(running, meal, target)
                # Feedback (1-5) mildly correlated with label so the model can
                # learn to prefer well-rated foods; real feedback is used at
                # inference/retraining time.
                feedback_mean = min(
                    5.0,
                    max(1.0, rng.gauss(3.6 if label == 1 else 3.0, 0.7)),
                )

                rows.append(
                    {
                        "user_goal": encode_goal(user["goal"]),
                        "meal_type": encode_meal_type(meal_type),
                        "hour_of_day": hour,
                        "day_of_week": day_of_week,
                        "current_protein_ratio": _ratio(
                            running["protein_g"], target["protein_g"]
                        ),
                        "current_carbs_ratio": _ratio(
                            running["carbs_g"], target["carbs_g"]
                        ),
                        "current_fat_ratio": _ratio(
                            running["fat_g"], target["fat_g"]
                        ),
                        "food_protein_per_100g": food["protein_g"],
                        "food_carbs_per_100g": food["carbs_g"],
                        "food_fat_per_100g": food["fat_g"],
                        "food_calories_per_100g": food["calories"],
                        "avg_feedback_score": round(feedback_mean, 3),
                        LABEL_COLUMN: label,
                    }
                )

                for macro in ("protein_g", "carbs_g", "fat_g"):
                    running[macro] += meal[macro]

    df = pd.DataFrame(rows, columns=FEATURE_COLUMNS + [LABEL_COLUMN])
    return df


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


if __name__ == "__main__":
    frame = generate_training_dataframe()
    print(f"Generated {len(frame)} rows.")
    print(f"Positive label rate: {frame[LABEL_COLUMN].mean():.3f}")
    print(frame.head())
