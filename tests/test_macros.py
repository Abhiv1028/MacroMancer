"""Unit tests for macro math and the optimizer's meal-labeling logic."""

from __future__ import annotations

import pytest

from backend.services.macro_calculator import (
    calculate_bmr,
    calculate_macro_targets,
    calculate_tdee,
)
from backend.services.optimizer import label_meal


def test_bmr_male_vs_female_differ_by_constant():
    male = calculate_bmr(80, 180, 30, "male")
    female = calculate_bmr(80, 180, 30, "female")
    # +5 (male) vs -161 (female) => 166 difference.
    assert round(male - female, 6) == 166.0


def test_bmr_defaults_to_female():
    assert calculate_bmr(80, 180, 30) == calculate_bmr(80, 180, 30, "female")


def test_tdee_activity_factor():
    bmr = 1600.0
    assert calculate_tdee(bmr, "sedentary") == pytest.approx(1600 * 1.2)
    assert calculate_tdee(bmr, "extra") == pytest.approx(1600 * 1.9)


def test_macro_targets_rules():
    t = calculate_macro_targets(75, 180, 30, "moderate", "cut", "male")
    # protein = 2.2 * 75 = 165 (below 250 cap)
    assert t.protein_g == pytest.approx(165.0, abs=0.1)
    # fat = 0.8 * 75 = 60 (above 40 floor)
    assert t.fat_g == pytest.approx(60.0, abs=0.1)
    # calories roughly balance across macros
    reconstructed = t.protein_g * 4 + t.carbs_g * 4 + t.fat_g * 9
    assert reconstructed == pytest.approx(t.calories, rel=0.02)


def test_protein_cap_and_fat_floor():
    # Very heavy person => protein cap 250 kicks in.
    heavy = calculate_macro_targets(200, 190, 30, "moderate", "bulk", "male")
    assert heavy.protein_g == 250.0
    # Very light person => fat floor 40 kicks in.
    light = calculate_macro_targets(40, 150, 30, "sedentary", "cut", "female")
    assert light.fat_g == 40.0


def test_goal_scaling_orders_calories():
    cut = calculate_macro_targets(75, 180, 30, "moderate", "cut", "male").calories
    maintain = calculate_macro_targets(75, 180, 30, "moderate", "maintain", "male").calories
    bulk = calculate_macro_targets(75, 180, 30, "moderate", "bulk", "male").calories
    assert cut < maintain < bulk


# --- label_meal ---------------------------------------------------------------
TARGET = {"protein_g": 165.0, "carbs_g": 270.0, "fat_g": 60.0}


def test_label_rewards_progress_from_empty():
    # A solid breakfast on an empty stomach reduces remaining error => good.
    meal = {"protein_g": 40.0, "carbs_g": 60.0, "fat_g": 15.0}
    assert label_meal({"protein_g": 0, "carbs_g": 0, "fat_g": 0}, meal, TARGET) == 1


def test_label_perfect_single_macro_hit_is_good():
    # Hitting protein exactly (remaining -> 0) must NOT be punished.
    meal = {"protein_g": 60.0, "carbs_g": 0.0, "fat_g": 0.0}
    assert label_meal({"protein_g": 0, "carbs_g": 0, "fat_g": 0}, meal, TARGET) == 1


def test_label_penalizes_overshoot():
    # Already near target on protein; a huge protein meal overshoots badly.
    eaten = {"protein_g": 160.0, "carbs_g": 0.0, "fat_g": 0.0}
    meal = {"protein_g": 80.0, "carbs_g": 0.0, "fat_g": 0.0}
    assert label_meal(eaten, meal, TARGET) == 0


def test_label_penalizes_moving_away():
    # Already over target: adding more only increases error.
    eaten = {"protein_g": 170.0, "carbs_g": 280.0, "fat_g": 62.0}
    meal = {"protein_g": 5.0, "carbs_g": 5.0, "fat_g": 2.0}
    assert label_meal(eaten, meal, TARGET) == 0
