"""Meal display card (used on the Dashboard for today's logged meals)."""

from __future__ import annotations

from typing import Dict

import streamlit as st

import api_client as api
from utils import helpers

_MEAL_EMOJI = {"breakfast": "🍳", "lunch": "🥪", "dinner": "🍽️", "snack": "🍎"}


def meal_card(meal: Dict) -> None:
    """Render a persisted meal (from GET /meals) with a working delete button."""
    emoji = _MEAL_EMOJI.get(meal.get("meal_type", ""), "🍴")
    ts = meal.get("timestamp", "")
    time_label = ts[11:16] if len(ts) >= 16 else ""
    name = meal.get("food_name") or meal.get("name") or "Meal"
    with st.container(border=True):
        c1, c2 = st.columns([5, 1])
        with c1:
            st.markdown(
                f"**{emoji} {name}**  "
                f"<span style='color:{helpers.COLORS['muted']};'>· {meal.get('meal_type','').title()} "
                f"{('· ' + time_label) if time_label else ''}</span>",
                unsafe_allow_html=True,
            )
            st.caption(
                f"{helpers.fmt(meal.get('grams_consumed'))} g · "
                f"🔥 {helpers.fmt(meal.get('calories'))} kcal · "
                f"🥩 {helpers.fmt(meal.get('protein_g'))}p · "
                f"🍚 {helpers.fmt(meal.get('carbs_g'))}c · "
                f"🥑 {helpers.fmt(meal.get('fat_g'))}f"
            )
        with c2:
            if st.button("🗑️", key=f"del_meal_{meal.get('id')}", help="Delete this meal"):
                try:
                    api.delete_meal(int(meal["id"]))
                    st.rerun()
                except api.APIError as exc:
                    st.error(str(exc))
