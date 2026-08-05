"""Meal display card (used on the Dashboard for today's logged meals)."""

from __future__ import annotations

from typing import Dict

import streamlit as st

from utils import helpers

_MEAL_EMOJI = {"breakfast": "🍳", "lunch": "🥪", "dinner": "🍽️", "snack": "🍎"}


def meal_card(meal: Dict, index: int) -> None:
    """Render a single logged meal with a delete (session-only) button."""
    emoji = _MEAL_EMOJI.get(meal.get("meal_type", ""), "🍴")
    ts = meal.get("timestamp", "")
    time_label = ts[11:16] if len(ts) >= 16 else ""
    with st.container(border=True):
        c1, c2 = st.columns([5, 1])
        with c1:
            st.markdown(
                f"**{emoji} {meal.get('name', 'Meal')}**  "
                f"<span style='color:{helpers.COLORS['muted']};'>· {meal.get('meal_type','').title()} "
                f"{('· ' + time_label) if time_label else ''}</span>",
                unsafe_allow_html=True,
            )
            st.caption(
                f"{helpers.fmt(meal.get('grams'))} g · "
                f"🔥 {helpers.fmt(meal.get('calories'))} kcal · "
                f"🥩 {helpers.fmt(meal.get('protein_g'))}p · "
                f"🍚 {helpers.fmt(meal.get('carbs_g'))}c · "
                f"🥑 {helpers.fmt(meal.get('fat_g'))}f"
            )
        with c2:
            if st.button("🗑️", key=f"del_meal_{index}", help="Remove from today"):
                st.session_state.today_meals.pop(index)
                st.rerun()
