"""Chat bubbles + structured meal-plan card."""

from __future__ import annotations

from typing import Dict, Optional

import streamlit as st

import api_client as api
from utils import helpers


def chat_bubble(role: str, content: str) -> None:
    """Render a chat message as a left/right bubble."""
    css = "mm-user" if role == "user" else "mm-assistant"
    justify = "flex-end" if role == "user" else "flex-start"
    st.markdown(
        f'<div class="mm-row" style="justify-content:{justify};">'
        f'<div class="mm-bubble {css}">{content}</div></div>',
        unsafe_allow_html=True,
    )


def meal_plan_card(plan: Optional[Dict], key: str) -> None:
    """Render an assistant meal plan with a 'Log this meal' action."""
    if not plan or not isinstance(plan, dict):
        return
    foods = plan.get("foods", []) or []
    tm = plan.get("total_macros", {}) or {}

    with st.container(border=True):
        st.markdown(
            f'<div style="padding:.4rem .5rem .1rem;display:flex;align-items:center;gap:.5rem;">'
            f'{helpers.icon("utensils", 17, helpers.COLORS["accent"])}'
            f'<span style="font-weight:600;">{plan.get("meal_name", "Meal plan")}</span></div>',
            unsafe_allow_html=True,
        )
        if plan.get("explanation"):
            st.markdown(
                f'<div class="mm-sub" style="padding:0 .5rem .3rem;">{plan["explanation"]}</div>',
                unsafe_allow_html=True,
            )
        rows = "".join(
            f'<div style="display:flex;justify-content:space-between;padding:.28rem .5rem;'
            f'border-top:1px solid {helpers.COLORS["border"]};">'
            f'<span style="color:{helpers.COLORS["text"]};">{f.get("name","—")}</span>'
            f'<span class="mm-sub">{helpers.fmt(f.get("grams"))} g</span></div>'
            for f in foods
        )
        st.markdown(rows, unsafe_allow_html=True)
        if tm:
            st.markdown(
                f'<div class="mm-sub" style="padding:.5rem .5rem 0;">Total &middot; '
                f'{helpers.fmt(tm.get("calories"))} kcal &nbsp; P {helpers.fmt(tm.get("protein_g"))} &nbsp;'
                f'C {helpers.fmt(tm.get("carbs_g"))} &nbsp; F {helpers.fmt(tm.get("fat_g"))}</div>',
                unsafe_allow_html=True,
            )

        loggable = [f for f in foods if f.get("food_id")]
        meal_type = st.selectbox("Meal type", helpers.meal_types(), index=2, key=f"mt_{key}")
        if st.button("Log this meal", key=f"log_plan_{key}", type="primary",
                     disabled=not loggable, use_container_width=True):
            _log_plan(loggable, meal_type)


def _log_plan(foods, meal_type) -> None:
    uid = int(st.session_state.user_id)
    logged = 0
    for f in foods:
        try:
            res = api.log_meal(uid, int(f["food_id"]), float(f.get("grams", 100)), meal_type)
            helpers.add_logged_meal({"meal_log_id": res.get("id")})
            logged += 1
        except api.APIError as exc:
            st.error(str(exc))
    if logged:
        st.success(f"Logged {logged} item(s).")
        st.rerun()
