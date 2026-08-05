"""Chat bubble + structured meal-plan card components."""

from __future__ import annotations

from typing import Dict, Optional

import streamlit as st

import api_client as api
from utils import helpers


def chat_bubble(role: str, content: str) -> None:
    """Render a chat message as a left/right bubble."""
    css = "mm-user" if role == "user" else "mm-assistant"
    align = "flex-end" if role == "user" else "flex-start"
    st.markdown(
        f"""<div style="display:flex;justify-content:{align};">
              <div class="mm-bubble {css}">{content}</div>
            </div>""",
        unsafe_allow_html=True,
    )


def meal_plan_card(plan: Optional[Dict], key: str) -> None:
    """Render an assistant meal plan with a 'Log This Meal' button."""
    if not plan or not isinstance(plan, dict):
        return
    foods = plan.get("foods", []) or []
    tm = plan.get("total_macros", {}) or {}
    with st.container(border=True):
        st.markdown(f"#### 🍱 {plan.get('meal_name', 'Meal plan')}")
        if plan.get("explanation"):
            st.caption(plan["explanation"])
        for f in foods:
            st.markdown(
                f"- **{f.get('name','?')}** — {helpers.fmt(f.get('grams'))} g"
            )
        if tm:
            st.caption(
                f"Total → 🔥 {helpers.fmt(tm.get('calories'))} kcal · "
                f"🥩 {helpers.fmt(tm.get('protein_g'))}p · "
                f"🍚 {helpers.fmt(tm.get('carbs_g'))}c · "
                f"🥑 {helpers.fmt(tm.get('fat_g'))}f"
            )

        loggable = [f for f in foods if f.get("food_id")]
        meal_type = st.selectbox(
            "Meal type", helpers.meal_types(), index=2, key=f"mt_{key}"
        )
        disabled = not loggable
        if st.button("✅ Log this meal", key=f"log_plan_{key}", type="primary",
                     disabled=disabled, use_container_width=True):
            _log_plan(loggable, meal_type)


def _log_plan(foods, meal_type) -> None:
    uid = int(st.session_state.user_id)
    logged = 0
    for f in foods:
        try:
            res = api.log_meal(uid, int(f["food_id"]), float(f.get("grams", 100)), meal_type)
            helpers.add_logged_meal({
                "name": f.get("name", "Meal"),
                "grams": res.get("grams_consumed", f.get("grams")),
                "meal_type": meal_type,
                "protein_g": res.get("protein_g", 0),
                "carbs_g": res.get("carbs_g", 0),
                "fat_g": res.get("fat_g", 0),
                "calories": res.get("calories", 0),
                "timestamp": res.get("timestamp", ""),
                "meal_log_id": res.get("id"),
            })
            logged += 1
        except api.APIError as exc:
            st.error(str(exc))
    if logged:
        st.success(f"Logged {logged} item(s)! 🎉")
        st.rerun()
