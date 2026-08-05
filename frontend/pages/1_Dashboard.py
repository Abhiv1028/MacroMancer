"""Dashboard: today's calories & macros + logged meals.

Note: the backend has no `GET /meals` endpoint, so 'today's meals' reflects
meals logged during this browser session (via any page).
"""

from __future__ import annotations

from datetime import date

import streamlit as st

import api_client as api
from components import macro_progress, meal_card
from utils import helpers

helpers.setup_page("Dashboard", icon="📊")
user_id = helpers.render_sidebar()

st.markdown(f"# 👋 Hello, User #{user_id}")
st.caption(date.today().strftime("%A, %B %d, %Y"))

try:
    targets = api.get_targets(user_id, helpers.today_str())
except api.APIError as exc:
    helpers.toast_error(exc)
    st.stop()

if not targets:
    st.warning("No targets yet. Create a user in the sidebar to generate them.")
    st.stop()

consumed = helpers.today_consumed()

# --- Calorie hero + macro cards ------------------------------------------- #
macro_progress.calorie_hero(consumed["calories"], targets.get("calories", 0))

c1, c2, c3 = st.columns(3)
with c1:
    macro_progress.macro_bar("🥩 Protein", consumed["protein_g"], targets.get("protein_g", 0),
                             color=helpers.MACRO_COLORS["protein"])
with c2:
    macro_progress.macro_bar("🍚 Carbs", consumed["carbs_g"], targets.get("carbs_g", 0),
                             color=helpers.MACRO_COLORS["carbs"])
with c3:
    macro_progress.macro_bar("🥑 Fat", consumed["fat_g"], targets.get("fat_g", 0),
                             color=helpers.MACRO_COLORS["fat"])

st.divider()

# --- Today's meals --------------------------------------------------------- #
st.markdown("### 🍽️ Today's meals")
meals = st.session_state.today_meals
if not meals:
    st.info("No meals logged yet this session. Use **Chat** or **Optimize** to plan and log a meal.")
else:
    for i in range(len(meals) - 1, -1, -1):
        meal_card.meal_card(meals[i], i)
    if st.button("Clear today's log"):
        st.session_state.today_meals = []
        st.rerun()
