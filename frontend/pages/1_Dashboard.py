"""Dashboard: today's energy & macros + logged meals (persisted via the API)."""

from __future__ import annotations

from datetime import date

import streamlit as st

import api_client as api
from components import macro_progress, meal_card
from utils import helpers

helpers.setup_page("Dashboard")
user_id = helpers.render_sidebar()

helpers.page_header("grid", "Dashboard", date.today().strftime("%A, %B %d"))

try:
    targets = api.get_targets(user_id, helpers.today_str())
except api.APIError as exc:
    helpers.toast_error(exc)
    st.stop()

if not targets:
    st.info("No targets yet — create a profile in the sidebar to generate them.")
    st.stop()

consumed = helpers.today_consumed(user_id)
MC = helpers.MACRO_COLORS

macro_progress.calorie_hero(consumed["calories"], targets.get("calories", 0))

c1, c2, c3 = st.columns(3, gap="medium")
with c1:
    macro_progress.macro_bar("Protein", consumed["protein_g"], targets.get("protein_g", 0), color=MC["protein"])
with c2:
    macro_progress.macro_bar("Carbohydrate", consumed["carbs_g"], targets.get("carbs_g", 0), color=MC["carbs"])
with c3:
    macro_progress.macro_bar("Fat", consumed["fat_g"], targets.get("fat_g", 0), color=MC["fat"])

st.markdown('<hr class="mm-rule">', unsafe_allow_html=True)

st.markdown(
    f'<div class="mm-head" style="margin-bottom:1rem;">{helpers.icon("list", 18, helpers.COLORS["accent"])}'
    f'<span style="font-weight:600;font-size:1.1rem;">Today\'s meals</span></div>',
    unsafe_allow_html=True,
)
meals = helpers.todays_meals(user_id)
if not meals:
    st.info("No meals logged today. Use Optimize or Chat to plan and log a meal.")
else:
    for m in meals:
        meal_card.meal_card(m)
