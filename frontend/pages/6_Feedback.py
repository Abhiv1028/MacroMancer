"""Feedback: rate meals logged this session (enjoyment/satiety/energy)."""

from __future__ import annotations

import streamlit as st

import api_client as api
from utils import helpers

helpers.setup_page("Feedback")
user_id = helpers.render_sidebar()

helpers.page_header("star", "Feedback", "Rate meals to improve recommendations and the RL reward.")

meals = helpers.todays_meals(user_id)
if not meals:
    st.info("No meals logged today. Log a meal from Optimize, Chat, or Restaurants first.")
    st.stop()

labels = [f"{m.get('food_name','Meal')} · {m.get('meal_type','').title()}" for m in meals]
idx = st.selectbox("Meal", range(len(meals)), format_func=lambda i: labels[i])
meal = meals[idx]

with st.form("feedback"):
    st.markdown(f'<div style="font-weight:600;margin-bottom:.4rem;">{meal.get("food_name","Meal")}</div>',
                unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3, gap="medium")
    enjoyment = c1.slider("Enjoyment", 1, 5, 4)
    satiety = c2.slider("Satiety", 1, 5, 4)
    energy = c3.slider("Energy", 1, 5, 4)
    workout = st.slider("Workout performance (optional)", 0, 5, 0, help="0 = not applicable")
    notes = st.text_area("Notes", placeholder="Anything to remember about this meal?")
    if st.form_submit_button("Submit feedback", type="primary", use_container_width=True):
        payload = {
            "user_id": user_id, "meal_log_id": meal["id"],
            "enjoyment": enjoyment, "satiety": satiety, "energy": energy,
            "notes": notes or None,
        }
        if workout > 0:
            payload["workout_performance"] = workout
        try:
            api.submit_feedback(payload)
            st.success("Thanks — your feedback was recorded.")
        except api.APIError as exc:
            helpers.toast_error(exc)
