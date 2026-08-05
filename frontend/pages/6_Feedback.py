"""Feedback: rate meals logged this session (enjoyment/satiety/energy)."""

from __future__ import annotations

import streamlit as st

import api_client as api
from utils import helpers

helpers.setup_page("Feedback", icon="⭐")
user_id = helpers.render_sidebar()

st.markdown("# ⭐ Rate Your Meals")
st.caption("Feedback improves your recommendations (and the RL reward). "
           "Meals shown were logged during this session.")

meals = [m for m in st.session_state.today_meals if m.get("meal_log_id")]
if not meals:
    st.info("No rateable meals yet. Log a meal from **Chat**, **Optimize**, or **Restaurants** first.")
    st.stop()

labels = [f"{m.get('name','Meal')} · {m.get('meal_type','').title()}" for m in meals]
idx = st.selectbox("Meal", range(len(meals)), format_func=lambda i: labels[i])
meal = meals[idx]

with st.form("feedback"):
    st.markdown(f"**{meal.get('name','Meal')}**")
    c1, c2, c3 = st.columns(3)
    enjoyment = c1.slider("😋 Enjoyment", 1, 5, 4)
    satiety = c2.slider("🍽️ Satiety", 1, 5, 4)
    energy = c3.slider("⚡ Energy", 1, 5, 4)
    workout = st.slider("💪 Workout performance (optional)", 0, 5, 0,
                        help="0 = not applicable")
    notes = st.text_area("Notes", placeholder="Anything to remember about this meal?")
    if st.form_submit_button("Submit feedback", type="primary", use_container_width=True):
        payload = {
            "user_id": user_id, "meal_log_id": meal["meal_log_id"],
            "enjoyment": enjoyment, "satiety": satiety, "energy": energy,
            "notes": notes or None,
        }
        if workout > 0:
            payload["workout_performance"] = workout
        try:
            api.submit_feedback(payload)
            st.success("Thanks! Your feedback was recorded. 🙌")
            st.balloons()
        except api.APIError as exc:
            helpers.toast_error(exc)
