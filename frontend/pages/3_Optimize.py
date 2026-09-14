"""Optimize: ML-ranked food recommendations for the next meal."""

from __future__ import annotations

import streamlit as st

import api_client as api
from utils import helpers

helpers.setup_page("Optimize", icon="🎯")
user_id = helpers.render_sidebar()

st.markdown("# 🎯 Optimize Your Next Meal")

try:
    targets = api.get_targets(user_id, helpers.today_str())
except api.APIError as exc:
    helpers.toast_error(exc)
    st.stop()

consumed = helpers.today_consumed(user_id)
remaining = {
    "protein_g": max(targets.get("protein_g", 0) - consumed["protein_g"], 0),
    "carbs_g": max(targets.get("carbs_g", 0) - consumed["carbs_g"], 0),
    "fat_g": max(targets.get("fat_g", 0) - consumed["fat_g"], 0),
}

st.markdown("### Remaining today")
r1, r2, r3 = st.columns(3)
r1.metric("🥩 Protein", f"{helpers.fmt(remaining['protein_g'])} g")
r2.metric("🍚 Carbs", f"{helpers.fmt(remaining['carbs_g'])} g")
r3.metric("🥑 Fat", f"{helpers.fmt(remaining['fat_g'])} g")

st.divider()
meal_type = st.selectbox("Meal type", helpers.meal_types(), index=2)

if st.button("✨ Get recommendations", type="primary"):
    with st.spinner("Scoring foods with the ML model…"):
        try:
            # current_macros = eaten so far (backend expects consumed).
            res = api.optimize(
                user_id,
                {"protein_g": consumed["protein_g"], "carbs_g": consumed["carbs_g"],
                 "fat_g": consumed["fat_g"]},
                meal_type,
            )
            st.session_state["opt_results"] = res.get("recommendations", [])
            st.session_state["opt_meal_type"] = meal_type
        except api.APIError as exc:
            helpers.toast_error(exc)

results = st.session_state.get("opt_results", [])
if results:
    st.markdown("### 🥇 Top picks")
    for i, rec in enumerate(results):
        macros = rec.get("macros_per_serving", {})
        with st.container(border=True):
            c1, c2 = st.columns([5, 2])
            with c1:
                score = rec.get("predicted_score", 0)
                st.markdown(
                    f"**{rec.get('name','Food')}** &nbsp; "
                    f"<span class='mm-pill' style='background:{helpers.ratio_color(score+0.3)};'>"
                    f"score {helpers.fmt(score*100)}%</span>",
                    unsafe_allow_html=True,
                )
                st.caption(
                    f"Suggested {helpers.fmt(rec.get('suggested_grams'))} g · "
                    f"🔥 {helpers.fmt(macros.get('calories'))} kcal · "
                    f"🥩 {helpers.fmt(macros.get('protein_g'))}p · "
                    f"🍚 {helpers.fmt(macros.get('carbs_g'))}c · "
                    f"🥑 {helpers.fmt(macros.get('fat_g'))}f"
                )
            with c2:
                grams = st.number_input(
                    "grams", 10, 2000, int(rec.get("suggested_grams", 100) or 100),
                    key=f"opt_g_{i}", label_visibility="collapsed",
                )
                if st.button("Log", key=f"opt_log_{i}", use_container_width=True):
                    try:
                        m = api.log_meal(user_id, int(rec["food_id"]), float(grams),
                                         st.session_state.get("opt_meal_type", meal_type))
                        helpers.add_logged_meal({
                            "name": rec.get("name"), "grams": m.get("grams_consumed", grams),
                            "meal_type": m.get("meal_type", meal_type),
                            "protein_g": m.get("protein_g", 0), "carbs_g": m.get("carbs_g", 0),
                            "fat_g": m.get("fat_g", 0), "calories": m.get("calories", 0),
                            "timestamp": m.get("timestamp", ""), "meal_log_id": m.get("id"),
                        })
                        st.success(f"Logged {rec.get('name')}! 🎉")
                        st.rerun()
                    except api.APIError as exc:
                        helpers.toast_error(exc)
