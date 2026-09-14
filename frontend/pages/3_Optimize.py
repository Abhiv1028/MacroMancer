"""Optimize: ML-ranked food recommendations for the next meal."""

from __future__ import annotations

import streamlit as st

import api_client as api
from utils import helpers

helpers.setup_page("Optimize")
user_id = helpers.render_sidebar()

helpers.page_header("target", "Optimize", "The best next foods for your remaining macros.")

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

st.markdown('<div class="mm-eyebrow" style="margin-bottom:.6rem;">Remaining today</div>',
            unsafe_allow_html=True)
r1, r2, r3 = st.columns(3, gap="medium")
r1.metric("Protein", f"{helpers.fmt(remaining['protein_g'])} g")
r2.metric("Carbohydrate", f"{helpers.fmt(remaining['carbs_g'])} g")
r3.metric("Fat", f"{helpers.fmt(remaining['fat_g'])} g")

st.markdown('<hr class="mm-rule">', unsafe_allow_html=True)
meal_type = st.selectbox("Meal type", helpers.meal_types(), index=2)

if st.button("Get recommendations", type="primary"):
    with st.spinner("Scoring foods with the ML model…"):
        try:
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
    st.markdown(
        f'<div class="mm-head" style="margin:1.2rem 0 .8rem;">{helpers.icon("star", 18, helpers.COLORS["accent"])}'
        f'<span style="font-weight:600;font-size:1.1rem;">Top picks</span></div>',
        unsafe_allow_html=True,
    )
    for i, rec in enumerate(results):
        macros = rec.get("macros_per_serving", {})
        with st.container(border=True):
            c1, c2 = st.columns([5, 2])
            with c1:
                score = rec.get("predicted_score", 0)
                st.markdown(
                    f'<div style="padding:.3rem .5rem;">'
                    f'<span style="font-weight:600;">{rec.get("name","Food")}</span> &nbsp;'
                    f'<span class="mm-pill">score {helpers.fmt(score*100)}%</span>'
                    f'<div class="mm-sub" style="margin-top:.35rem;">'
                    f'Suggested {helpers.fmt(rec.get("suggested_grams"))} g &nbsp;·&nbsp; '
                    f'{helpers.fmt(macros.get("calories"))} kcal &nbsp;·&nbsp; '
                    f'P {helpers.fmt(macros.get("protein_g"))} &nbsp; C {helpers.fmt(macros.get("carbs_g"))} &nbsp; F {helpers.fmt(macros.get("fat_g"))}'
                    f'</div></div>',
                    unsafe_allow_html=True,
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
                        st.success(f"Logged {rec.get('name')}.")
                        st.rerun()
                    except api.APIError as exc:
                        helpers.toast_error(exc)
