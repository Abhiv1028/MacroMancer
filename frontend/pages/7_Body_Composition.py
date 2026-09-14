"""Body Composition & TDEE: log weight, recalc targets, view weight trend."""

from __future__ import annotations

from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import api_client as api
from utils import helpers

helpers.setup_page("Body & TDEE")
user_id = helpers.render_sidebar()

helpers.page_header("scale", "Body & TDEE", "Log weight; your targets recalculate.")

# --- TDEE cards ------------------------------------------------------------ #
try:
    tdee = api.get_tdee(user_id)
    c1, c2, c3 = st.columns(3, gap="medium")
    c1.metric("Static TDEE", f"{helpers.fmt(tdee.get('current_tdee'))} kcal")
    adaptive = tdee.get("adaptive_tdee")
    c2.metric("Adaptive TDEE", f"{helpers.fmt(adaptive)} kcal" if adaptive else "—",
              help="Needs ≥ 7 days of data")
    c3.metric("Method", tdee.get("method_used", "mifflin").title())
except api.APIError as exc:
    helpers.toast_error(exc)

st.markdown('<hr class="mm-rule">', unsafe_allow_html=True)

# --- Log a measurement ----------------------------------------------------- #
st.markdown('<div class="mm-eyebrow" style="margin-bottom:.6rem;">Log a measurement</div>',
            unsafe_allow_html=True)
with st.form("bodycomp"):
    c1, c2, c3 = st.columns(3, gap="medium")
    weight = c1.number_input("Weight (kg)", 20.0, 400.0, 75.0, step=0.1)
    body_fat = c2.number_input("Body fat % (optional)", 0.0, 100.0, 0.0, step=0.1)
    d = c3.date_input("Date", value=date.today())
    if st.form_submit_button("Save & recalc targets", type="primary", use_container_width=True):
        payload = {"user_id": user_id, "weight_kg": weight, "date": d.isoformat()}
        if body_fat > 0:
            payload["body_fat_percent"] = body_fat
        try:
            res = api.log_body_composition(payload)
            # Bust cached user/targets/tdee so the sidebar & cards refresh.
            api.get_user.clear()
            api.get_targets.clear()
            api.get_tdee.clear()
            t = res.get("updated_targets", {})
            st.success(
                f"Saved. New targets → {helpers.fmt(t.get('calories'))} kcal · "
                f"P {helpers.fmt(t.get('protein_g'))} · C {helpers.fmt(t.get('carbs_g'))} · "
                f"F {helpers.fmt(t.get('fat_g'))}"
            )
        except api.APIError as exc:
            helpers.toast_error(exc)

# --- Weight trend ---------------------------------------------------------- #
st.markdown('<hr class="mm-rule">', unsafe_allow_html=True)
st.markdown(
    f'<div class="mm-head" style="margin-bottom:.8rem;">{helpers.icon("trending-up", 18, helpers.COLORS["accent"])}'
    f'<span style="font-weight:600;font-size:1.1rem;">Weight trend</span></div>',
    unsafe_allow_html=True,
)
try:
    history = api.body_comp_history(user_id)
except api.APIError:
    history = []
if len(history) < 2:
    st.info("Log at least two measurements to see a trend line.")
else:
    df = pd.DataFrame([{"date": h["date"], "weight": h["weight_kg"]} for h in history]).sort_values("date")
    fig = go.Figure(
        go.Scatter(
            x=df["date"], y=df["weight"], mode="lines+markers",
            line=dict(color=helpers.COLORS["accent"], width=2.5),
            marker=dict(size=7, color=helpers.COLORS["accent_600"]),
        )
    )
    fig.update_layout(
        height=340, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        yaxis_title="kg", xaxis_title=None,
        font=dict(color=helpers.COLORS["text"], family="Inter"),
        xaxis=dict(gridcolor=helpers.COLORS["border"]),
        yaxis=dict(gridcolor=helpers.COLORS["border"]),
    )
    st.plotly_chart(fig, use_container_width=True)
