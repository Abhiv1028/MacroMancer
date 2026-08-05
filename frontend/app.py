"""Macromancer — Streamlit frontend entry point (landing page).

Run with:  streamlit run frontend/app.py
The Dashboard and feature pages live in ``frontend/pages/``.
"""

from __future__ import annotations

import streamlit as st

from utils import helpers

helpers.setup_page("Home", icon="🥗")
user_id = helpers.render_sidebar()

st.markdown("# 🥗 Welcome to Macromancer")
st.markdown(
    "#### Your personal, AI-driven nutrition coach — meal planning, macro "
    "optimization, restaurants, groceries, and adaptive targets, all in one place."
)

st.markdown("")
cols = st.columns(3)
FEATURES = [
    ("📊", "Dashboard", "Track today's calories & macros at a glance."),
    ("💬", "Chat", "Ask the AI to build a meal plan from your remaining macros."),
    ("🎯", "Optimize", "Get the best foods for your next meal via ML."),
    ("📍", "Restaurants", "Find nearby spots with macro-friendly menu items."),
    ("🛒", "Grocery", "Turn a meal plan into a categorized shopping list."),
    ("⚖️", "Body & TDEE", "Log weight and watch your targets adapt."),
]
for i, (emoji, title, desc) in enumerate(FEATURES):
    with cols[i % 3]:
        st.markdown(
            f"""<div class="mm-card" style="min-height:130px;">
                  <div style="font-size:2rem;">{emoji}</div>
                  <div style="font-weight:700;font-size:1.1rem;color:{helpers.COLORS['text']};">{title}</div>
                  <div class="mm-metric-sub">{desc}</div>
                </div>""",
            unsafe_allow_html=True,
        )

st.divider()
st.markdown(
    f"""
    <div class="mm-card" style="border-left:6px solid {helpers.COLORS['secondary']};">
      <b>Getting started</b>
      <ol style="margin:.4rem 0 0 1rem;color:{helpers.COLORS['text']};">
        <li>Pick or create a user in the sidebar 👤</li>
        <li>Open the <b>Dashboard</b> to see today's targets</li>
        <li>Use <b>Chat</b> or <b>Optimize</b> to plan meals, then log them</li>
      </ol>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption(f"Currently viewing as **User #{user_id}** · {helpers.today_str()}")
