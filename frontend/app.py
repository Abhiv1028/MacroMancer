"""Macromancer — Streamlit frontend entry point (landing)."""

from __future__ import annotations

import streamlit as st

from utils import helpers

helpers.setup_page("Home")
user_id = helpers.render_sidebar()
C = helpers.COLORS

st.markdown(
    f"""
    <div style="padding:1rem 0 .5rem;">
      <div class="mm-eyebrow">Precision nutrition</div>
      <h1 style="font-size:2.3rem;margin:.5rem 0 .4rem;max-width:20ch;">An adaptive macro coach that learns from your feedback.</h1>
      <p style="color:{C['muted']};font-size:1.05rem;max-width:58ch;line-height:1.6;">
        Track macros, get food recommendations from a model measured against baselines,
        plan meals in natural language, and watch your targets adapt — quietly, elegantly.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<hr class="mm-rule">', unsafe_allow_html=True)

FEATURES = [
    ("grid", "Dashboard", "Today's energy and macros, at a glance."),
    ("message", "Chat", "Describe what you want; get a meal plan."),
    ("target", "Optimize", "The best next foods for your remaining budget."),
    ("pin", "Restaurants", "Nearby menus ranked to your macros."),
    ("cart", "Grocery", "Turn a plan into a categorized list."),
    ("scale", "Body & TDEE", "Log weight; targets recalculate."),
]
cols = st.columns(3, gap="medium")
for i, (ic, title, desc) in enumerate(FEATURES):
    with cols[i % 3]:
        st.markdown(
            f"""
            <div class="mm-card" style="min-height:132px;">
              <div class="mm-head-ic" style="width:38px;height:38px;border-radius:10px;">{helpers.icon(ic, 19)}</div>
              <div style="font-weight:600;font-size:1.05rem;margin:.7rem 0 .25rem;color:{C['text']};">{title}</div>
              <div class="mm-sub">{desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown('<hr class="mm-rule">', unsafe_allow_html=True)
st.markdown(
    f"""
    <div class="mm-card" style="border-left:3px solid {C['accent']};">
      <div class="mm-eyebrow">Getting started</div>
      <ol style="margin:.6rem 0 0 1.1rem;color:{C['text']};line-height:1.9;">
        <li>Pick or create a profile in the sidebar.</li>
        <li>Open <b>Dashboard</b> to see today's targets.</li>
        <li>Use <b>Chat</b> or <b>Optimize</b> to plan a meal, then log it.</li>
      </ol>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    f'<p class="mm-sub" style="margin-top:1rem;">Viewing profile {user_id} &middot; {helpers.today_str()}</p>',
    unsafe_allow_html=True,
)
