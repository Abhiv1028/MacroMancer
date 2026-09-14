"""A logged-meal row (persisted) with a quiet remove action."""

from __future__ import annotations

from typing import Dict

import streamlit as st

import api_client as api
from utils import helpers


def meal_card(meal: Dict) -> None:
    """Render a persisted meal (from GET /meals) with a working remove button."""
    ts = meal.get("timestamp", "")
    time_label = ts[11:16] if len(ts) >= 16 else ""
    name = meal.get("food_name") or meal.get("name") or "Meal"
    meta = meal.get("meal_type", "").title()
    if time_label:
        meta += f" &middot; {time_label}"

    with st.container(border=True):
        c1, c2 = st.columns([6, 1])
        with c1:
            st.markdown(
                f"""
                <div style="padding:.35rem .5rem;">
                  <div style="font-weight:600;color:{helpers.COLORS['text']};">{name}</div>
                  <div class="mm-sub" style="margin-top:.15rem;">{meta}</div>
                  <div class="mm-sub" style="margin-top:.35rem;">
                    {helpers.fmt(meal.get('grams_consumed'))} g &nbsp;·&nbsp;
                    {helpers.fmt(meal.get('calories'))} kcal &nbsp;·&nbsp;
                    P {helpers.fmt(meal.get('protein_g'))} &nbsp; C {helpers.fmt(meal.get('carbs_g'))} &nbsp; F {helpers.fmt(meal.get('fat_g'))}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown('<div style="height:.7rem;"></div>', unsafe_allow_html=True)
            if st.button("Remove", key=f"del_meal_{meal.get('id')}", use_container_width=True):
                try:
                    api.delete_meal(int(meal["id"]))
                    st.rerun()
                except api.APIError as exc:
                    st.error(str(exc))
