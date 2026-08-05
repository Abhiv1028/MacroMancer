"""Restaurant result card with expandable macro-friendly menu items."""

from __future__ import annotations

from typing import Dict

import streamlit as st

import api_client as api
from utils import helpers


def _fit_pill(fit: float) -> str:
    color = helpers.ratio_color(fit)  # fit in [0,1] reuses traffic-light-ish scale
    return f'<span class="mm-pill" style="background:{color};">fit {helpers.fmt(fit*100)}%</span>'


def restaurant_card(restaurant: Dict, user_id: int, idx: int) -> None:
    """Render a restaurant as an expander listing its top items with Log buttons."""
    name = restaurant.get("name", "Restaurant")
    address = restaurant.get("address") or ""
    cuisine = restaurant.get("cuisine") or ""
    items = restaurant.get("menu_items", []) or []

    header = f"🍴 {name}"
    with st.expander(header, expanded=(idx == 0)):
        meta = " · ".join([p for p in [address, cuisine.title()] if p])
        if meta:
            st.caption(meta)
        if not items:
            st.info("No menu items available for this restaurant.")
            return
        for j, item in enumerate(items):
            with st.container(border=True):
                c1, c2 = st.columns([5, 2])
                with c1:
                    st.markdown(
                        f"**{item.get('name','Item')}** &nbsp; {_fit_pill(item.get('macro_fit', 0))}",
                        unsafe_allow_html=True,
                    )
                    st.caption(
                        f"🔥 {helpers.fmt(item.get('calories'))} kcal · "
                        f"🥩 {helpers.fmt(item.get('protein_g'))}p · "
                        f"🍚 {helpers.fmt(item.get('carbs_g'))}c · "
                        f"🥑 {helpers.fmt(item.get('fat_g'))}f · "
                        f"score {helpers.fmt(item.get('score', 0)*100)}%"
                    )
                with c2:
                    grams = st.number_input(
                        "grams", 10, 2000, int(item.get("serving_size_g", 100) or 100),
                        key=f"g_{idx}_{j}", label_visibility="collapsed",
                    )
                    if st.button("Log", key=f"rlog_{idx}_{j}", use_container_width=True):
                        _log_item(user_id, name, item.get("name", ""), grams)


def _log_item(user_id: int, restaurant: str, item_name: str, grams: float) -> None:
    try:
        res = api.nearby_log(user_id, restaurant, item_name, grams)
        macros = res.get("macros", {})
        helpers.add_logged_meal({
            "name": f"{restaurant} · {item_name}",
            "grams": grams, "meal_type": "dinner",
            "protein_g": macros.get("protein_g", 0),
            "carbs_g": macros.get("carbs_g", 0),
            "fat_g": macros.get("fat_g", 0),
            "calories": macros.get("calories", 0),
            "timestamp": "", "meal_log_id": res.get("meal_log_id"),
        })
        st.success(f"Logged {item_name}! 🎉")
        st.rerun()
    except api.APIError as exc:
        st.error(str(exc))
