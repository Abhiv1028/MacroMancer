"""Restaurant result card with expandable, macro-ranked menu items."""

from __future__ import annotations

from typing import Dict

import streamlit as st

import api_client as api
from utils import helpers


def _fit_pill(fit: float) -> str:
    return f'<span class="mm-pill">{helpers.fmt(fit * 100)}% fit</span>'


def restaurant_card(restaurant: Dict, user_id: int, idx: int) -> None:
    """Render a restaurant as an expander of its top macro-fit items."""
    name = restaurant.get("name", "Restaurant")
    address = restaurant.get("address") or ""
    cuisine = (restaurant.get("cuisine") or "").title()
    items = restaurant.get("menu_items", []) or []

    with st.expander(name, expanded=(idx == 0)):
        meta = " · ".join(p for p in [address, cuisine] if p)
        if meta:
            st.markdown(f'<div class="mm-sub" style="margin:-.2rem 0 .6rem;">{meta}</div>',
                        unsafe_allow_html=True)
        if not items:
            st.info("No menu items available for this restaurant.")
            return
        for j, item in enumerate(items):
            with st.container(border=True):
                c1, c2 = st.columns([6, 2])
                with c1:
                    st.markdown(
                        f'<div style="padding:.3rem .5rem;">'
                        f'<span style="font-weight:600;">{item.get("name","Item")}</span> &nbsp; {_fit_pill(item.get("macro_fit", 0))}'
                        f'<div class="mm-sub" style="margin-top:.35rem;">'
                        f'{helpers.fmt(item.get("calories"))} kcal &nbsp;·&nbsp; '
                        f'P {helpers.fmt(item.get("protein_g"))} &nbsp; C {helpers.fmt(item.get("carbs_g"))} &nbsp; F {helpers.fmt(item.get("fat_g"))}'
                        f'</div></div>',
                        unsafe_allow_html=True,
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
        api.nearby_log(user_id, restaurant, item_name, grams)
        st.success(f"Logged {item_name}.")
        st.rerun()
    except api.APIError as exc:
        st.error(str(exc))
