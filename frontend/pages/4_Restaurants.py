"""Restaurants: find nearby spots with macro-friendly menu items."""

from __future__ import annotations

import streamlit as st

import api_client as api
from components import restaurant_card
from utils import helpers

helpers.setup_page("Restaurants")
user_id = helpers.render_sidebar()

helpers.page_header("pin", "Restaurants", "Nearby menus ranked to your macros.")
st.markdown(
    '<p class="mm-sub" style="margin:-.4rem 0 1rem;">Powered by free OpenStreetMap + Nutritionix. '
    'Requires the backend to have Nutritionix keys configured.</p>',
    unsafe_allow_html=True,
)

with st.form("nearby"):
    use_ip = st.checkbox("Use my approximate location (IP-based)", value=True)
    c1, c2, c3 = st.columns(3, gap="medium")
    lat = c1.number_input("Latitude", -90.0, 90.0, 40.7128, disabled=use_ip, format="%.4f")
    lon = c2.number_input("Longitude", -180.0, 180.0, -74.0060, disabled=use_ip, format="%.4f")
    radius_km = c3.slider("Radius (km)", 1, 10, 2)
    submitted = st.form_submit_button("Find restaurants", type="primary")

if submitted:
    with st.spinner("Searching nearby restaurants…"):
        try:
            kwargs = {"radius": radius_km * 1000}
            if not use_ip:
                kwargs.update(lat=lat, lon=lon)
            res = api.nearby_search(user_id, **kwargs)
            st.session_state["nearby_results"] = res
        except api.APIError as exc:
            msg = str(exc)
            if "503" in msg or "Nutritionix" in msg:
                st.warning(
                    "Nutritionix isn't configured on the backend. Set "
                    "NUTRITIONIX_APP_ID and NUTRITIONIX_API_KEY and restart it. "
                    "(Free signup at nutritionix.com — no card required.)"
                )
            else:
                helpers.toast_error(exc)

res = st.session_state.get("nearby_results")
if res:
    label = res.get("location", "")
    action = res.get("action_name")
    st.markdown(
        f'<div class="mm-head" style="margin:1rem 0 .5rem;">{helpers.icon("pin", 18, helpers.COLORS["accent"])}'
        f'<span style="font-weight:600;font-size:1.1rem;">{label}</span></div>',
        unsafe_allow_html=True,
    )
    if res.get("used_rl") and action:
        st.markdown(
            f'<p class="mm-sub" style="margin:-.2rem 0 .6rem;">RL strategy for you today: '
            f'<b>{action.replace("_", " ").title()}</b></p>',
            unsafe_allow_html=True,
        )
    restaurants = res.get("restaurants", [])
    if not restaurants:
        st.info("No restaurants with menu data found here. Try a larger radius.")
    for idx, r in enumerate(restaurants):
        restaurant_card.restaurant_card(r, user_id, idx)
