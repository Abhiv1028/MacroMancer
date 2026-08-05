"""Grocery: view a list grouped by aisle, check items off, generate new lists.

Note: the backend has no 'list all grocery lists' endpoint, so this page tracks
lists created during this session (plus lets you open one by id).
"""

from __future__ import annotations

from collections import defaultdict

import streamlit as st

import api_client as api
from utils import helpers

helpers.setup_page("Grocery", icon="🛒")
user_id = helpers.render_sidebar()

st.markdown("# 🛒 Grocery Lists")

_CATEGORY_EMOJI = {
    "produce": "🥦", "dairy": "🧀", "meat": "🥩", "seafood": "🐟",
    "pantry": "🥫", "frozen": "🧊", "spices": "🧂", "other": "📦",
}

# --- Generate a new list --------------------------------------------------- #
with st.expander("➕ Generate a new grocery list", expanded=False):
    st.caption("Enter foods as `name: grams`, one per line.")
    default = "Chicken Breast: 600\nBroccoli: 400\nWhite Rice: 500"
    text = st.text_area("Foods", value=default, height=120, label_visibility="collapsed")
    c1, c2 = st.columns(2)
    name = c1.text_input("List name", "Weekly Prep")
    prep_days = c2.number_input("Meal-prep days", 1, 30, 2)
    if st.button("Create list", type="primary"):
        foods = []
        for line in text.splitlines():
            if ":" in line:
                nm, _, grams = line.partition(":")
                try:
                    foods.append({"name": nm.strip(), "grams": float(grams.strip())})
                except ValueError:
                    continue
        if not foods:
            st.warning("Add at least one `name: grams` line.")
        else:
            try:
                res = api.create_grocery_list({
                    "user_id": user_id, "name": name,
                    "meal_prep_days": int(prep_days), "meal_plan": foods,
                })
                gid = res["grocery_list_id"]
                if gid not in st.session_state.grocery_list_ids:
                    st.session_state.grocery_list_ids.append(gid)
                st.session_state["grocery_selected"] = gid
                st.success(f"Created list #{gid} 🎉")
                st.rerun()
            except api.APIError as exc:
                helpers.toast_error(exc)

st.divider()

# --- Select a list --------------------------------------------------------- #
ids = st.session_state.grocery_list_ids
c1, c2 = st.columns([3, 2])
selected = None
if ids:
    selected = c1.selectbox("Your lists (this session)", ids,
                            index=len(ids) - 1, format_func=lambda i: f"List #{i}")
manual = c2.number_input("…or open by ID", min_value=0, step=1, value=0)
if manual:
    selected = manual

if not selected:
    st.info("Create a list above, or ask in **Chat** ('give me a grocery list'), "
            "then it'll appear here.")
    st.stop()

# --- Display the list ------------------------------------------------------ #
try:
    data = api.get_grocery_list(int(selected))
except api.APIError as exc:
    helpers.toast_error(exc)
    st.stop()

st.markdown(f"### {data.get('name', 'Grocery List')}  ·  List #{selected}")
items = data.get("items", [])
if not items:
    st.info("This list has no items.")
    st.stop()

grouped = defaultdict(list)
for it in items:
    grouped[it.get("category", "other")].append(it)

checked_count = sum(1 for it in items if it.get("checked"))
st.progress(checked_count / len(items), text=f"{checked_count}/{len(items)} checked off")

for category in sorted(grouped):
    st.markdown(f"#### {_CATEGORY_EMOJI.get(category, '📦')} {category.title()}")
    for it in grouped[category]:
        label = f"{it['food_name']} — {helpers.fmt(it['quantity'], 2)} {it['unit']}"
        new_val = st.checkbox(label, value=it.get("checked", False), key=f"gi_{it['id']}")
        if new_val != it.get("checked", False):
            try:
                api.toggle_grocery_item(it["id"])
                st.rerun()
            except api.APIError as exc:
                helpers.toast_error(exc)
