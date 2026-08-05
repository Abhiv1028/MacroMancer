"""Shared formatting, theming, layout, and session-state helpers."""

from __future__ import annotations

from datetime import date
from typing import Dict, List

import streamlit as st

import api_client as api

# --- palette --------------------------------------------------------------- #
COLORS = {
    "primary": "#2ECC71",
    "secondary": "#3498DB",
    "accent": "#F39C12",
    "bg": "#F8F9FA",
    "text": "#2C3E50",
    "success": "#27AE60",
    "warning": "#E74C3C",
    "muted": "#95A5A6",
}

MACRO_COLORS = {
    "calories": COLORS["accent"],
    "protein": COLORS["primary"],
    "carbs": COLORS["secondary"],
    "fat": "#9B59B6",
}


# --- formatting ------------------------------------------------------------ #
def today_str() -> str:
    return date.today().isoformat()


def fmt(value: float, digits: int = 0) -> str:
    try:
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return "0"


def pct(consumed: float, target: float) -> float:
    if not target:
        return 0.0
    return max(0.0, (consumed / target) * 100.0)


def ratio_color(ratio: float) -> str:
    """Traffic-light color for a consumed/target ratio."""
    if ratio <= 0:
        return COLORS["muted"]
    if 0.85 <= ratio <= 1.10:
        return COLORS["success"]          # on track
    if 0.60 <= ratio < 0.85 or 1.10 < ratio <= 1.25:
        return COLORS["accent"]           # getting close
    return COLORS["warning"]              # under / over


# --- theming --------------------------------------------------------------- #
def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        .stApp {{ background-color: {COLORS['bg']}; }}
        .block-container {{ padding-top: 2rem; max-width: 1100px; }}
        h1, h2, h3 {{ color: {COLORS['text']}; font-weight: 700; }}
        /* Cards */
        .mm-card {{
            background: #fff; border-radius: 12px; padding: 1.1rem 1.3rem;
            box-shadow: 0 2px 10px rgba(44,62,80,0.07);
            border: 1px solid #ECF0F1; margin-bottom: 0.9rem;
            transition: transform .15s ease, box-shadow .15s ease;
        }}
        .mm-card:hover {{ transform: translateY(-2px); box-shadow: 0 6px 18px rgba(44,62,80,0.12); }}
        .mm-metric-label {{ color: {COLORS['muted']}; font-size: .8rem; text-transform: uppercase; letter-spacing: .05em; }}
        .mm-metric-value {{ font-size: 1.9rem; font-weight: 800; line-height: 1.1; }}
        .mm-metric-sub {{ color: {COLORS['muted']}; font-size: .85rem; }}
        /* Progress bar */
        .mm-bar-track {{ background: #ECF0F1; border-radius: 999px; height: 12px; overflow: hidden; margin-top: .4rem; }}
        .mm-bar-fill {{ height: 100%; border-radius: 999px; transition: width .4s ease; }}
        /* Chat bubbles */
        .mm-bubble {{ padding: .7rem 1rem; border-radius: 16px; margin: .35rem 0; max-width: 80%;
                      line-height: 1.45; box-shadow: 0 1px 4px rgba(44,62,80,0.08); }}
        .mm-user {{ background: {COLORS['secondary']}; color: #fff; margin-left: auto; border-bottom-right-radius: 4px; }}
        .mm-assistant {{ background: #fff; color: {COLORS['text']}; border: 1px solid #ECF0F1; border-bottom-left-radius: 4px; }}
        .mm-pill {{ display:inline-block; padding:.15rem .6rem; border-radius:999px; font-size:.75rem;
                    font-weight:700; color:#fff; }}
        .stButton>button {{ border-radius: 8px; font-weight: 600; border: 0; }}
        .stButton>button:hover {{ filter: brightness(0.96); }}
        ::-webkit-scrollbar {{ width: 10px; height: 10px; }}
        ::-webkit-scrollbar-thumb {{ background: #CBD5DC; border-radius: 999px; }}
        ::-webkit-scrollbar-thumb:hover {{ background: {COLORS['muted']}; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def setup_page(title: str, icon: str = "🥗") -> None:
    """Call at the top of every page: config + CSS + session init."""
    st.set_page_config(
        page_title=f"Macromancer · {title}",
        page_icon=icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()
    init_state()


# --- session state --------------------------------------------------------- #
def init_state() -> None:
    ss = st.session_state
    ss.setdefault("user_id", 1)
    ss.setdefault("chat_history", [])          # list of {role, content, meal_plan, grocery_list_id}
    ss.setdefault("chat_session_id", None)
    ss.setdefault("today_meals", [])           # meals logged this session (no GET /meals endpoint)
    ss.setdefault("grocery_list_ids", [])      # grocery lists created this session
    ss.setdefault("weight_history", [])        # {date, weight} logged this session
    ss.setdefault("last_plan_foods", [])       # foods from the latest chat/optimize result


def add_logged_meal(meal: Dict) -> None:
    """Record a meal logged via the UI (backend has no list-meals endpoint)."""
    st.session_state.today_meals.append(meal)


def today_consumed() -> Dict[str, float]:
    totals = {"protein_g": 0.0, "carbs_g": 0.0, "fat_g": 0.0, "calories": 0.0}
    for m in st.session_state.today_meals:
        for k in totals:
            totals[k] += float(m.get(k, 0) or 0)
    return totals


# --- sidebar --------------------------------------------------------------- #
def render_sidebar() -> int:
    """Render the shared sidebar (user selector + quick stats). Returns user_id."""
    with st.sidebar:
        st.markdown("## 🥗 Macromancer")
        st.caption("Your AI nutrition coach")

        if not api.health():
            st.error("⚠️ Backend offline. Start it with `python run.py`.")

        st.session_state.user_id = st.number_input(
            "👤 User ID", min_value=1, step=1, value=int(st.session_state.user_id)
        )
        user_id = int(st.session_state.user_id)

        with st.expander("➕ Create a demo user"):
            _create_user_form()

        st.divider()
        _quick_stats(user_id)
        st.divider()
        st.caption("Navigate using the pages above ⬆️")
    return user_id


def _create_user_form() -> None:
    with st.form("create_user", clear_on_submit=False):
        c1, c2 = st.columns(2)
        age = c1.number_input("Age", 10, 120, 30)
        sex = c2.selectbox("Sex", ["male", "female"])
        weight = c1.number_input("Weight (kg)", 20.0, 400.0, 75.0)
        height = c2.number_input("Height (cm)", 50.0, 300.0, 178.0)
        activity = c1.selectbox("Activity", ["sedentary", "light", "moderate", "very", "extra"], index=2)
        goal = c2.selectbox("Goal", ["cut", "maintain", "bulk"], index=1)
        if st.form_submit_button("Create user", type="primary", use_container_width=True):
            try:
                res = api.create_user({
                    "age": age, "sex": sex, "weight_kg": weight, "height_cm": height,
                    "activity_level": activity, "goal": goal,
                })
                new_id = res["user"]["id"]
                st.session_state.user_id = new_id
                st.success(f"Created user #{new_id} 🎉")
                st.rerun()
            except api.APIError as exc:
                st.error(str(exc))


def _quick_stats(user_id: int) -> None:
    st.markdown("### Quick stats")
    try:
        user = api.get_user(user_id)
        st.metric("Weight", f"{fmt(user.get('weight_kg'), 1)} kg")
        st.caption(f"Goal: **{user.get('goal', '—').title()}** · {user.get('activity_level', '—').title()}")
    except api.APIError:
        st.info("Create or select a user to see stats.")
        return
    try:
        tdee = api.get_tdee(user_id)
        method = tdee.get("method_used", "mifflin")
        val = tdee.get("adaptive_tdee") or tdee.get("current_tdee")
        st.metric("TDEE", f"{fmt(val)} kcal", help=f"Method: {method}")
    except api.APIError:
        pass


def toast_error(exc: Exception) -> None:
    st.error(f"😕 {exc}")


def meal_types() -> List[str]:
    return ["breakfast", "lunch", "dinner", "snack"]
