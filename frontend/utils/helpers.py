"""Design system + shared helpers: palette, Lucide icons, CSS, layout, session."""

from __future__ import annotations

import os
from datetime import date
from typing import Dict, List

import streamlit as st

import api_client as api

# --------------------------------------------------------------------------- #
# Palette — a single cohesive scheme (deep sage accent + warm neutrals)
# --------------------------------------------------------------------------- #
COLORS = {
    "bg": "#F3F2EE",        # warm ivory (app background)
    "surface": "#FFFFFF",   # cards / sidebar / inputs
    "surface_2": "#FBFAF7", # subtle raised surface
    "text": "#1E211D",      # warm charcoal
    "muted": "#6B6A63",     # secondary text
    "faint": "#9A988F",     # tertiary text
    "border": "#E6E4DD",    # hairline borders
    "accent": "#46584A",    # the single accent (sage)
    "accent_600": "#3A4A3E",# darker accent (hover / hero)
    "tint": "#EAEEE9",      # accent wash (tracks, soft fills)
    "warn": "#B4785B",      # restrained clay (over-budget only)
    # legacy aliases (kept so components/pages don't break)
    "primary": "#46584A",
    "secondary": "#2E3A31",
    "success": "#46584A",
    "warning": "#B4785B",
}

# Subtle tonal variations of the accent — cohesive, not rainbow.
MACRO_COLORS = {
    "calories": "#3A4A3E",
    "protein": "#46584A",
    "carbs": "#5E7060",
    "fat": "#7C8A76",
}


# --------------------------------------------------------------------------- #
# Lucide icons (the Lucide icon set, inlined as SVG since Streamlit isn't React)
# --------------------------------------------------------------------------- #
_ICONS = {
    "leaf": '<path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10Z"/><path d="M2 21c0-3 1.85-5.36 5.08-6C9.5 14.52 12 13 13 12"/>',
    "grid": '<rect width="7" height="9" x="3" y="3" rx="1.5"/><rect width="7" height="5" x="14" y="3" rx="1.5"/><rect width="7" height="9" x="14" y="12" rx="1.5"/><rect width="7" height="5" x="3" y="16" rx="1.5"/>',
    "message": '<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>',
    "target": '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1"/>',
    "pin": '<path d="M20 10c0 6-8 12-8 12s-8-6-8-12a8 8 0 0 1 16 0Z"/><circle cx="12" cy="10" r="3"/>',
    "cart": '<circle cx="8" cy="21" r="1"/><circle cx="19" cy="21" r="1"/><path d="M2 3h2l2.6 12.4a2 2 0 0 0 2 1.6h9.8a2 2 0 0 0 1.9-1.5L22 7H5"/>',
    "star": '<polygon points="12 2.5 14.9 8.4 21.4 9.3 16.7 13.9 17.8 20.3 12 17.3 6.2 20.3 7.3 13.9 2.6 9.3 9.1 8.4"/>',
    "scale": '<path d="m16 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/><path d="m2 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/><path d="M7 21h10"/><path d="M12 3v18"/><path d="M3 7h2c2 0 5-1 7-2 2 1 5 2 7 2h2"/>',
    "search": '<circle cx="11" cy="11" r="7"/><path d="m21 21-4.3-4.3"/>',
    "user": '<path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/>',
    "user-plus": '<path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M19 8v6"/><path d="M22 11h-6"/>',
    "send": '<path d="M22 2 11 13"/><path d="M22 2 15 22l-4-9-9-4 20-7z"/>',
    "plus": '<path d="M5 12h14"/><path d="M12 5v14"/>',
    "check": '<path d="M20 6 9 17l-5-5"/>',
    "x": '<path d="M18 6 6 18"/><path d="m6 6 12 12"/>',
    "trash": '<path d="M3 6h18"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6"/><path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>',
    "chevron-right": '<path d="m9 18 6-6-6-6"/>',
    "log-out": '<path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/><path d="m16 17 5-5-5-5"/><path d="M21 12H9"/>',
    "trending-up": '<path d="M22 7 13.5 15.5 8.5 10.5 2 17"/><path d="M16 7h6v6"/>',
    "utensils": '<path d="M3 2v7c0 1.1.9 2 2 2a2 2 0 0 0 2-2V2"/><path d="M7 2v20"/><path d="M21 15V2a5 5 0 0 0-5 5v6c0 1.1.9 2 2 2h3Zm0 0v7"/>',
    "activity": '<path d="M22 12h-4l-3 9L9 3l-3 9H2"/>',
    "flame": '<path d="M12 2c1 4 5 5.5 5 9a5 5 0 0 1-10 0c0-1.5.6-2.7 1.4-3.6C8.9 8 9 9 10 9c0-2.5-1-4 2-7Z"/>',
    "refresh": '<path d="M3 12a9 9 0 0 1 15-6.7L21 8"/><path d="M21 3v5h-5"/><path d="M21 12a9 9 0 0 1-15 6.7L3 16"/><path d="M8 16H3v5"/>',
    "info": '<circle cx="12" cy="12" r="9"/><path d="M12 16v-4"/><path d="M12 8h.01"/>',
    "alert": '<circle cx="12" cy="12" r="9"/><path d="M12 8v4"/><path d="M12 16h.01"/>',
    "arrow-right": '<path d="M5 12h14"/><path d="m12 5 7 7-7 7"/>',
    "sparkles": '<path d="M12 3l1.8 4.7L18.5 9.5 13.8 11.3 12 16l-1.8-4.7L5.5 9.5l4.7-1.8z"/><path d="M19 15l.9 2.3L22 18l-2.1.7L19 21l-.9-2.3L16 18l2.1-.7z"/>',
    "clock": '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
    "calendar": '<rect width="18" height="18" x="3" y="4" rx="2"/><path d="M3 10h18"/><path d="M8 2v4"/><path d="M16 2v4"/>',
    "dot": '<circle cx="12" cy="12" r="5" fill="currentColor" stroke="none"/>',
    "compass": '<circle cx="12" cy="12" r="9"/><polygon points="16 8 10 10 8 16 14 14"/>',
    "list": '<path d="M8 6h13"/><path d="M8 12h13"/><path d="M8 18h13"/><path d="M3 6h.01"/><path d="M3 12h.01"/><path d="M3 18h.01"/>',
}


def icon(name: str, size: int = 18, color: str = "currentColor", stroke: float = 1.75) -> str:
    """Return an inline Lucide SVG string for `name` (safe to drop into markdown)."""
    inner = _ICONS.get(name, _ICONS["dot"])
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke}" '
        f'stroke-linecap="round" stroke-linejoin="round" '
        f'style="vertical-align:-0.16em;flex:none;">{inner}</svg>'
    )


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #
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
    """Accent for on-track/under; a restrained clay only when over budget."""
    if ratio > 1.08:
        return COLORS["warn"]
    return COLORS["accent"]


# --------------------------------------------------------------------------- #
# Theme / CSS
# --------------------------------------------------------------------------- #
def inject_css() -> None:
    c = COLORS
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        :root {{
            --bg:{c['bg']}; --surface:{c['surface']}; --text:{c['text']};
            --muted:{c['muted']}; --faint:{c['faint']}; --border:{c['border']};
            --accent:{c['accent']}; --accent-600:{c['accent_600']}; --tint:{c['tint']};
        }}

        html, body, [class*="css"], .stApp, [data-testid="stAppViewContainer"] {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            color: var(--text);
        }}
        .stApp {{ background: var(--bg); }}
        #MainMenu, header[data-testid="stHeader"], footer {{ visibility: hidden; height: 0; }}

        .block-container {{ padding: 3rem 2.5rem 4rem; max-width: 1080px; }}
        @media (max-width: 640px) {{ .block-container {{ padding: 1.5rem 1.1rem 3rem; }} }}

        /* Typography */
        h1, h2, h3, h4 {{ color: var(--text); font-weight: 600; letter-spacing: -0.02em; }}
        h1 {{ font-size: 1.7rem; }} h2 {{ font-size: 1.3rem; }} h3 {{ font-size: 1.05rem; }}
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li {{ color: var(--text); line-height: 1.6; }}
        small, .stCaption, [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * {{ color: var(--muted) !important; }}
        a {{ color: var(--accent); }}

        /* Page header */
        .mm-head {{ display:flex; align-items:center; gap:.85rem; margin-bottom:.35rem; }}
        .mm-head-ic {{ width:42px; height:42px; border-radius:12px; background:var(--tint);
            color:var(--accent); display:flex; align-items:center; justify-content:center; flex:none; }}
        .mm-head h1 {{ margin:0; line-height:1.1; }}
        .mm-head-sub {{ color:var(--muted); font-size:.92rem; margin:.15rem 0 0; }}
        .mm-rule {{ height:1px; background:var(--border); border:0; margin:1.4rem 0 1.8rem; }}

        /* Cards & bordered containers */
        .mm-card {{ background:var(--surface); border:1px solid var(--border); border-radius:16px;
            padding:1.3rem 1.4rem; margin-bottom:1rem; box-shadow:0 1px 2px rgba(30,33,29,.03);
            transition:border-color .2s ease, box-shadow .2s ease, transform .2s ease; }}
        .mm-card:hover {{ border-color:#D8D6CE; box-shadow:0 10px 30px -12px rgba(30,33,29,.16); transform:translateY(-1px); }}
        [data-testid="stVerticalBlockBorderWrapper"] {{
            background:var(--surface); border:1px solid var(--border) !important;
            border-radius:16px; padding:.35rem .25rem; box-shadow:0 1px 2px rgba(30,33,29,.03); }}

        .mm-eyebrow {{ color:var(--faint); font-size:.72rem; font-weight:600; text-transform:uppercase;
            letter-spacing:.14em; }}
        .mm-value {{ font-size:2rem; font-weight:600; letter-spacing:-0.03em; line-height:1.05; color:var(--text); }}
        .mm-unit {{ font-size:.95rem; color:var(--muted); font-weight:500; }}
        .mm-sub {{ color:var(--muted); font-size:.85rem; }}

        /* Progress */
        .mm-track {{ background:var(--tint); border-radius:999px; height:8px; overflow:hidden; margin:.7rem 0 .5rem; }}
        .mm-fill {{ height:100%; border-radius:999px; transition:width .5s cubic-bezier(.2,.8,.2,1); }}
        .mm-dot {{ display:inline-block; width:8px; height:8px; border-radius:999px; margin-right:.5rem; vertical-align:middle; }}

        /* Chat */
        .mm-row {{ display:flex; margin:.4rem 0; }}
        .mm-bubble {{ padding:.7rem 1rem; border-radius:16px; max-width:82%; line-height:1.55; font-size:.94rem; }}
        .mm-user {{ background:var(--accent); color:#F4F3EF; margin-left:auto; border-bottom-right-radius:5px; }}
        .mm-assistant {{ background:var(--surface); color:var(--text); border:1px solid var(--border);
            border-bottom-left-radius:5px; }}

        .mm-pill {{ display:inline-flex; align-items:center; gap:.3rem; padding:.2rem .6rem; border-radius:999px;
            font-size:.72rem; font-weight:600; background:var(--tint); color:var(--accent-600);
            border:1px solid #DCE2DA; letter-spacing:.01em; }}

        /* Buttons */
        .stButton>button, .stFormSubmitButton>button, [data-testid="stFormSubmitButton"] button {{
            border-radius:11px; font-weight:500; font-size:.9rem; padding:.5rem 1.1rem;
            border:1px solid var(--border); background:var(--surface); color:var(--text);
            transition:all .18s ease; box-shadow:none; }}
        .stButton>button:hover, .stFormSubmitButton>button:hover {{ border-color:var(--accent); color:var(--accent); background:var(--surface_2); }}
        .stButton>button[kind="primary"], .stFormSubmitButton>button[kind="primary"],
        button[data-testid="baseButton-primary"] {{
            background:var(--accent); border:1px solid var(--accent); color:#F4F3EF; }}
        .stButton>button[kind="primary"]:hover, button[data-testid="baseButton-primary"]:hover {{
            background:var(--accent-600); border-color:var(--accent-600); color:#fff; }}

        /* Inputs — explicit colors fix white-on-white */
        [data-baseweb="input"], [data-baseweb="select"] > div, [data-baseweb="textarea"], .stTextArea textarea,
        .stNumberInput input, .stTextInput input, .stDateInput input {{
            background:var(--surface) !important; color:var(--text) !important;
            border-radius:11px !important; }}
        [data-baseweb="input"], [data-baseweb="select"] > div, [data-baseweb="textarea"] {{
            border:1px solid var(--border) !important; }}
        [data-baseweb="input"]:focus-within, [data-baseweb="select"] > div:focus-within,
        [data-baseweb="textarea"]:focus-within {{ border-color:var(--accent) !important; box-shadow:0 0 0 3px var(--tint) !important; }}
        input, textarea {{ color:var(--text) !important; -webkit-text-fill-color:var(--text) !important; }}
        [data-baseweb="select"] * {{ color:var(--text) !important; }}
        [data-testid="stWidgetLabel"] p, label, .stSelectbox label, .stSlider label {{
            color:var(--muted) !important; font-size:.82rem !important; font-weight:500 !important; }}
        [data-baseweb="menu"], [role="listbox"] {{ background:var(--surface) !important; border:1px solid var(--border) !important; }}
        [role="option"] {{ color:var(--text) !important; }}
        [role="option"]:hover {{ background:var(--tint) !important; }}

        /* Sliders */
        [data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {{ background:var(--accent) !important; }}
        [data-testid="stSlider"] [data-testid="stTickBar"] {{ background:transparent; }}

        /* Metrics */
        [data-testid="stMetric"] {{ background:var(--surface); border:1px solid var(--border);
            border-radius:14px; padding:1rem 1.1rem; }}
        [data-testid="stMetricValue"] {{ color:var(--text); font-weight:600; letter-spacing:-.02em; }}
        [data-testid="stMetricLabel"] p {{ color:var(--muted) !important; font-weight:500; }}

        /* Alerts — quiet, on-palette */
        [data-testid="stAlert"] {{ border-radius:12px; border:1px solid var(--border);
            background:var(--surface_2); color:var(--text); }}
        [data-testid="stAlert"] p {{ color:var(--text) !important; }}

        /* Expander */
        [data-testid="stExpander"] {{ border:1px solid var(--border); border-radius:14px; background:var(--surface); }}
        [data-testid="stExpander"] summary {{ color:var(--text); font-weight:500; }}
        [data-testid="stExpander"] summary:hover {{ color:var(--accent); }}

        /* Checkbox */
        [data-testid="stCheckbox"] label span {{ color:var(--text) !important; }}
        [data-baseweb="checkbox"] [data-checked="true"] {{ background:var(--accent) !important; border-color:var(--accent) !important; }}

        /* Divider */
        hr, [data-testid="stDivider"] {{ border-color:var(--border) !important; }}

        /* Inline code */
        code {{ background:var(--tint); color:var(--accent-600); border-radius:6px; padding:.1rem .35rem; font-size:.85em; }}

        /* Progress widget (st.progress) */
        [data-testid="stProgress"] > div > div > div {{ background:var(--accent) !important; }}
        [data-testid="stProgress"] > div > div {{ background:var(--tint) !important; }}

        /* Sidebar */
        section[data-testid="stSidebar"] {{ background:var(--surface); border-right:1px solid var(--border); }}
        section[data-testid="stSidebar"] .block-container {{ padding:2rem 1.35rem; }}
        .mm-brand {{ display:flex; align-items:center; gap:.6rem; margin-bottom:.15rem; }}
        .mm-brand-name {{ font-size:1.15rem; font-weight:600; letter-spacing:-.02em; color:var(--text); }}
        .mm-brand-ic {{ color:var(--accent); display:flex; }}
        .mm-stat {{ display:flex; align-items:baseline; justify-content:space-between; padding:.55rem 0; border-bottom:1px solid var(--border); }}
        .mm-stat:last-child {{ border-bottom:0; }}
        .mm-stat-k {{ color:var(--muted); font-size:.82rem; }}
        .mm-stat-v {{ color:var(--text); font-weight:600; font-size:.95rem; }}

        /* Chat input */
        [data-testid="stChatInput"] {{ border-radius:14px; }}

        /* Scrollbar */
        ::-webkit-scrollbar {{ width:9px; height:9px; }}
        ::-webkit-scrollbar-thumb {{ background:#D6D4CB; border-radius:999px; }}
        ::-webkit-scrollbar-thumb:hover {{ background:var(--faint); }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(icon_name: str, title: str, subtitle: str = "") -> None:
    """A consistent premium page header: icon chip + title + optional subtitle."""
    sub = f'<p class="mm-head-sub">{subtitle}</p>' if subtitle else ""
    st.markdown(
        f'<div class="mm-head"><div class="mm-head-ic">{icon(icon_name, 22)}</div>'
        f'<div><h1>{title}</h1>{sub}</div></div><hr class="mm-rule">',
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Page setup + backend bootstrap
# --------------------------------------------------------------------------- #
def setup_page(title: str) -> None:
    """Call at the top of every page: config + CSS + session init + backend."""
    st.set_page_config(
        page_title=f"Macromancer · {title}",
        page_icon="◍",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()
    init_state()
    _backend_ready()


@st.cache_resource(show_spinner="Preparing your workspace… (first load can take ~30s)")
def _backend_ready() -> bool:
    """Use an already-running backend if reachable; otherwise embed our own."""
    if api.health():
        return True
    return _ensure_embedded_backend()


def _ensure_embedded_backend() -> bool:
    """Run the FastAPI backend in a daemon thread, in this same process."""
    import sys
    import threading
    import time

    os.environ.setdefault("SEED_DEMO", "true")
    os.environ.setdefault("DATABASE_URL", "sqlite:////tmp/macromentor.db")
    os.environ.setdefault("RL_BANDIT_PATH", "/tmp/rl_bandit_weights.json")
    os.environ.setdefault("DISABLE_RATELIMIT", "1")

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if root not in sys.path:
        sys.path.insert(0, root)

    import asyncio
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    import uvicorn
    from backend.main import app as backend_app

    class _Server(uvicorn.Server):
        def install_signal_handlers(self) -> None:
            pass

    server = _Server(
        uvicorn.Config(backend_app, host="127.0.0.1", port=8000, log_level="warning")
    )
    threading.Thread(target=server.run, daemon=True).start()

    for _ in range(120):
        if api.health():
            return True
        time.sleep(0.5)
    return False


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
def init_state() -> None:
    ss = st.session_state
    ss.setdefault("user_id", 1)
    ss.setdefault("chat_history", [])
    ss.setdefault("chat_session_id", None)
    ss.setdefault("today_meals", [])
    ss.setdefault("grocery_list_ids", [])
    ss.setdefault("weight_history", [])
    ss.setdefault("last_plan_foods", [])


def add_logged_meal(meal: Dict) -> None:
    """Optimistic local echo of a just-logged meal (source of truth is the API)."""
    st.session_state.today_meals.append(meal)


def todays_meals(user_id: int) -> List[Dict]:
    try:
        return api.list_meals(user_id, today_str())
    except api.APIError:
        return []


def today_consumed(user_id: int) -> Dict[str, float]:
    totals = {"protein_g": 0.0, "carbs_g": 0.0, "fat_g": 0.0, "calories": 0.0}
    for m in todays_meals(user_id):
        for k in totals:
            totals[k] += float(m.get(k, 0) or 0)
    return totals


def meal_types() -> List[str]:
    return ["breakfast", "lunch", "dinner", "snack"]


def toast_error(exc: Exception) -> None:
    st.error(str(exc))


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def render_sidebar() -> int:
    """Render the shared sidebar (brand + user selector + quick stats)."""
    with st.sidebar:
        st.markdown(
            f'<div class="mm-brand"><span class="mm-brand-ic">{icon("leaf", 22, COLORS["accent"])}</span>'
            f'<span class="mm-brand-name">Macromancer</span></div>'
            f'<p class="mm-sub" style="margin:.1rem 0 1.3rem;">Precision nutrition, quietly intelligent.</p>',
            unsafe_allow_html=True,
        )

        if not api.health():
            st.warning("Backend is starting…")

        st.session_state.user_id = st.number_input(
            "Profile ID", min_value=1, step=1, value=int(st.session_state.user_id)
        )
        user_id = int(st.session_state.user_id)

        with st.expander("New profile"):
            _create_user_form()

        st.markdown('<hr class="mm-rule" style="margin:1.3rem 0;">', unsafe_allow_html=True)
        _quick_stats(user_id)
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
        if st.form_submit_button("Create profile", type="primary", use_container_width=True):
            try:
                res = api.create_user({
                    "age": age, "sex": sex, "weight_kg": weight, "height_cm": height,
                    "activity_level": activity, "goal": goal,
                })
                st.session_state.user_id = res["user"]["id"]
                st.success(f"Created profile {res['user']['id']}.")
                st.rerun()
            except api.APIError as exc:
                st.error(str(exc))


def _stat_row(label: str, value: str) -> str:
    return f'<div class="mm-stat"><span class="mm-stat-k">{label}</span><span class="mm-stat-v">{value}</span></div>'


def _quick_stats(user_id: int) -> None:
    st.markdown('<div class="mm-eyebrow" style="margin-bottom:.5rem;">Overview</div>', unsafe_allow_html=True)
    try:
        user = api.get_user(user_id)
    except api.APIError:
        st.info("Select or create a profile to see stats.")
        return
    rows = [
        _stat_row("Weight", f"{fmt(user.get('weight_kg'), 1)} kg"),
        _stat_row("Goal", str(user.get("goal", "—")).title()),
        _stat_row("Activity", str(user.get("activity_level", "—")).title()),
    ]
    try:
        tdee = api.get_tdee(user_id)
        val = tdee.get("adaptive_tdee") or tdee.get("current_tdee")
        rows.append(_stat_row("TDEE", f"{fmt(val)} kcal"))
    except api.APIError:
        pass
    st.markdown("".join(rows), unsafe_allow_html=True)
