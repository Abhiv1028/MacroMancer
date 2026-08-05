"""Reusable macro progress bars and metric cards."""

from __future__ import annotations

import streamlit as st

from utils import helpers


def macro_bar(label: str, consumed: float, target: float, unit: str = "g", color: str = None) -> None:
    """Render a labeled horizontal macro progress bar inside a card."""
    ratio = (consumed / target) if target else 0.0
    fill = min(ratio * 100.0, 100.0)
    bar_color = color or helpers.ratio_color(ratio)
    remaining = max(target - consumed, 0)
    st.markdown(
        f"""
        <div class="mm-card">
          <div class="mm-metric-label">{label}</div>
          <div class="mm-metric-value" style="color:{bar_color};">
            {helpers.fmt(consumed)}<span style="font-size:1rem;color:{helpers.COLORS['muted']};"> / {helpers.fmt(target)} {unit}</span>
          </div>
          <div class="mm-bar-track"><div class="mm-bar-fill" style="width:{fill}%;background:{bar_color};"></div></div>
          <div class="mm-metric-sub">{helpers.fmt(ratio*100)}% · {helpers.fmt(remaining)} {unit} left</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def calorie_hero(consumed: float, target: float) -> None:
    """A larger hero card for the day's calories."""
    ratio = (consumed / target) if target else 0.0
    fill = min(ratio * 100.0, 100.0)
    color = helpers.ratio_color(ratio)
    st.markdown(
        f"""
        <div class="mm-card" style="border-left:6px solid {color};">
          <div class="mm-metric-label">Calories today</div>
          <div class="mm-metric-value" style="font-size:2.4rem;color:{color};">
            {helpers.fmt(consumed)}<span style="font-size:1.1rem;color:{helpers.COLORS['muted']};"> / {helpers.fmt(target)} kcal</span>
          </div>
          <div class="mm-bar-track" style="height:16px;"><div class="mm-bar-fill" style="width:{fill}%;background:{color};"></div></div>
          <div class="mm-metric-sub">{helpers.fmt(max(target-consumed,0))} kcal remaining</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
