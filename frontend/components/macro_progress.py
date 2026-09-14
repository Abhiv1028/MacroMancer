"""Macro progress cards + calorie hero (premium, single-palette)."""

from __future__ import annotations

import streamlit as st

from utils import helpers


def _remaining_label(consumed: float, target: float, unit: str) -> str:
    if consumed > target:
        return f"{helpers.fmt(consumed - target)} {unit} over"
    return f"{helpers.fmt(max(target - consumed, 0))} {unit} remaining"


def macro_bar(label: str, consumed: float, target: float, unit: str = "g", color: str = None) -> None:
    """A labeled macro progress card with a dot marker and thin track."""
    ratio = (consumed / target) if target else 0.0
    fill = min(ratio * 100.0, 100.0)
    over = ratio > 1.08
    col = color or helpers.ratio_color(ratio)
    grad = (f"linear-gradient(90deg, {helpers.COLORS['warn']}, #D19A6F)" if over
            else f"linear-gradient(90deg, {col}, {helpers.COLORS['accent_bright']})")
    st.markdown(
        f"""
        <div class="mm-card">
          <div class="mm-eyebrow"><span class="mm-dot" style="background:{col};"></span>{label}</div>
          <div class="mm-value" style="margin-top:.55rem;">{helpers.fmt(consumed)}<span class="mm-unit"> / {helpers.fmt(target)} {unit}</span></div>
          <div class="mm-track"><div class="mm-fill" style="width:{fill}%;background:{grad};"></div></div>
          <div class="mm-sub">{helpers.fmt(ratio * 100)}% &middot; {_remaining_label(consumed, target, unit)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def calorie_hero(consumed: float, target: float) -> None:
    """A wide hero card for the day's energy balance."""
    ratio = (consumed / target) if target else 0.0
    fill = min(ratio * 100.0, 100.0)
    over = ratio > 1.08
    grad = (f"linear-gradient(90deg, {helpers.COLORS['warn']}, #D19A6F)" if over
            else f"linear-gradient(90deg, {helpers.COLORS['accent']}, {helpers.COLORS['accent_bright']})")
    st.markdown(
        f"""
        <div class="mm-card" style="padding:1.6rem 1.7rem;">
          <div class="mm-eyebrow" style="display:flex;align-items:center;gap:.5rem;">
            {helpers.icon('flame', 15, helpers.COLORS['accent'])} Energy today
          </div>
          <div class="mm-value" style="font-size:2.6rem;margin-top:.5rem;">{helpers.fmt(consumed)}<span class="mm-unit" style="font-size:1.1rem;"> / {helpers.fmt(target)} kcal</span></div>
          <div class="mm-track" style="height:10px;"><div class="mm-fill" style="width:{fill}%;background:{grad};"></div></div>
          <div class="mm-sub">{_remaining_label(consumed, target, 'kcal')}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
