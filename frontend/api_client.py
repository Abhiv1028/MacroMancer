"""Thin wrapper around the Macromancer backend API.

Notes on base paths: some routers are mounted at the root (`/users`, `/foods`,
`/meals`, `/optimize`) and some under `/api/v1` (`/chat`, `/feedback`,
`/body_composition`, `/nearby`, `/grocery_lists`, `/users/{id}/tdee`, ...). This
client uses the correct base per call. Set ``MACROMANCER_API_URL`` to point at a
non-local backend.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import requests
import streamlit as st

BASE = os.getenv("MACROMANCER_API_URL", "http://localhost:8000").rstrip("/")
API = f"{BASE}/api/v1"
DEFAULT_TIMEOUT = 30
CHAT_TIMEOUT = 120  # Ollama can be slow


class APIError(Exception):
    """Raised for any backend/network failure, with a friendly message."""


# --------------------------------------------------------------------------- #
# Low-level request helpers
# --------------------------------------------------------------------------- #
def _handle(resp: requests.Response):
    if resp.status_code >= 400:
        detail = resp.text
        try:
            body = resp.json()
            detail = body.get("detail", body)
        except ValueError:
            pass
        raise APIError(f"[{resp.status_code}] {detail}")
    return resp.json() if resp.content else {}


def _request(method: str, url: str, timeout: int = DEFAULT_TIMEOUT, **kwargs):
    try:
        resp = requests.request(method, url, timeout=timeout, **kwargs)
    except requests.ConnectionError:
        raise APIError(
            f"Cannot connect to the backend at {BASE}. Is it running "
            "(`python run.py`)?"
        )
    except requests.Timeout:
        raise APIError("The request timed out. The backend may be busy.")
    except requests.RequestException as exc:  # pragma: no cover
        raise APIError(f"Request failed: {exc}")
    return _handle(resp)


def health() -> bool:
    """Return True if the backend health endpoint responds OK."""
    try:
        requests.get(f"{BASE}/health", timeout=5).raise_for_status()
        return True
    except requests.RequestException:
        return False


# --------------------------------------------------------------------------- #
# Users / targets / TDEE
# --------------------------------------------------------------------------- #
def create_user(payload: Dict) -> Dict:
    return _request("POST", f"{BASE}/users", json=payload)


@st.cache_data(ttl=30, show_spinner=False)
def get_user(user_id: int) -> Dict:
    return _request("GET", f"{BASE}/users/{user_id}")


@st.cache_data(ttl=30, show_spinner=False)
def get_targets(user_id: int, date: Optional[str] = None) -> Dict:
    """Return the target row for a date (endpoint returns a list; take first)."""
    rows = _request("GET", f"{BASE}/users/{user_id}/targets", params={"date": date})
    if isinstance(rows, list):
        return rows[0] if rows else {}
    return rows


@st.cache_data(ttl=30, show_spinner=False)
def get_tdee(user_id: int) -> Dict:
    return _request("GET", f"{API}/users/{user_id}/tdee")


def update_goals(user_id: int, goal: str) -> Dict:
    return _request("PUT", f"{API}/users/{user_id}/goals", json={"goal": goal})


# --------------------------------------------------------------------------- #
# Foods / meals / optimize
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=300, show_spinner=False)
def search_foods(query: str, limit: int = 25) -> List[Dict]:
    return _request("GET", f"{BASE}/foods", params={"search": query, "limit": limit})


def add_food(payload: Dict) -> Dict:
    return _request("POST", f"{BASE}/foods", json=payload)


def log_meal(user_id: int, food_id: int, grams: float, meal_type: str) -> Dict:
    return _request(
        "POST",
        f"{BASE}/meals",
        json={
            "user_id": user_id,
            "food_id": food_id,
            "grams": grams,
            "meal_type": meal_type,
        },
    )


def optimize(
    user_id: int,
    current_macros: Dict,
    meal_type: str,
    available_food_ids: Optional[List[int]] = None,
) -> Dict:
    return _request(
        "POST",
        f"{BASE}/optimize",
        json={
            "user_id": user_id,
            "current_macros": current_macros,
            "meal_type": meal_type,
            "available_food_ids": available_food_ids,
        },
    )


# --------------------------------------------------------------------------- #
# Chat
# --------------------------------------------------------------------------- #
def chat(user_id: int, message: str, session_id: Optional[int] = None) -> Dict:
    return _request(
        "POST",
        f"{API}/chat",
        timeout=CHAT_TIMEOUT,
        json={"user_id": user_id, "message": message, "session_id": session_id},
    )


# --------------------------------------------------------------------------- #
# Feedback / body composition
# --------------------------------------------------------------------------- #
def submit_feedback(payload: Dict) -> Dict:
    return _request("POST", f"{API}/feedback", json=payload)


def log_body_composition(payload: Dict) -> Dict:
    return _request("POST", f"{API}/body_composition", json=payload)


# --------------------------------------------------------------------------- #
# Grocery
# --------------------------------------------------------------------------- #
def create_grocery_list(payload: Dict) -> Dict:
    return _request("POST", f"{API}/grocery_lists", json=payload)


def get_grocery_list(list_id: int) -> Dict:
    return _request("GET", f"{API}/grocery_lists/{list_id}")


def toggle_grocery_item(item_id: int) -> Dict:
    return _request("PUT", f"{API}/grocery_items/{item_id}")


# --------------------------------------------------------------------------- #
# Nearby restaurants
# --------------------------------------------------------------------------- #
def nearby_search(
    user_id: int,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    radius: int = 2000,
) -> Dict:
    params = {"user_id": user_id, "radius": radius}
    if lat is not None and lon is not None:
        params["lat"] = lat
        params["lon"] = lon
    return _request("GET", f"{API}/nearby/search", params=params)


def nearby_log(user_id: int, restaurant_name: str, item_name: str, grams: float) -> Dict:
    return _request(
        "POST",
        f"{API}/nearby/log",
        json={
            "user_id": user_id,
            "restaurant_name": restaurant_name,
            "item_name": item_name,
            "grams": grams,
        },
    )
