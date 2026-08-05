"""Infer which meal a chat message refers to."""

from __future__ import annotations

# Order matters only for stable behavior; keywords are mutually exclusive here.
_MEAL_KEYWORDS = ("breakfast", "lunch", "dinner", "snack")


def detect_meal_type(message: str, current_hour: int) -> str:
    """Detect the intended meal type from a message, falling back to the clock.

    Args:
        message: The user's natural-language message.
        current_hour: Current hour of day (0-23) used when no keyword is present.

    Returns:
        One of ``"breakfast"``, ``"lunch"``, ``"dinner"``, ``"snack"``.
    """
    text = (message or "").lower()
    for keyword in _MEAL_KEYWORDS:
        if keyword in text:
            return keyword

    # Time-of-day fallback.
    if 6 <= current_hour < 11:
        return "breakfast"
    if 11 <= current_hour < 15:
        return "lunch"
    if 15 <= current_hour < 20:
        return "dinner"
    return "snack"
