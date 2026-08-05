"""In-process TTL caches for expensive read paths (thread-safe wrappers).

- ``food_search`` results: TTL 300s (5 min).
- ``optimize`` recommendations: TTL 60s, keyed by user + rounded macros so a user
  spamming refresh doesn't recompute the model each time.

Caches are best-effort and process-local; for multi-process deployments swap the
backing store for Redis. Writes invalidate the relevant entries (see
:func:`invalidate_user_optimize` and :func:`clear_food_search`).
"""

from __future__ import annotations

from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

from cachetools import TTLCache

# TTLCache is not thread-safe on its own; guard every access with a lock.
_lock = Lock()
_food_search_cache: TTLCache = TTLCache(maxsize=512, ttl=300)
_optimize_cache: TTLCache = TTLCache(maxsize=2048, ttl=60)
# Fuzzy dish->food match results; the food DB changes rarely, so cache a day.
_food_match_cache: TTLCache = TTLCache(maxsize=4096, ttl=86400)


# --------------------------------------------------------------------------- #
# Food search cache
# --------------------------------------------------------------------------- #
def food_search_key(search: str, skip: int, limit: int) -> Tuple:
    return ((search or "").strip().lower(), int(skip), int(limit))


def get_food_search(key: Tuple) -> Optional[List[dict]]:
    with _lock:
        return _food_search_cache.get(key)


def set_food_search(key: Tuple, value: List[dict]) -> None:
    with _lock:
        _food_search_cache[key] = value


def clear_food_search() -> None:
    """Invalidate all cached food searches (call when foods change)."""
    with _lock:
        _food_search_cache.clear()


# --------------------------------------------------------------------------- #
# Fuzzy food-match cache (dish name -> (food_id, score))
# --------------------------------------------------------------------------- #
def get_food_match(dish_name: str) -> Optional[Tuple[Optional[int], float]]:
    with _lock:
        return _food_match_cache.get((dish_name or "").strip().lower())


def set_food_match(dish_name: str, value: Tuple[Optional[int], float]) -> None:
    with _lock:
        _food_match_cache[(dish_name or "").strip().lower()] = value


def clear_food_match() -> None:
    """Invalidate cached fuzzy matches (call when foods change)."""
    with _lock:
        _food_match_cache.clear()


# --------------------------------------------------------------------------- #
# Optimize cache
# --------------------------------------------------------------------------- #
def optimize_key(
    user_id: int,
    meal_type: str,
    macros: Dict[str, float],
    available_food_ids: Optional[List[int]],
) -> Tuple:
    """Build a cache key; macros are rounded so trivial float jitter still hits."""
    macros_key = (
        round(float(macros.get("protein_g", 0.0)), 1),
        round(float(macros.get("carbs_g", 0.0)), 1),
        round(float(macros.get("fat_g", 0.0)), 1),
    )
    ids_key = tuple(sorted(available_food_ids)) if available_food_ids else None
    return (int(user_id), str(meal_type).lower(), macros_key, ids_key)


def get_optimize(key: Tuple) -> Optional[Any]:
    with _lock:
        return _optimize_cache.get(key)


def set_optimize(key: Tuple, value: Any) -> None:
    with _lock:
        _optimize_cache[key] = value


def invalidate_user_optimize(user_id: int) -> None:
    """Drop all cached optimize results for a user (call when they log a meal)."""
    with _lock:
        stale = [k for k in list(_optimize_cache.keys()) if k and k[0] == user_id]
        for k in stale:
            _optimize_cache.pop(k, None)
