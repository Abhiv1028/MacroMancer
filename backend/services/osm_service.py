"""Nearby restaurant search via the free OpenStreetMap Overpass API.

Respects OSM usage policy with a client-side 1 request/second throttle. Returns
lightweight restaurant records; the route layer handles DB caching. Best-effort:
network/parse failures yield an empty list rather than raising.
"""

from __future__ import annotations

import asyncio
import time
from typing import Dict, List

import httpx

from backend.config import settings

# Minimum seconds between Overpass calls (OSM fair-use policy).
_MIN_INTERVAL_S = 1.0
_throttle_lock = asyncio.Lock()
_last_call_ts = 0.0


async def _throttle() -> None:
    """Enforce >= 1s between Overpass requests across concurrent callers."""
    global _last_call_ts
    async with _throttle_lock:
        elapsed = time.monotonic() - _last_call_ts
        if elapsed < _MIN_INTERVAL_S:
            await asyncio.sleep(_MIN_INTERVAL_S - elapsed)
        _last_call_ts = time.monotonic()


def _format_address(tags: Dict) -> str:
    """Build a readable address from OSM address tags (best-effort)."""
    if tags.get("addr:full"):
        return tags["addr:full"]
    parts = []
    if tags.get("addr:housenumber"):
        parts.append(tags["addr:housenumber"])
    if tags.get("addr:street"):
        parts.append(tags["addr:street"])
    street = " ".join(parts)
    city = tags.get("addr:city", "")
    return ", ".join(p for p in (street, city) if p)


async def search_nearby_restaurants(
    lat: float, lon: float, radius_m: int = None
) -> List[Dict]:
    """Query Overpass for named restaurants around a point.

    Returns a list of ``{osm_id, name, lat, lon, address, cuisine}``. Nodes
    without a name are filtered out.
    """
    if radius_m is None:
        radius_m = settings.NEARBY_SEARCH_RADIUS_DEFAULT

    query = (
        "[out:json][timeout:25];"
        f'(node["amenity"="restaurant"](around:{radius_m},{lat},{lon}););'
        "out body;"
    )

    await _throttle()
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(settings.OSM_OVERPASS_URL, data={"data": query})
            resp.raise_for_status()
            payload = resp.json()
    except (httpx.HTTPError, ValueError):
        return []

    restaurants: List[Dict] = []
    for element in payload.get("elements", []):
        tags = element.get("tags", {}) or {}
        name = tags.get("name")
        if not name:
            continue
        restaurants.append(
            {
                "osm_id": f"{element.get('type', 'node')}/{element.get('id')}",
                "name": name,
                "lat": element.get("lat"),
                "lon": element.get("lon"),
                "address": _format_address(tags),
                "cuisine": tags.get("cuisine"),
            }
        )
    return restaurants
