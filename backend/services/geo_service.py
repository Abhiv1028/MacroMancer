"""IP geolocation via the free ip-api.com service (no key required).

Results are cached per IP for an hour to stay well within the free rate limit.
Any failure falls back to a default city so the caller always gets coordinates.
"""

from __future__ import annotations

from threading import Lock
from typing import Dict, Optional

import httpx
from cachetools import TTLCache

from backend.config import settings

# Default location used when geolocation is unavailable (New York, NY).
DEFAULT_LOCATION: Dict = {
    "lat": 40.7128,
    "lon": -74.0060,
    "city": "New York",
    "region": "NY",
    "source": "default",
}

_lock = Lock()
_ip_cache: TTLCache = TTLCache(maxsize=1024, ttl=settings.IP_GEO_TTL)


async def get_location_from_ip(ip: Optional[str] = None) -> Dict:
    """Return approximate ``{lat, lon, city, region, source}`` for an IP.

    Args:
        ip: Client IP to look up; ``None`` lets ip-api use the request's IP.

    Returns:
        A location dict. Never raises -- on any error it returns
        :data:`DEFAULT_LOCATION`.
    """
    cache_key = ip or "_self"
    with _lock:
        cached = _ip_cache.get(cache_key)
    if cached is not None:
        return cached

    url = settings.IP_API_URL + (ip or "")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
        if data.get("status") == "success":
            location = {
                "lat": float(data["lat"]),
                "lon": float(data["lon"]),
                "city": data.get("city", ""),
                "region": data.get("region", "") or data.get("regionName", ""),
                "source": "ip-api",
            }
        else:
            location = dict(DEFAULT_LOCATION)
    except (httpx.HTTPError, KeyError, ValueError, TypeError):
        location = dict(DEFAULT_LOCATION)

    with _lock:
        _ip_cache[cache_key] = location
    return location


def location_label(location: Dict) -> str:
    """Human-readable 'City, Region' label for a location dict."""
    city = location.get("city") or ""
    region = location.get("region") or ""
    if city and region:
        return f"{city}, {region}"
    return city or region or "Unknown location"
