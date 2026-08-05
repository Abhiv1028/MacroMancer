"""Parse raw OCR text from a receipt/menu into likely dish names."""

from __future__ import annotations

import re
from typing import Dict, List, Optional

# Price like $12.99, 12.99, 12,99, or a trailing "12.99".
_PRICE_RE = re.compile(r"(?<![\d.])\$?\s?(\d{1,4}[.,]\d{2})(?!\d)")
# Leading list markers / quantities: "2x", "1 ", "3.", bullets.
_LEADING_QTY_RE = re.compile(r"^\s*(\d+\s*[xX]\s*|\d+[.)]\s+|[-*•]\s*)")

# Non-dish lines commonly seen on receipts (matched case-insensitively).
_ARTIFACT_KEYWORDS = {
    "total", "subtotal", "sub total", "tax", "gratuity", "tip", "balance",
    "change", "cash", "credit", "debit", "card", "visa", "mastercard", "amex",
    "receipt", "invoice", "order", "table", "server", "cashier", "thank you",
    "thanks", "welcome", "phone", "tel", "fax", "www", "http", "email",
    "address", "street", "ave", "suite", "date", "time", "qty", "amount",
    "payment", "auth", "approved", "merchant", "terminal", "ref", "guest",
    "check", "tender", "due", "discount", "coupon", "loyalty", "points",
}
# Standalone number-only lines and long digit runs (phone numbers, ids).
_NUMERIC_LINE_RE = re.compile(r"^[\s\d.,:$%#*/()+-]+$")

_MIN_NAME_LEN = 3


def _extract_price(line: str) -> Optional[float]:
    """Return the first price found in the line, or None."""
    match = _PRICE_RE.search(line)
    if not match:
        return None
    try:
        return float(match.group(1).replace(",", "."))
    except ValueError:
        return None


def _clean_name(line: str) -> str:
    """Strip prices, leading quantities, and trailing punctuation from a line."""
    text = _PRICE_RE.sub("", line)
    text = _LEADING_QTY_RE.sub("", text)
    # Drop trailing dotted leaders / stray symbols.
    text = re.sub(r"[.\s]{2,}$", "", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip(" .-*:•\t")


def _is_artifact(line: str) -> bool:
    lowered = line.lower()
    return any(keyword in lowered for keyword in _ARTIFACT_KEYWORDS)


def parse_menu_items(raw_text: str) -> List[Dict]:
    """Split OCR text into likely dish names with optional prices.

    Heuristic: keep lines that either contain a price or are reasonably long
    words (not receipt boilerplate like TOTAL/TAX/SUBTOTAL, not pure numbers).

    Returns a list of ``{"name": str, "price": Optional[float]}`` with duplicate
    names removed (case-insensitive, first occurrence wins). Empty/blank input
    yields an empty list.
    """
    if not raw_text or not raw_text.strip():
        return []

    items: List[Dict] = []
    seen = set()
    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if not line or _NUMERIC_LINE_RE.match(line):
            continue
        if _is_artifact(line):
            continue

        price = _extract_price(line)
        name = _clean_name(line)

        # Keep if it has a price, or looks like a real dish name (has letters
        # and enough length). Reject if too short or has no letters left.
        if not re.search(r"[A-Za-z]", name):
            continue
        if price is None and len(name) < _MIN_NAME_LEN:
            continue

        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        items.append({"name": name, "price": price})

    return items
