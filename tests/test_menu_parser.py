"""Tests for the receipt/menu text parser heuristics."""

from __future__ import annotations

from backend.services.menu_parser import parse_menu_items

SAMPLE_RECEIPT = """
The Green Fork
123 Main Street
Table 4    Server: Sam

Grilled Chicken Salad      $12.99
2x Salmon Bowl             24.50
Veggie Burger              10.00
Sparkling Water             3.50

SUBTOTAL                   50.99
TAX                         4.08
TOTAL                      55.07
VISA ************1234
THANK YOU!
"""


def test_parses_dish_names():
    items = parse_menu_items(SAMPLE_RECEIPT)
    names = [i["name"].lower() for i in items]
    assert any("grilled chicken salad" in n for n in names)
    assert any("salmon bowl" in n for n in names)
    assert any("veggie burger" in n for n in names)


def test_strips_receipt_artifacts():
    names = [i["name"].lower() for i in parse_menu_items(SAMPLE_RECEIPT)]
    for junk in ("total", "subtotal", "tax", "visa", "thank you"):
        assert not any(junk in n for n in names), junk


def test_extracts_prices():
    items = parse_menu_items(SAMPLE_RECEIPT)
    salad = next(i for i in items if "chicken salad" in i["name"].lower())
    assert salad["price"] == 12.99


def test_strips_leading_quantity():
    items = parse_menu_items("2x Salmon Bowl 24.50")
    assert items[0]["name"] == "Salmon Bowl"


def test_empty_input_returns_empty_list():
    assert parse_menu_items("") == []
    assert parse_menu_items("   \n  \n") == []


def test_numeric_and_short_lines_dropped():
    items = parse_menu_items("55.07\n$4.08\nOK\n1234567890")
    # "OK" is too short and priceless; numeric lines dropped.
    assert items == []


def test_deduplicates_names():
    items = parse_menu_items("Latte $4.00\nLatte $4.00\nMocha $5.00")
    names = [i["name"] for i in items]
    assert names.count("Latte") == 1
