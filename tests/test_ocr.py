"""Tests for the OCR service (Tesseract is mocked; no binary required)."""

from __future__ import annotations

import asyncio
import io

import pytest

from backend.services import ocr_service
from backend.services.ocr_service import (
    InvalidImageError,
    OCRUnavailableError,
    extract_text_from_image,
)


def _run(coro):
    return asyncio.run(coro)


def _png_bytes() -> bytes:
    """A tiny valid PNG so PIL can decode it before OCR runs."""
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (10, 10), "white").save(buf, format="PNG")
    return buf.getvalue()


def test_extract_text_mocked(monkeypatch):
    monkeypatch.setattr(
        ocr_service.pytesseract, "image_to_string", lambda img: "Grilled Chicken $9.99"
    )
    text = _run(extract_text_from_image(_png_bytes()))
    assert "Grilled Chicken" in text


def test_empty_upload_raises_invalid_image():
    with pytest.raises(InvalidImageError):
        _run(extract_text_from_image(b""))


def test_non_image_bytes_raises_invalid_image():
    with pytest.raises(InvalidImageError):
        _run(extract_text_from_image(b"this is not an image"))


def test_missing_tesseract_binary_raises_unavailable(monkeypatch):
    class _NotFound(Exception):
        pass

    # Simulate the Tesseract engine not being installed.
    monkeypatch.setattr(ocr_service.pytesseract, "TesseractNotFoundError", _NotFound)

    def _boom(img):
        raise _NotFound("tesseract is not installed")

    monkeypatch.setattr(ocr_service.pytesseract, "image_to_string", _boom)
    with pytest.raises(OCRUnavailableError):
        _run(extract_text_from_image(_png_bytes()))
