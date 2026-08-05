"""OCR service: extract text from a receipt/menu image via Tesseract.

OCR is CPU-heavy and blocking, so it runs in a worker thread
(``asyncio.to_thread``) to keep the event loop responsive. Missing Python
packages or a missing Tesseract binary raise :class:`OCRUnavailableError`, which
the route layer maps to HTTP 503 with install instructions. Undecodable image
bytes raise :class:`InvalidImageError` (mapped to HTTP 400).
"""

from __future__ import annotations

import asyncio
import io

# Optional heavy deps: import lazily-tolerant so the app still boots without them.
try:
    import pytesseract
    from PIL import Image, UnidentifiedImageError

    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - env without OCR deps
    pytesseract = None  # type: ignore[assignment]
    Image = None  # type: ignore[assignment]
    UnidentifiedImageError = Exception  # type: ignore[assignment,misc]
    _IMPORT_ERROR = exc

INSTALL_HELP = (
    "OCR is unavailable. Install the Tesseract engine and Python bindings:\n"
    "  macOS:          brew install tesseract tesseract-lang\n"
    "  Debian/Ubuntu:  sudo apt-get install tesseract-ocr\n"
    "  Windows:        https://github.com/UB-Mannheim/tesseract/wiki\n"
    "  Python:         pip install pytesseract Pillow\n"
    "You can still log restaurant meals manually via "
    "POST /api/v1/restaurant/log_meal."
)


class OCRUnavailableError(RuntimeError):
    """Raised when OCR cannot run (missing pytesseract/Pillow or Tesseract)."""


class InvalidImageError(ValueError):
    """Raised when the uploaded bytes are not a decodable image."""


def _ocr_sync(image_bytes: bytes) -> str:
    """Blocking OCR: decode the image and run Tesseract. Runs in a thread."""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.load()
    except UnidentifiedImageError as exc:
        raise InvalidImageError("Uploaded file is not a valid image.") from exc

    try:
        return pytesseract.image_to_string(image)
    except pytesseract.TesseractNotFoundError as exc:  # type: ignore[union-attr]
        raise OCRUnavailableError(INSTALL_HELP) from exc


async def extract_text_from_image(image_bytes: bytes) -> str:
    """Extract text from an image (receipt/menu) using Tesseract OCR.

    Args:
        image_bytes: Raw bytes of the uploaded image.

    Returns:
        The extracted text (may be empty if the image has no legible text).

    Raises:
        OCRUnavailableError: pytesseract/Pillow not installed, or Tesseract
            binary not found.
        InvalidImageError: bytes are not a decodable image.
    """
    if pytesseract is None or Image is None:
        raise OCRUnavailableError(f"{INSTALL_HELP}\n\nImport error: {_IMPORT_ERROR}")
    if not image_bytes:
        raise InvalidImageError("Empty upload.")
    return await asyncio.to_thread(_ocr_sync, image_bytes)


def ocr_available() -> bool:
    """Whether OCR can run in this environment (deps + Tesseract binary)."""
    if pytesseract is None or Image is None:
        return False
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False
