"""USDA FoodData Central Foundation Foods pipeline.

Responsibilities:
    1. Download the Foundation Foods CSV bundle (zip), auto-discovering a working
       URL from the USDA website if the configured one fails.
    2. Extract and parse the relevant CSVs (food, nutrient, food_nutrient,
       food_portion, food_category).
    3. Normalize into the app schema and load into SQLite via SQLAlchemy.

Foundation Foods nutrient amounts are already expressed per 100 g, which maps
directly onto :attr:`Nutrient.amount_per_100g`.
"""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from sqlalchemy.orm import Session

from backend.config import settings
from backend.db import session_scope
from backend.models import Food, Nutrient, Portion

# Nutrient names (as they appear in USDA nutrient.csv) that we treat as the
# core macros. Keys map to canonical labels stored on the Nutrient rows.
MACRO_NUTRIENT_NAMES = {
    "Protein": "Protein",
    "Total lipid (fat)": "Fat",
    "Carbohydrate, by difference": "Carbohydrate",
    "Energy": "Energy",
    "Fiber, total dietary": "Fiber",
    "Sugars, total including NLEA": "Sugars",
}

_HTTP_HEADERS = {"User-Agent": "Macromancer/1.0 (+https://example.local)"}


class USDALoadError(RuntimeError):
    """Raised when the USDA dataset cannot be obtained or parsed."""


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #
def _discover_download_url() -> Optional[str]:
    """Scrape the USDA datasets page for a Foundation Foods CSV zip link."""
    try:
        resp = requests.get(
            settings.USDA_DATASETS_PAGE, headers=_HTTP_HEADERS, timeout=60
        )
        resp.raise_for_status()
    except requests.RequestException as exc:  # pragma: no cover - network
        print(f"[usda] Could not reach datasets page for discovery: {exc}")
        return None

    # Look for any href pointing to a foundation food csv zip.
    candidates = re.findall(
        r'href=["\']([^"\']*foundation_food[^"\']*\.zip)["\']',
        resp.text,
        flags=re.IGNORECASE,
    )
    if not candidates:
        return None

    url = candidates[0]
    if url.startswith("//"):
        url = "https:" + url
    elif url.startswith("/"):
        url = "https://fdc.nal.usda.gov" + url
    print(f"[usda] Auto-discovered download URL: {url}")
    return url


def download_usda_zip(force: bool = False) -> Path:
    """Download the Foundation Foods zip to ``settings.USDA_ZIP_PATH``.

    Tries the configured URL first, then an auto-discovered one. Raises
    :class:`USDALoadError` with manual-download instructions on total failure.
    """
    dest = settings.USDA_ZIP_PATH
    if dest.exists() and not force:
        print(f"[usda] Using cached zip at {dest}")
        return dest

    urls: List[str] = [settings.USDA_CSV_URL]
    discovered = _discover_download_url()
    if discovered and discovered not in urls:
        urls.append(discovered)

    last_error: Optional[Exception] = None
    for url in urls:
        try:
            print(f"[usda] Downloading {url} ...")
            with requests.get(
                url, headers=_HTTP_HEADERS, stream=True, timeout=300
            ) as resp:
                resp.raise_for_status()
                dest.parent.mkdir(parents=True, exist_ok=True)
                with open(dest, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        if chunk:
                            fh.write(chunk)
            # Validate it is actually a zip.
            if not zipfile.is_zipfile(dest):
                raise USDALoadError("Downloaded file is not a valid zip archive.")
            print(f"[usda] Saved zip to {dest} ({dest.stat().st_size} bytes)")
            return dest
        except (requests.RequestException, USDALoadError) as exc:
            last_error = exc
            print(f"[usda] Download failed for {url}: {exc}")

    raise USDALoadError(
        "Failed to download the USDA Foundation Foods CSV bundle.\n"
        "Please download it manually and place the extracted CSVs so that\n"
        f"  {settings.USDA_EXTRACT_DIR}\n"
        "contains food.csv, nutrient.csv, food_nutrient.csv, food_portion.csv,\n"
        "and food_category.csv.\n\n"
        "Get the bundle from: https://fdc.nal.usda.gov/download-datasets\n"
        "(choose 'Foundation Foods' -> CSV).\n"
        f"Last error: {last_error}"
    )


def extract_usda_zip(zip_path: Optional[Path] = None) -> Path:
    """Extract the zip into ``settings.USDA_EXTRACT_DIR`` and return that dir."""
    zip_path = zip_path or settings.USDA_ZIP_PATH
    extract_dir = settings.USDA_EXTRACT_DIR
    extract_dir.mkdir(parents=True, exist_ok=True)

    if not zip_path.exists():
        raise USDALoadError(f"Zip not found at {zip_path}; run download first.")

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)
    print(f"[usda] Extracted to {extract_dir}")
    return extract_dir


# --------------------------------------------------------------------------- #
# Parse
# --------------------------------------------------------------------------- #
def _find_csv(root: Path, filename: str) -> Optional[Path]:
    """Locate ``filename`` anywhere under ``root`` (USDA nests a subfolder)."""
    matches = list(root.rglob(filename))
    return matches[0] if matches else None


def _read_csv(path: Path, usecols: Optional[List[str]] = None) -> pd.DataFrame:
    """Read a USDA CSV, tolerating quoting/encoding quirks."""
    return pd.read_csv(
        path,
        usecols=lambda c: usecols is None or c in usecols,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8",
        low_memory=False,
    )


def parse_usda_csvs(extract_dir: Optional[Path] = None) -> Dict[str, pd.DataFrame]:
    """Load the relevant USDA CSVs into a dict of DataFrames.

    Returns keys: ``food``, ``nutrient``, ``food_nutrient``, ``food_portion``,
    ``food_category`` (the last two may be empty DataFrames if absent).
    """
    root = extract_dir or settings.USDA_EXTRACT_DIR
    required = {
        "food": "food.csv",
        "nutrient": "nutrient.csv",
        "food_nutrient": "food_nutrient.csv",
    }
    optional = {
        "food_portion": "food_portion.csv",
        "food_category": "food_category.csv",
    }

    frames: Dict[str, pd.DataFrame] = {}
    for key, fname in required.items():
        path = _find_csv(root, fname)
        if path is None:
            raise USDALoadError(
                f"Required file {fname} not found under {root}. "
                "Re-run the download/extract step."
            )
        frames[key] = _read_csv(path)

    for key, fname in optional.items():
        path = _find_csv(root, fname)
        frames[key] = _read_csv(path) if path is not None else pd.DataFrame()

    return frames


def _build_records(
    frames: Dict[str, pd.DataFrame]
) -> Tuple[List[dict], Dict[int, List[dict]], Dict[int, List[dict]]]:
    """Transform raw frames into food records + nutrient/portion maps by fdc_id."""
    food_df = frames["food"]
    nutrient_df = frames["nutrient"]
    food_nutrient_df = frames["food_nutrient"]
    portion_df = frames.get("food_portion", pd.DataFrame())
    category_df = frames.get("food_category", pd.DataFrame())

    # nutrient_id -> (name, unit)
    nutrient_lookup: Dict[str, Tuple[str, str]] = {}
    for _, row in nutrient_df.iterrows():
        nutrient_lookup[row["id"]] = (
            row.get("name", ""),
            row.get("unit_name", ""),
        )

    # category_id -> description
    category_lookup: Dict[str, str] = {}
    if not category_df.empty and "id" in category_df.columns:
        for _, row in category_df.iterrows():
            category_lookup[row["id"]] = row.get("description", "")

    # Foods (optionally restrict to foundation_food data type if present).
    foods: List[dict] = []
    for _, row in food_df.iterrows():
        data_type = row.get("data_type", "")
        if data_type and data_type not in ("foundation_food", "sr_legacy_food"):
            # Foundation bundle should only contain foundation_food, but be lenient.
            continue
        fdc_id = int(row["fdc_id"])
        cat_id = row.get("food_category_id", "")
        foods.append(
            {
                "fdc_id": fdc_id,
                "name": row.get("description", "").strip(),
                "category": category_lookup.get(cat_id, cat_id or None) or None,
            }
        )

    valid_ids = {f["fdc_id"] for f in foods}

    # Nutrients grouped by fdc_id, keeping only macro-relevant nutrients.
    nutrients_by_food: Dict[int, List[dict]] = {}
    for _, row in food_nutrient_df.iterrows():
        try:
            fdc_id = int(row["fdc_id"])
        except (ValueError, KeyError):
            continue
        if fdc_id not in valid_ids:
            continue
        name, unit = nutrient_lookup.get(row.get("nutrient_id", ""), ("", ""))
        if name not in MACRO_NUTRIENT_NAMES:
            continue
        # For Energy, keep only the kcal record.
        if name == "Energy" and unit.upper() not in ("KCAL", ""):
            continue
        try:
            amount = float(row.get("amount", "") or 0.0)
        except ValueError:
            amount = 0.0
        nutrients_by_food.setdefault(fdc_id, []).append(
            {
                "nutrient_name": MACRO_NUTRIENT_NAMES[name],
                "amount_per_100g": amount,
                "unit": "kcal" if name == "Energy" else (unit or "g"),
            }
        )

    # Portions grouped by fdc_id.
    portions_by_food: Dict[int, List[dict]] = {}
    if not portion_df.empty:
        for _, row in portion_df.iterrows():
            try:
                fdc_id = int(row["fdc_id"])
            except (ValueError, KeyError):
                continue
            if fdc_id not in valid_ids:
                continue
            try:
                grams = float(row.get("gram_weight", "") or 0.0)
            except ValueError:
                continue
            if grams <= 0:
                continue
            desc = (
                row.get("portion_description", "")
                or row.get("modifier", "")
                or "portion"
            ).strip()
            portions_by_food.setdefault(fdc_id, []).append(
                {"description": desc or "portion", "gram_weight": grams}
            )

    return foods, nutrients_by_food, portions_by_food


# --------------------------------------------------------------------------- #
# Load
# --------------------------------------------------------------------------- #
def load_into_db(
    frames: Dict[str, pd.DataFrame], db: Optional[Session] = None
) -> int:
    """Insert parsed USDA foods/nutrients/portions. Returns count of foods loaded.

    Idempotent per ``usda_food_id``: foods already present are skipped.
    """
    foods, nutrients_by_food, portions_by_food = _build_records(frames)

    def _load(session: Session) -> int:
        existing_ids = {
            fid
            for (fid,) in session.query(Food.usda_food_id)
            .filter(Food.usda_food_id.isnot(None))
            .all()
        }
        loaded = 0
        for f in foods:
            if f["fdc_id"] in existing_ids:
                continue
            food = Food(
                usda_food_id=f["fdc_id"],
                name=f["name"] or f"USDA food {f['fdc_id']}",
                category=f["category"],
                default_grams=100.0,
                is_custom=False,
            )
            for n in nutrients_by_food.get(f["fdc_id"], []):
                food.nutrients.append(Nutrient(**n))
            for p in portions_by_food.get(f["fdc_id"], []):
                food.portions.append(Portion(**p))
            session.add(food)
            loaded += 1
        session.flush()
        return loaded

    if db is not None:
        count = _load(db)
        db.commit()
        return count

    with session_scope() as session:
        return _load(session)


def run_pipeline(force_download: bool = False) -> int:
    """End-to-end: download -> extract -> parse -> load. Returns foods loaded."""
    zip_path = download_usda_zip(force=force_download)
    extract_dir = extract_usda_zip(zip_path)
    frames = parse_usda_csvs(extract_dir)
    count = load_into_db(frames)
    print(f"[usda] Loaded {count} foods into the database.")
    return count
