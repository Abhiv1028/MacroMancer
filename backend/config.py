"""Central configuration for the Macromancer backend.

Values can be overridden via environment variables (loaded from a ``.env`` file
if present). Paths are resolved relative to the project root so the app behaves
the same regardless of the current working directory.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Project root is two levels up from this file: <root>/backend/config.py
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# Load environment variables from <root>/.env if it exists.
load_dotenv(PROJECT_ROOT / ".env")


class Settings:
    """Runtime settings resolved from environment variables with sane defaults."""

    # --- Filesystem layout -------------------------------------------------
    PROJECT_ROOT: Path = PROJECT_ROOT
    DATA_DIR: Path = PROJECT_ROOT / "data"
    ML_DIR: Path = PROJECT_ROOT / "ml"

    # SQLite database file (spec-mandated name).
    DB_FILE: Path = Path(os.getenv("MACROMANCER_DB", str(PROJECT_ROOT / "macromentor.db")))
    DATABASE_URL: str = os.getenv("DATABASE_URL", f"sqlite:///{DB_FILE}")

    # --- USDA pipeline -----------------------------------------------------
    # Primary download URL for the Foundation Foods CSV bundle.
    USDA_CSV_URL: str = os.getenv(
        "USDA_CSV_URL",
        "https://fdc.nal.usda.gov/fdc-datasets/"
        "FoodData_Central_foundation_food_csv_2024-10-31.zip",
    )
    # Page scraped to auto-discover a working download link if the primary fails.
    USDA_DATASETS_PAGE: str = os.getenv(
        "USDA_DATASETS_PAGE",
        "https://fdc.nal.usda.gov/download-datasets",
    )
    USDA_ZIP_PATH: Path = DATA_DIR / "usda_foundation_food.zip"
    USDA_EXTRACT_DIR: Path = DATA_DIR / "usda_csv"

    # --- Machine learning --------------------------------------------------
    MODEL_PATH: Path = Path(
        os.getenv("MODEL_PATH", str(ML_DIR / "xgboost_macro_model.json"))
    )

    # --- Synthetic data generation ---------------------------------------
    SYNTH_USERS: int = int(os.getenv("SYNTH_USERS", "50"))
    SYNTH_DAYS: int = int(os.getenv("SYNTH_DAYS", "30"))
    RANDOM_SEED: int = int(os.getenv("RANDOM_SEED", "42"))

    # Skip auto-training/loading the model on app startup (useful for tests or
    # environments without the OpenMP runtime). Set MACROMANCER_SKIP_MODEL_INIT=1.
    SKIP_MODEL_INIT: bool = os.getenv("MACROMANCER_SKIP_MODEL_INIT", "").lower() in (
        "1",
        "true",
        "yes",
    )

    # --- Optimizer ---------------------------------------------------------
    # Foods above this many calories per 100g are excluded from the default
    # candidate pool for practicality.
    MAX_CANDIDATE_CALORIES_PER_100G: float = float(
        os.getenv("MAX_CANDIDATE_CALORIES_PER_100G", "800")
    )
    TOP_N_RECOMMENDATIONS: int = int(os.getenv("TOP_N_RECOMMENDATIONS", "5"))

    # --- Reinforcement learning (Phase 6) ---------------------------------
    RL_BANDIT_PATH: Path = Path(
        os.getenv("RL_BANDIT_PATH", str(DATA_DIR / "rl_bandit_weights.json"))
    )
    # LinUCB exploration coefficient.
    RL_BANDIT_ALPHA: float = float(os.getenv("RL_BANDIT_ALPHA", "1.0"))

    # --- Nearby restaurants (Phase 7, free API stack) ---------------------
    NUTRITIONIX_APP_ID: str = os.getenv("NUTRITIONIX_APP_ID", "")
    NUTRITIONIX_API_KEY: str = os.getenv("NUTRITIONIX_API_KEY", "")
    NUTRITIONIX_BASE_URL: str = os.getenv(
        "NUTRITIONIX_BASE_URL", "https://trackapi.nutritionix.com/v2/search/instant"
    )
    OSM_OVERPASS_URL: str = os.getenv(
        "OSM_OVERPASS_URL", "https://overpass-api.de/api/interpreter"
    )
    IP_API_URL: str = os.getenv("IP_API_URL", "http://ip-api.com/json/")
    NEARBY_SEARCH_RADIUS_DEFAULT: int = int(
        os.getenv("NEARBY_SEARCH_RADIUS_DEFAULT", "2000")
    )
    NEARBY_MAX_RESTAURANTS: int = int(os.getenv("NEARBY_MAX_RESTAURANTS", "10"))
    NEARBY_MAX_ITEMS_PER_RESTAURANT: int = int(
        os.getenv("NEARBY_MAX_ITEMS_PER_RESTAURANT", "5")
    )
    # Cache TTLs (seconds): nearby search 1h, restaurant menu 7d, IP geo 1h.
    NEARBY_SEARCH_TTL: int = int(os.getenv("NEARBY_SEARCH_TTL", "3600"))
    RESTAURANT_MENU_TTL: int = int(os.getenv("RESTAURANT_MENU_TTL", str(7 * 86400)))
    IP_GEO_TTL: int = int(os.getenv("IP_GEO_TTL", "3600"))

    # --- LLM (Ollama) ------------------------------------------------------
    # Local Ollama server used for meal-plan generation (free, offline).
    OLLAMA_BASE_URL: str = os.getenv(
        "OLLAMA_BASE_URL", "http://localhost:11434/api/generate"
    )
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    OLLAMA_TIMEOUT_SECONDS: float = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "60"))
    OLLAMA_MAX_TOKENS: int = int(os.getenv("OLLAMA_MAX_TOKENS", "800"))
    OLLAMA_TEMPERATURE: float = float(os.getenv("OLLAMA_TEMPERATURE", "0.3"))


settings = Settings()

# Ensure key directories exist at import time.
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.ML_DIR.mkdir(parents=True, exist_ok=True)
