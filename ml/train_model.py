"""CLI entrypoint to train the XGBoost macro model.

Usage:
    python -m ml.train_model            # train on synthetic data (fresh)
    python -m ml.train_model --logs     # retrain from existing MealLog data

Saves the model to ``settings.MODEL_PATH`` (ml/xgboost_macro_model.json).
"""

from __future__ import annotations

import argparse

from backend.config import settings
from backend.db import init_db, session_scope
from backend.services import optimizer


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the macro optimizer model.")
    parser.add_argument(
        "--logs",
        action="store_true",
        help="Retrain from existing MealLog data (falls back to synthetic).",
    )
    args = parser.parse_args()

    init_db()
    if args.logs:
        with session_scope() as db:
            optimizer.retrain_from_logs(db)
    else:
        optimizer.train_on_synthetic()

    print(f"Model saved to {settings.MODEL_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
