"""CLI entrypoint: download, extract, parse, and load USDA Foundation Foods.

Usage:
    python -m data.download_usda [--force]

On failure to download, prints clear manual-download instructions.
"""

from __future__ import annotations

import argparse
import sys

from backend.db import init_db
from backend.services.usda_loader import USDALoadError, run_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Load USDA Foundation Foods.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if a cached zip exists.",
    )
    args = parser.parse_args()

    init_db()
    try:
        count = run_pipeline(force_download=args.force)
    except USDALoadError as exc:
        print("\n[ERROR] USDA load failed:\n", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        return 1
    print(f"\nDone. {count} foods available in the database.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
