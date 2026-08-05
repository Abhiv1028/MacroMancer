"""Uvicorn wrapper to launch the Macromancer API.

Usage:
    python run.py
Environment:
    HOST (default 0.0.0.0), PORT (default 8000), RELOAD (default false)
"""

from __future__ import annotations

import os

import uvicorn


def main() -> None:
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("RELOAD", "false").lower() in ("1", "true", "yes")
    uvicorn.run("backend.main:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    main()
