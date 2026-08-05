"""Pytest fixtures for Macromancer.

Configures an isolated temporary SQLite database and disables model init before
importing the app, so tests neither touch the real DB nor require the OpenMP
runtime (XGBoost) unless a test explicitly exercises the optimizer.
"""

from __future__ import annotations

import os
import tempfile

# Must be set BEFORE importing backend modules (config reads these at import).
_TMP_DIR = tempfile.mkdtemp(prefix="macromancer_test_")
os.environ["MACROMANCER_DB"] = os.path.join(_TMP_DIR, "test.db")
os.environ.setdefault("MACROMANCER_SKIP_MODEL_INIT", "1")
# Disable per-IP rate limiting so tests can hammer endpoints freely.
os.environ.setdefault("DISABLE_RATELIMIT", "1")
# Keep RL bandit weights out of the repo's data/ dir during tests.
os.environ.setdefault("RL_BANDIT_PATH", os.path.join(_TMP_DIR, "rl_bandit_weights.json"))

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from backend.db import SessionLocal, init_db  # noqa: E402
from backend.main import app  # noqa: E402


@pytest.fixture(scope="session")
def client():
    """A TestClient with the app's lifespan (DB init) active for the session."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def db_session():
    """A raw SQLAlchemy session against the temp DB (tables ensured)."""
    init_db()
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def xgboost_available() -> bool:
    """Whether XGBoost + its OpenMP runtime can be loaded in this environment."""
    try:
        from backend.services.optimizer import _import_xgboost

        _import_xgboost()
        return True
    except Exception:
        return False
