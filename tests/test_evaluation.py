"""Guards the recommender evaluation harness (scripts/evaluate_recommender.py)."""

from __future__ import annotations

import numpy as np
import pytest

from scripts import evaluate_recommender as ev
from tests.conftest import xgboost_available

K = f"precision@{ev.K}"


def _points(n, seed):
    rng = np.random.default_rng(seed)
    pool = ev.load_candidate_pool()
    return pool, ev.make_decision_points(pool, n, rng), rng


def test_candidate_pool_loads():
    pool = ev.load_candidate_pool()
    assert len(pool) >= 40
    assert {"name", "p100", "c100", "f100", "meal"} <= set(pool[0])


def test_metrics_in_range_and_heuristic_beats_random():
    pool, pts, rng = _points(150, 1)
    assert len(pts) > 50
    mf = ev.per_point_metrics(ev.score_macro_fit(pts, pool, rng), pts)
    rnd = ev.per_point_metrics(ev.score_random(pts, pool, rng), pts)
    for metrics in (mf, rnd):
        for arr in metrics.values():
            assert (0.0 <= arr).all() and (arr <= 1.0).all()
    # A sensible heuristic should out-rank random.
    assert mf["ndcg@5"].mean() > rnd["ndcg@5"].mean()


def test_bootstrap_ci_orders_correctly():
    mean, lo, hi = ev.bootstrap_ci(np.array([0.9] * 100 + [0.1] * 100))
    assert lo <= mean <= hi


@pytest.mark.skipif(
    not xgboost_available(), reason="XGBoost/OpenMP runtime not available"
)
def test_xgboost_beats_random():
    pool, pts, rng = _points(150, 2)
    xgb = ev.per_point_metrics(ev.score_xgboost(pts, pool, rng), pts)
    rnd = ev.per_point_metrics(ev.score_random(pts, pool, rng), pts)
    assert xgb[K].mean() > rnd[K].mean()
    assert xgb["ndcg@5"].mean() > rnd["ndcg@5"].mean()
