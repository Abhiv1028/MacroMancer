"""Offline evaluation of the food recommender against baselines.

Task (ranking): at a decision point (a user's context + how much they've already
eaten today), rank a fixed pool of candidate foods for the next meal. A food is
*relevant* if eating a standard serving moves the day's macros toward target
without overshooting any macro by >20% -- the same notion the model is trained
to predict, but evaluated on **held-out** decision points and a pool of **42 real
foods** (the demo seed), which the model never saw during training.

Methods compared:
  - random      : shuffle (floor)
  - popularity  : rank by each food's marginal relevance rate (context-free)
  - macro_fit   : rank by remaining-macro error reduction (a strong, no-ML heuristic)
  - xgboost     : the trained model

Metrics (top-k, averaged over decision points, with bootstrap 95% CIs):
  precision@5, NDCG@5, hit-rate@5, MAP.

Usage:  python -m scripts.evaluate_recommender  [--points 800] [--seed 7]
Writes results to ml/eval_results.json (+ ml/eval_results.png if matplotlib is
installed).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.config import settings  # noqa: E402
from backend.services import optimizer  # noqa: E402
from backend.services.macro_calculator import calculate_macro_targets  # noqa: E402
from backend.services.optimizer import (  # noqa: E402
    FEATURE_COLUMNS,
    encode_goal,
    encode_meal_type,
    label_meal,
)

SERVING_G = 100.0            # candidate serving (matches per-100g features used in training)
K = 5                        # top-k for @k metrics
# Fraction of the day already eaten. Matches the training distribution so this is
# an honest held-out generalization test (unseen contexts + 42 real foods the
# model never trained on).
FRAC_LOW, FRAC_HIGH = 0.20, 0.98
NEUTRAL_FEEDBACK = 3.0
SEXES = ["male", "female"]
ACTIVITIES = ["sedentary", "light", "moderate", "very", "extra"]
GOALS = ["cut", "maintain", "bulk"]
MEALS = ["breakfast", "lunch", "dinner", "snack"]
MACROS = ("protein_g", "carbs_g", "fat_g")


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_candidate_pool() -> List[Dict]:
    """The 42 curated demo foods, as per-100g macro profiles."""
    path = settings.DATA_DIR / "seed_foods.json"
    with open(path, "r", encoding="utf-8") as fh:
        foods = json.load(fh)["foods"]
    pool = []
    for f in foods:
        p, c, ft = f["protein_per_100g"], f["carbs_per_100g"], f["fat_per_100g"]
        pool.append({
            "name": f["name"],
            "p100": p, "c100": c, "f100": ft,
            "cal100": p * 4 + c * 4 + ft * 9,
            # macros contributed by one SERVING_G serving
            "meal": {
                "protein_g": p * SERVING_G / 100.0,
                "carbs_g": c * SERVING_G / 100.0,
                "fat_g": ft * SERVING_G / 100.0,
            },
        })
    return pool


class _Target:
    def __init__(self, t):
        self.protein_g, self.carbs_g, self.fat_g = t.protein_g, t.carbs_g, t.fat_g


def make_decision_points(pool: List[Dict], n: int, rng: np.random.Generator) -> List[Dict]:
    """Sample held-out decision points, keeping only those with both classes."""
    points = []
    tries = 0
    while len(points) < n and tries < n * 20:
        tries += 1
        weight = float(rng.uniform(55, 110))
        height = float(rng.uniform(155, 195))
        age = int(rng.integers(18, 66))
        sex = SEXES[int(rng.integers(2))]
        activity = ACTIVITIES[int(rng.integers(5))]
        goal = GOALS[int(rng.integers(3))]
        t = calculate_macro_targets(weight, height, age, activity, goal, sex)
        target = {"protein_g": t.protein_g, "carbs_g": t.carbs_g, "fat_g": t.fat_g}

        # Realistic "what should I eat next" state: most of the day is already
        # eaten, so the remaining budget is roughly one meal. This is the regime
        # where overshoot matters and the right food depends on the context.
        frac = float(rng.uniform(FRAC_LOW, FRAC_HIGH))
        eaten = {
            m: max(target[m] * frac * float(rng.uniform(0.75, 1.25)), 0.0) for m in MACROS
        }
        labels = np.array(
            [label_meal(eaten, food["meal"], target) for food in pool], dtype=int
        )
        if not (1 <= labels.sum() <= len(pool) - 1):
            continue  # need a ranking signal (both relevant and irrelevant present)

        points.append({
            "goal": goal, "meal_type": MEALS[int(rng.integers(4))],
            "hour": int(rng.integers(0, 24)), "dow": int(rng.integers(0, 7)),
            "eaten": eaten, "target": target, "labels": labels,
        })
    return points


# --------------------------------------------------------------------------- #
# Scorers (higher score = ranked earlier)
# --------------------------------------------------------------------------- #
def score_random(points, pool, rng):
    return [rng.random(len(pool)) for _ in points]


def score_popularity(points, pool, rng):
    pop = np.mean([p["labels"] for p in points], axis=0)  # marginal relevance rate
    return [pop for _ in points]


def score_macro_fit(points, pool, rng):
    out = []
    for pt in points:
        eaten, target = pt["eaten"], pt["target"]
        pre = sum(abs(target[m] - eaten[m]) for m in MACROS)
        scores = []
        for food in pool:
            post = sum(abs(target[m] - (eaten[m] + food["meal"][m])) for m in MACROS)
            scores.append(pre - post)  # error reduction
        out.append(np.array(scores))
    return out


def _ratio(num, den):
    return num / den if den else 0.0


def score_xgboost(points, pool, rng):
    model = optimizer.ensure_model()
    out = []
    for pt in points:
        rows = []
        for food in pool:
            rows.append([
                encode_goal(pt["goal"]), encode_meal_type(pt["meal_type"]),
                pt["hour"], pt["dow"],
                _ratio(pt["eaten"]["protein_g"], pt["target"]["protein_g"]),
                _ratio(pt["eaten"]["carbs_g"], pt["target"]["carbs_g"]),
                _ratio(pt["eaten"]["fat_g"], pt["target"]["fat_g"]),
                food["p100"], food["c100"], food["f100"], food["cal100"],
                NEUTRAL_FEEDBACK,
            ])
        import pandas as pd
        X = pd.DataFrame(rows, columns=FEATURE_COLUMNS).astype(float)
        out.append(model.predict_proba(X)[:, 1])
    return out


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _precision_at_k(order, labels, k):
    top = order[:k]
    return labels[top].sum() / k


def _ndcg_at_k(order, labels, k):
    top = order[:k]
    dcg = sum(labels[idx] / np.log2(pos + 2) for pos, idx in enumerate(top))
    ideal = np.sort(labels)[::-1][:k]
    idcg = sum(r / np.log2(pos + 2) for pos, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def _hit_at_k(order, labels, k):
    return 1.0 if labels[order[:k]].sum() > 0 else 0.0


def _average_precision(order, labels):
    hits, cum = 0, 0.0
    total_rel = labels.sum()
    for pos, idx in enumerate(order):
        if labels[idx]:
            hits += 1
            cum += hits / (pos + 1)
    return cum / total_rel if total_rel > 0 else 0.0


def per_point_metrics(scores, points):
    """Return dict metric -> array of per-decision-point values."""
    rng = np.random.default_rng(0)
    prec, ndcg, hit, ap = [], [], [], []
    for s, pt in zip(scores, points):
        labels = pt["labels"]
        # argsort desc, breaking ties randomly for fairness across methods
        jitter = rng.random(len(s)) * 1e-9
        order = np.argsort(-(np.asarray(s) + jitter))
        prec.append(_precision_at_k(order, labels, K))
        ndcg.append(_ndcg_at_k(order, labels, K))
        hit.append(_hit_at_k(order, labels, K))
        ap.append(_average_precision(order, labels))
    return {
        f"precision@{K}": np.array(prec),
        f"ndcg@{K}": np.array(ndcg),
        f"hit@{K}": np.array(hit),
        "map": np.array(ap),
    }


def bootstrap_ci(values, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(values)
    means = [values[rng.integers(0, n, n)].mean() for _ in range(n_boot)]
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(values.mean()), float(lo), float(hi)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", type=int, default=800)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    pool = load_candidate_pool()
    points = make_decision_points(pool, args.points, rng)
    base_rate = float(np.mean([p["labels"].mean() for p in points]))
    print(f"Decision points: {len(points)} | candidates/pt: {len(pool)} | "
          f"avg relevant fraction: {base_rate:.3f}\n")

    methods = {
        "random": score_random,
        "popularity": score_popularity,
        "macro_fit": score_macro_fit,
        "xgboost": score_xgboost,
    }
    metric_names = [f"precision@{K}", f"ndcg@{K}", f"hit@{K}", "map"]
    results = {}
    for name, fn in methods.items():
        scores = fn(points, pool, np.random.default_rng(args.seed + 1))
        pm = per_point_metrics(scores, points)
        results[name] = {m: bootstrap_ci(pm[m]) for m in metric_names}

    # Console table.
    header = f"{'method':<12}" + "".join(f"{m:>16}" for m in metric_names)
    print(header)
    print("-" * len(header))
    for name in methods:
        row = f"{name:<12}"
        for m in metric_names:
            mean, lo, hi = results[name][m]
            row += f"{mean:>10.3f}        "[:16]
        print(row)

    # Markdown table (paste-ready for the README).
    print("\nMarkdown:\n")
    md = ["| Method | " + " | ".join(metric_names) + " |",
          "|" + "---|" * (len(metric_names) + 1)]
    for name in methods:
        cells = []
        for m in metric_names:
            mean, lo, hi = results[name][m]
            cells.append(f"{mean:.3f} [{lo:.3f}, {hi:.3f}]")
        md.append(f"| {name} | " + " | ".join(cells) + " |")
    print("\n".join(md))

    out = {
        "n_points": len(points), "pool_size": len(pool),
        "serving_g": SERVING_G, "k": K, "avg_relevant_fraction": base_rate,
        "results": results, "seed": args.seed,
    }
    settings.ML_DIR.mkdir(parents=True, exist_ok=True)
    with open(settings.ML_DIR / "eval_results.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nSaved -> {settings.ML_DIR / 'eval_results.json'}")

    _maybe_plot(results, metric_names)
    return 0


def _maybe_plot(results, metric_names):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed — skipping chart)")
        return
    methods = list(results)
    show = [f"precision@{K}", f"ndcg@{K}", "map"]
    x = np.arange(len(show))
    width = 0.2
    colors = {"random": "#95A5A6", "popularity": "#3498DB",
              "macro_fit": "#F39C12", "xgboost": "#2ECC71"}
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, name in enumerate(methods):
        means = [results[name][m][0] for m in show]
        errs = [[results[name][m][0] - results[name][m][1] for m in show],
                [results[name][m][2] - results[name][m][0] for m in show]]
        ax.bar(x + (i - 1.5) * width, means, width, yerr=errs, capsize=3,
               label=name, color=colors.get(name, "#777"))
    ax.set_xticks(x)
    ax.set_xticklabels(show)
    ax.set_ylabel("score")
    ax.set_title("Food recommender vs. baselines (held-out, 95% CI)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(settings.ML_DIR / "eval_results.png", dpi=130)
    print(f"Saved chart -> {settings.ML_DIR / 'eval_results.png'}")


if __name__ == "__main__":
    raise SystemExit(main())
