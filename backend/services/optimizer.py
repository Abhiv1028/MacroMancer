"""XGBoost-based macro optimizer: training, persistence, and inference.

The model is a binary classifier predicting whether logging a given food (at a
sensible serving) is a "good" choice for the user's remaining macro budget.
Feature schema is shared between synthetic training, log-based retraining, and
inference so the vectors always line up.
"""

from __future__ import annotations

from datetime import date as date_cls, datetime
from typing import Dict, List, Optional

import pandas as pd
from sqlalchemy.orm import Session

from backend.config import settings
from backend.models import Food, MealLog, User
from backend.services import food_utils
from backend.services.macro_calculator import get_or_create_targets

# Substrings that indicate a missing OpenMP runtime in a native-load failure.
_OPENMP_ERROR_MARKERS = ("libomp", "libgomp", "openmp")

_OPENMP_HELP = (
    "XGBoost requires the OpenMP runtime, which is not installed:\n"
    "  macOS (Homebrew):  brew install libomp\n"
    "  Debian / Ubuntu:   apt-get install libomp-dev   (or: libgomp1)\n"
    "  Conda:             conda install -c conda-forge libomp\n"
    "No Homebrew on macOS? scikit-learn ships a compatible libomp; point the\n"
    "dynamic loader at it before launching, e.g.:\n"
    '  export DYLD_LIBRARY_PATH="$(python -c \'import os,sklearn;'
    'print(os.path.join(os.path.dirname(sklearn.__file__),\".dylibs\"))\')'
    ':$DYLD_LIBRARY_PATH"\n'
    "Then re-run. See the README (\"OpenMP / libomp\") for details."
)


def _import_xgboost():
    """Import and return the xgboost module.

    XGBoost links against the native OpenMP runtime (``libomp`` on macOS,
    ``libgomp`` on Linux). If that shared library is missing the import fails
    with a cryptic ``dlopen``/loader error; we translate it into a clear,
    actionable :class:`RuntimeError` instead of surfacing the raw message.
    """
    try:
        import xgboost as xgb

        return xgb
    except Exception as exc:  # ImportError, OSError, XGBoostError, ...
        message = str(exc).lower()
        if any(marker in message for marker in _OPENMP_ERROR_MARKERS):
            raise RuntimeError(f"{_OPENMP_HELP}\n\nOriginal error: {exc}") from exc
        raise

# --------------------------------------------------------------------------- #
# Feature schema & encoders (shared everywhere)
# --------------------------------------------------------------------------- #
GOAL_ENCODING = {"cut": 0, "maintain": 1, "bulk": 2}
MEAL_ENCODING = {"breakfast": 0, "lunch": 1, "dinner": 2, "snack": 3}

FEATURE_COLUMNS: List[str] = [
    "user_goal",
    "meal_type",
    "hour_of_day",
    "day_of_week",
    "current_protein_ratio",
    "current_carbs_ratio",
    "current_fat_ratio",
    "food_protein_per_100g",
    "food_carbs_per_100g",
    "food_fat_per_100g",
    "food_calories_per_100g",
    # Phase 3: mean historical feedback (enjoyment+satiety+energy)/3 for the
    # user/food, neutral 3.0 when unknown. Lets the model learn feedback prefs.
    "avg_feedback_score",
]
LABEL_COLUMN = "label"

# A meal may push a macro at most this fraction of target *over* the goal
# before it counts as a harmful overshoot.
OVERSHOOT_TOLERANCE = 0.20

MACROS = ("protein_g", "carbs_g", "fat_g")


def encode_goal(goal: str) -> int:
    return GOAL_ENCODING.get(str(goal).lower(), 1)


def encode_meal_type(meal_type: str) -> int:
    return MEAL_ENCODING.get(str(meal_type).lower(), 3)


def label_meal(
    eaten_before: Dict[str, float],
    meal: Dict[str, float],
    target: Dict[str, float],
) -> int:
    """Return 1 if the meal is a *good* choice for the remaining macro budget.

    A meal is good when it **reduces total remaining error** (sum of absolute
    gaps to target across protein/carbs/fat) *and* does not overshoot any single
    macro by more than :data:`OVERSHOOT_TOLERANCE` of that macro's target.

    Remaining for a macro is ``target - eaten``. This rewards any meal that moves
    you toward your goals -- including a solid high-protein breakfast when
    protein is still low -- and penalizes only meals that either move you away
    from target or blow past it. (The previous "remaining within 20% of target"
    rule perversely punished hitting a macro exactly, since remaining -> 0 then
    fell outside the 80-120% band.)
    """
    pre_error = 0.0
    post_error = 0.0
    for macro in MACROS:
        tgt = target.get(macro, 0.0)
        remaining_before = tgt - eaten_before.get(macro, 0.0)
        remaining_after = remaining_before - meal.get(macro, 0.0)
        pre_error += abs(remaining_before)
        post_error += abs(remaining_after)
        # remaining_after < 0 means we've exceeded target; cap the overshoot.
        if remaining_after < -OVERSHOOT_TOLERANCE * tgt:
            return 0
    return 1 if post_error < pre_error else 0


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train_from_dataframe(
    df: pd.DataFrame, continue_from_existing: bool = False
) -> "object":
    """Train (or continue-train) the XGBoost classifier and persist it.

    Returns the fitted model. Requires both classes to be present; if only one
    class exists the data is minimally augmented to keep XGBoost happy.
    """
    xgb = _import_xgboost()  # imported lazily so the app starts without training

    if df.empty:
        raise ValueError("Cannot train on an empty dataset.")

    missing = [c for c in FEATURE_COLUMNS + [LABEL_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(f"Training data missing columns: {missing}")

    X = df[FEATURE_COLUMNS].astype(float)
    y = df[LABEL_COLUMN].astype(int)

    if y.nunique() < 2:
        # Degenerate single-class data: flip one row so the classifier trains.
        y = y.copy()
        y.iloc[0] = 1 - int(y.iloc[0])

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=2,
        random_state=settings.RANDOM_SEED,
    )

    fit_kwargs = {}
    if continue_from_existing and settings.MODEL_PATH.exists():
        # Warm-start from the previously trained booster.
        prev = xgb.XGBClassifier()
        prev.load_model(str(settings.MODEL_PATH))
        fit_kwargs["xgb_model"] = prev.get_booster()

    model.fit(X, y, **fit_kwargs)

    settings.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(settings.MODEL_PATH))
    print(
        f"[optimizer] Trained on {len(df)} rows "
        f"(positives={int(y.sum())}); saved -> {settings.MODEL_PATH}"
    )
    return model


def train_on_synthetic() -> "object":
    """Generate synthetic data and train a fresh model."""
    # Imported here to avoid a circular import at module load time.
    from data.synthetic_data_generator import generate_training_dataframe

    df = generate_training_dataframe()
    return train_from_dataframe(df, continue_from_existing=False)


def _meal_feedback_score(db: Session, log: "MealLog", user_id: int) -> float:
    """Composite feedback for a logged meal (its own feedback, else user/food avg).

    Prefers feedback attached directly to this meal; otherwise falls back to the
    user's average feedback for that food, then a neutral 3.0.
    """
    from backend.models import Feedback
    from backend.services import feedback_adjuster

    fb = (
        db.query(Feedback)
        .filter(Feedback.meal_log_id == log.id)
        .order_by(Feedback.id.desc())
        .first()
    )
    if fb is not None:
        return round((fb.enjoyment + fb.satiety + fb.energy) / 3.0, 3)
    return feedback_adjuster.avg_feedback_score(db, user_id, log.food_id)


def build_training_data_from_logs(db: Session) -> pd.DataFrame:
    """Reconstruct feature rows from real MealLog history for retraining.

    For each user, meals are replayed in chronological order per day, tracking
    running consumed macros and labeling against that day's target. Each row
    includes the Phase-3 ``avg_feedback_score`` feature.
    """
    rows: List[dict] = []
    users = db.query(User).all()
    for user in users:
        logs = (
            db.query(MealLog)
            .filter(MealLog.user_id == user.id)
            .order_by(MealLog.timestamp.asc())
            .all()
        )
        if not logs:
            continue

        # Running consumed macros, reset per calendar day.
        running: Dict[str, float] = {"protein_g": 0.0, "carbs_g": 0.0, "fat_g": 0.0}
        current_day: Optional[date_cls] = None

        for log in logs:
            log_day = log.timestamp.date()
            if log_day != current_day:
                running = {"protein_g": 0.0, "carbs_g": 0.0, "fat_g": 0.0}
                current_day = log_day

            target = get_or_create_targets(db, user, log_day)
            target_d = {
                "protein_g": target.protein_g,
                "carbs_g": target.carbs_g,
                "fat_g": target.fat_g,
            }
            food = db.get(Food, log.food_id)
            if food is None:
                continue
            per100 = food_utils.macros_per_100g(food)
            meal = {
                "protein_g": log.protein_g,
                "carbs_g": log.carbs_g,
                "fat_g": log.fat_g,
            }

            rows.append(
                {
                    "user_goal": encode_goal(user.goal),
                    "meal_type": encode_meal_type(log.meal_type),
                    "hour_of_day": log.timestamp.hour,
                    "day_of_week": log.timestamp.weekday(),
                    "current_protein_ratio": _safe_ratio(
                        running["protein_g"], target_d["protein_g"]
                    ),
                    "current_carbs_ratio": _safe_ratio(
                        running["carbs_g"], target_d["carbs_g"]
                    ),
                    "current_fat_ratio": _safe_ratio(
                        running["fat_g"], target_d["fat_g"]
                    ),
                    "food_protein_per_100g": per100["protein_g"],
                    "food_carbs_per_100g": per100["carbs_g"],
                    "food_fat_per_100g": per100["fat_g"],
                    "food_calories_per_100g": per100["calories"],
                    "avg_feedback_score": _meal_feedback_score(db, log, user.id),
                    LABEL_COLUMN: label_meal(running, meal, target_d),
                }
            )

            for macro in ("protein_g", "carbs_g", "fat_g"):
                running[macro] += meal[macro]

    return pd.DataFrame(rows)


def retrain_from_logs(db: Session) -> Optional["object"]:
    """Retrain using MealLog data (warm-started). Falls back to synthetic."""
    df = build_training_data_from_logs(db)
    if df.empty or len(df) < 10:
        print("[optimizer] Not enough log data; retraining on synthetic data.")
        return train_on_synthetic()
    return train_from_dataframe(df, continue_from_existing=settings.MODEL_PATH.exists())


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
_MODEL_CACHE: Dict[str, object] = {}


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def load_model(force_reload: bool = False) -> Optional["object"]:
    """Load and cache the persisted model; returns None if not yet trained."""
    if not force_reload and "model" in _MODEL_CACHE:
        return _MODEL_CACHE["model"]
    if not settings.MODEL_PATH.exists():
        return None
    xgb = _import_xgboost()

    model = xgb.XGBClassifier()
    model.load_model(str(settings.MODEL_PATH))
    _MODEL_CACHE["model"] = model
    return model


def ensure_model() -> "object":
    """Return a loaded model, (re)training on synthetic data when needed.

    Retrains automatically if no model exists or if a persisted model's feature
    count no longer matches :data:`FEATURE_COLUMNS` -- e.g. after the Phase-3
    ``avg_feedback_score`` feature was added to a previously-trained model.
    """
    model = load_model()
    if model is not None:
        expected = len(FEATURE_COLUMNS)
        actual = getattr(model, "n_features_in_", expected)
        if actual != expected:
            print(
                f"[optimizer] Model feature count {actual} != expected {expected}; "
                "retraining on synthetic data."
            )
            model = None
    if model is None:
        print("[optimizer] Training model on synthetic data ...")
        model = train_on_synthetic()
        _MODEL_CACHE["model"] = model
    return model


def invalidate_cache() -> None:
    """Drop the cached model (call after retraining)."""
    _MODEL_CACHE.pop("model", None)


def _suggested_grams(food: Food) -> float:
    """Pick a reasonable serving size: a common portion else the default."""
    reasonable = [
        p.gram_weight for p in food.portions if 20.0 <= p.gram_weight <= 400.0
    ]
    if reasonable:
        return round(sorted(reasonable)[len(reasonable) // 2], 1)
    if food.default_grams and food.default_grams > 0:
        return float(food.default_grams)
    return 100.0


def score_candidates(
    db: Session,
    user: User,
    current_macros: Dict[str, float],
    meal_type: str,
    candidate_foods: List[Food],
    top_n: int = None,
    sort: bool = True,
) -> List[dict]:
    """Score candidate foods and return the top-N as recommendation dicts.

    The XGBoost probability is multiplied by a per-user feedback modifier
    (see :func:`feedback_adjuster.adjust_scores_with_feedback`) so historical
    enjoyment/satiety nudges the ranking without retraining.

    Each recommendation contains: food_id, name, predicted_score,
    macros_per_serving, suggested_grams.
    """
    # Imported here to avoid a circular import (feedback_adjuster is lightweight
    # but keeps the optimizer import graph clean).
    from backend.services import feedback_adjuster

    if top_n is None:
        top_n = settings.TOP_N_RECOMMENDATIONS
    if not candidate_foods:
        return []

    model = ensure_model()

    target = get_or_create_targets(db, user, date_cls.today())
    now = datetime.now()

    p_ratio = _safe_ratio(current_macros.get("protein_g", 0.0), target.protein_g)
    c_ratio = _safe_ratio(current_macros.get("carbs_g", 0.0), target.carbs_g)
    f_ratio = _safe_ratio(current_macros.get("fat_g", 0.0), target.fat_g)

    feature_rows: List[List[float]] = []
    meta: List[dict] = []
    for food in candidate_foods:
        per100 = food_utils.macros_per_100g(food)
        fb_score = feedback_adjuster.avg_feedback_score(db, user.id, food.id)
        feature_rows.append(
            [
                encode_goal(user.goal),
                encode_meal_type(meal_type),
                now.hour,
                now.weekday(),
                p_ratio,
                c_ratio,
                f_ratio,
                per100["protein_g"],
                per100["carbs_g"],
                per100["fat_g"],
                per100["calories"],
                fb_score,
            ]
        )
        grams = _suggested_grams(food)
        meta.append(
            {
                "food": food,
                "suggested_grams": grams,
                "macros_per_serving": food_utils.macros_for_grams(food, grams),
                "modifier": feedback_adjuster.adjust_scores_with_feedback(
                    db, food.id, user.id
                ),
            }
        )

    X = pd.DataFrame(feature_rows, columns=FEATURE_COLUMNS).astype(float)
    probs = model.predict_proba(X)[:, 1]

    scored = []
    for prob, m in zip(probs, meta):
        food = m["food"]
        adjusted = float(prob) * m["modifier"]
        scored.append(
            {
                "food_id": food.id,
                "name": food.name,
                "predicted_score": round(adjusted, 4),
                "base_score": round(float(prob), 4),
                "feedback_modifier": m["modifier"],
                "macros_per_serving": m["macros_per_serving"],
                "suggested_grams": m["suggested_grams"],
            }
        )

    if not sort:
        # Preserve input order (caller maps results back to its items by index).
        return scored
    scored.sort(key=lambda r: r["predicted_score"], reverse=True)
    return scored[:top_n]


def default_candidate_foods(db: Session) -> List[Food]:
    """Foods under the calorie ceiling -- the default optimizer candidate pool."""
    foods = db.query(Food).all()
    return [
        food
        for food in foods
        if food_utils.macros_per_100g(food)["calories"]
        <= settings.MAX_CANDIDATE_CALORIES_PER_100G
    ]


def _macro_proportions(protein_g: float, carbs_g: float, fat_g: float) -> tuple:
    """Return (protein, carb, fat) share of calories for a macro triple."""
    cals = protein_g * 4.0 + carbs_g * 4.0 + fat_g * 9.0
    if cals <= 0:
        return (0.0, 0.0, 0.0)
    return (protein_g * 4.0 / cals, carbs_g * 4.0 / cals, fat_g * 9.0 / cals)


def _macro_fit(rec_macros: Dict[str, float], target_macros: Dict[str, float]) -> float:
    """Similarity in [0,1] between a food's macro split and a target split.

    Compares calorie-share proportions (protein/carb/fat) via L1 distance, so a
    food whose macro composition matches the RL action's strategy scores ~1.0.
    """
    fp = _macro_proportions(
        rec_macros.get("protein_g", 0.0),
        rec_macros.get("carbs_g", 0.0),
        rec_macros.get("fat_g", 0.0),
    )
    tp = _macro_proportions(
        target_macros.get("protein_g", 0.0),
        target_macros.get("carbs_g", 0.0),
        target_macros.get("fat_g", 0.0),
    )
    l1 = sum(abs(a - b) for a, b in zip(fp, tp))  # in [0, 2]
    return max(0.0, 1.0 - 0.5 * l1)


def get_top_recommendations(
    db: Session,
    user: User,
    current_macros: Dict[str, float],
    meal_type: str,
    limit: int = None,
    target_macros: Optional[Dict[str, float]] = None,
) -> List[dict]:
    """Return the top food recommendations with flattened per-serving macros.

    Convenience wrapper over :func:`score_candidates` that (a) selects the
    default candidate pool and (b) flattens ``macros_per_serving`` into
    top-level ``protein_g``/``carbs_g``/``fat_g``/``calories`` keys, which is the
    shape the chat/meal-planning layer expects.

    Args:
        target_macros: Optional desired macro composition (e.g. from an RL
            action). When provided, the XGBoost base score is blended with each
            candidate's macro-split fit to ``target_macros`` so the ranking
            reflects the chosen strategy (higher-protein foods for
            ``high_protein``, etc.).

    Note:
        ``current_macros`` must be the macros **already consumed** today (not the
        remaining budget). The model was trained on consumed/target ratios, so
        passing remaining would invert the signal.
    """
    if limit is None:
        limit = settings.TOP_N_RECOMMENDATIONS

    candidates = default_candidate_foods(db)
    # When re-ranking by strategy fit, score the whole pool first.
    top_n = len(candidates) if target_macros else limit
    scored = score_candidates(
        db=db,
        user=user,
        current_macros=current_macros,
        meal_type=meal_type,
        candidate_foods=candidates,
        top_n=top_n,
    )

    if target_macros:
        for r in scored:
            fit = _macro_fit(r["macros_per_serving"], target_macros)
            # Blend base model score with strategy fit (keeps quality + steers).
            r["strategy_fit"] = round(fit, 4)
            r["combined_score"] = round(r["predicted_score"] * (0.5 + 0.5 * fit), 6)
        scored.sort(key=lambda r: r["combined_score"], reverse=True)
        scored = scored[:limit]

    recommendations: List[dict] = []
    for r in scored:
        macros = r["macros_per_serving"]
        recommendations.append(
            {
                "food_id": r["food_id"],
                "name": r["name"],
                "protein_g": macros["protein_g"],
                "carbs_g": macros["carbs_g"],
                "fat_g": macros["fat_g"],
                "calories": macros["calories"],
                "suggested_grams": r["suggested_grams"],
                "predicted_score": r["predicted_score"],
                "strategy_fit": r.get("strategy_fit"),
            }
        )
    return recommendations
