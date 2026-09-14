# Model Card — Macromancer Food Recommender

A concise model card for the XGBoost recommender at the core of Macromancer.
Generated results come from [`scripts/evaluate_recommender.py`](../scripts/evaluate_recommender.py).

## Model details
- **Type:** Gradient-boosted decision trees (`XGBoost`, `XGBClassifier`,
  `binary:logistic`).
- **Output:** `P(relevant)` for a `(context, candidate food)` pair; foods are
  ranked by this probability. A live per-user feedback modifier (0.8–1.2) can
  scale the score at inference.
- **Version:** retrained whenever the feature schema changes; also retrainable
  online from real logs (`POST /train`).
- **Artifact:** `ml/xgboost_macro_model.json` (portable JSON).

## Intended use
- **In scope:** suggesting which foods best fit a user's *remaining* daily macro
  budget for their next meal, in a personal/educational nutrition app.
- **Out of scope:** medical, clinical, or eating-disorder contexts; precise
  calorie prescription; any use with real personal data on an unauthenticated
  public deployment. **Not medical advice.**

## Problem formulation
At a *decision point* (user context + macros already eaten today), a candidate
food (a 100 g serving) is **relevant** if adding it moves the day's macros toward
target *without overshooting* any macro by more than 20%. Binary
classification / ranking.

## Features (12)
| group | features |
|---|---|
| user | `goal` (cut/maintain/bulk), `activity_level` |
| time | `hour_of_day`, `day_of_week` |
| remaining budget | `protein_ratio`, `carbs_ratio`, `fat_ratio` (eaten ÷ target) |
| candidate food | `protein/100g`, `carbs/100g`, `fat/100g`, `calories/100g` |
| preference | `avg_feedback_score` (per-user/food mean; neutral 3.0 when unrated) |

## Training data
~6,000 simulated decision points: random user profiles (Mifflin–St Jeor
targets) × a partially-eaten day (fraction sampled across the full 0.2–0.98
range, including tight end-of-day budgets) × a candidate food from a diverse
per-100g pool. Labels via the relevance rule above. **Synthetic** — see
Limitations.

## Evaluation
Held-out decision points + a pool of **42 real foods the model never trained on**.
Baselines: `random`, `popularity` (context-free marginal relevance), and
`macro_fit` (a no-ML error-reduction heuristic). 800 points, bootstrap 95% CIs.

| Method | precision@5 | NDCG@5 | MAP |
|---|---|---|---|
| random | 0.815 [0.80, 0.83] | 0.819 | 0.833 |
| popularity | 0.955 [0.94, 0.97] | 0.959 | 0.952 |
| macro_fit (heuristic) | 0.911 [0.90, 0.92] | 0.932 | 0.918 |
| **XGBoost (ours)** | **0.982 [0.97, 0.99]** | **0.984** | **0.981** |

The model beats all baselines with non-overlapping CIs, chiefly by learning the
overshoot penalty the heuristic ignores.

## Limitations & caveats
- **Synthetic evaluation.** Contexts are simulated (foods are real). This is a
  controlled generalization test, not a user study; no claim of real-world
  behavior change.
- **Weakly context-dependent objective.** Base relevance rate is high (~0.82),
  so precision@5 saturates — NDCG/MAP are the cleaner signals.
- **Label is a proxy.** "Macro adherence without overshoot" ≠ health outcomes,
  taste, or budget. Feedback is incorporated but sparse.
- **Distribution shift.** Real logs differ from the synthetic generator; monitor
  and retrain (`POST /train`) before trusting on real data.
- **Fairness/coverage** were not formally audited.

## Ethical considerations
Handles health-adjacent data (weight, body fat, meals). The app stores it locally
and unencrypted with no auth by default — deploy responsibly (see the README
Limitations). Recommendations should never replace professional dietary advice.
