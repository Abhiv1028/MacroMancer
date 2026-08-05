"""End-to-end smoke test for the Macromancer API.

Run the server first (``python run.py``), then:
    python test_api.py [--base-url http://localhost:8000]

Exercises: create user -> create custom foods -> log a meal -> optimize -> train.
Exits non-zero if any step fails.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime

import requests


def _check(resp: requests.Response, expected: int, label: str) -> dict:
    if resp.status_code != expected:
        print(f"[FAIL] {label}: {resp.status_code} -> {resp.text}")
        raise SystemExit(1)
    print(f"[ OK ] {label}")
    return resp.json() if resp.content else {}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")

    # Health check.
    _check(requests.get(f"{base}/health"), 200, "GET /health")

    # 1. Create a user.
    user_resp = _check(
        requests.post(
            f"{base}/users",
            json={
                "age": 30,
                "weight_kg": 75,
                "height_cm": 180,
                "activity_level": "moderate",
                "goal": "cut",
                "sex": "male",
                "dietary_restrictions": ["none"],
            },
        ),
        201,
        "POST /users",
    )
    user_id = user_resp["user"]["id"]
    print(f"       user_id={user_id}, targets={user_resp['targets']}")

    # 2. Get targets.
    _check(requests.get(f"{base}/users/{user_id}/targets"), 200, "GET /users/{id}/targets")

    # 3. Create custom foods (so the test works even without USDA data).
    foods = [
        {"name": "Grilled Chicken Breast", "protein_per_100g": 31, "carbs_per_100g": 0, "fat_per_100g": 3.6,
         "portions": [{"description": "1 breast", "gram_weight": 174}]},
        {"name": "White Rice, cooked", "protein_per_100g": 2.7, "carbs_per_100g": 28, "fat_per_100g": 0.3},
        {"name": "Broccoli, steamed", "protein_per_100g": 2.8, "carbs_per_100g": 7, "fat_per_100g": 0.4},
        {"name": "Olive Oil", "protein_per_100g": 0, "carbs_per_100g": 0, "fat_per_100g": 100},
        {"name": "Greek Yogurt, nonfat", "protein_per_100g": 10, "carbs_per_100g": 3.6, "fat_per_100g": 0.4},
    ]
    food_ids = []
    for f in foods:
        created = _check(requests.post(f"{base}/foods", json=f), 201, f"POST /foods ({f['name']})")
        food_ids.append(created["id"])

    # 4. Search foods.
    results = _check(requests.get(f"{base}/foods", params={"search": "chicken"}), 200, "GET /foods?search=chicken")
    print(f"       search returned {len(results)} result(s)")

    # 5. Log a meal.
    meal = _check(
        requests.post(
            f"{base}/meals",
            json={
                "user_id": user_id,
                "food_id": food_ids[0],
                "grams": 174,
                "meal_type": "lunch",
                "timestamp": datetime.now().isoformat(),
            },
        ),
        201,
        "POST /meals",
    )
    print(f"       logged meal: {meal['protein_g']}g P, {meal['carbs_g']}g C, "
          f"{meal['fat_g']}g F, {meal['calories']} kcal")

    # 6. Optimize.
    opt = _check(
        requests.post(
            f"{base}/optimize",
            json={
                "user_id": user_id,
                "current_macros": {
                    "protein_g": meal["protein_g"],
                    "carbs_g": meal["carbs_g"],
                    "fat_g": meal["fat_g"],
                },
                "meal_type": "dinner",
            },
        ),
        200,
        "POST /optimize",
    )
    recs = opt["recommendations"]
    print(f"       {len(recs)} recommendation(s):")
    for r in recs:
        print(f"         - {r['name']}: score={r['predicted_score']} "
              f"({r['suggested_grams']}g -> {r['macros_per_serving']})")

    # 7. Trigger training (synchronous for a deterministic test).
    _check(
        requests.post(f"{base}/train", params={"run_async": "false"}),
        200,
        "POST /train?run_async=false",
    )

    print("\nAll checks passed ✅")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except requests.ConnectionError:
        print("[FAIL] Could not connect. Is the server running (python run.py)?")
        sys.exit(1)
