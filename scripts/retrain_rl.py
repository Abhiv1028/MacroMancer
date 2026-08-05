"""Offline re-fit of the LinUCB bandit from all stored transitions.

Replays every :class:`RLTransition` (context + action + reward) into a fresh
:class:`LinearUCB` and overwrites the persisted weights. Run manually or via cron:

    python -m scripts.retrain_rl
    # or
    python scripts/retrain_rl.py

Safe to run repeatedly; it always rebuilds from scratch.
"""

from __future__ import annotations

import os
import sys

# Allow ``python scripts/retrain_rl.py`` (add project root to the path).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.config import settings  # noqa: E402
from backend.db import init_db, session_scope  # noqa: E402
from backend.models import RLTransition  # noqa: E402
from backend.services.rl_actions import ACTION_NAMES, N_ACTIONS  # noqa: E402
from backend.services.rl_bandit import LinearUCB, load_bandit  # noqa: E402
from backend.services.rl_reward import CONTEXT_DIM, context_dict_to_vector  # noqa: E402


def retrain() -> dict:
    """Rebuild the bandit from all transitions and persist it. Returns a summary."""
    init_db()
    bandit = LinearUCB(
        n_actions=N_ACTIONS,
        context_dim=CONTEXT_DIM,
        alpha=settings.RL_BANDIT_ALPHA,
    )

    replayed = 0
    skipped = 0
    with session_scope() as db:
        transitions = (
            db.query(RLTransition).order_by(RLTransition.created_at.asc()).all()
        )
        for tx in transitions:
            action_log = tx.action_log
            reward = tx.reward
            if action_log is None or reward is None or action_log.action is None:
                skipped += 1
                continue
            action_name = action_log.action.name
            if action_name not in ACTION_NAMES:
                skipped += 1
                continue
            action_index = ACTION_NAMES.index(action_name)
            context = context_dict_to_vector(action_log.context_json or {})
            bandit.update(action_index, context, float(reward.reward))
            replayed += 1

    bandit.save(settings.RL_BANDIT_PATH)
    # Drop any in-memory singleton so the next access reloads the fresh weights.
    load_bandit(force=True)

    summary = {
        "transitions_replayed": replayed,
        "transitions_skipped": skipped,
        "update_count": bandit.update_count,
        "weights_path": str(settings.RL_BANDIT_PATH),
    }
    return summary


def main() -> int:
    summary = retrain()
    print(
        f"[retrain_rl] Replayed {summary['transitions_replayed']} transitions "
        f"(skipped {summary['transitions_skipped']}); "
        f"saved bandit with {summary['update_count']} updates -> "
        f"{summary['weights_path']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
