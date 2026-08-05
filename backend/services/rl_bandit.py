"""Linear UCB contextual bandit for macro-strategy selection (Phase 6, Part 2).

Each arm (RL action) keeps ridge-regression sufficient statistics ``A`` (d×d)
and ``b`` (d) and derived weights ``theta``. Action selection uses epsilon-greedy
exploration around the LinUCB score, with epsilon decaying as more updates
arrive (so cold start is fully exploratory). Weights persist as JSON so they
survive restarts.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import numpy as np

from backend.config import settings
from backend.services.rl_actions import N_ACTIONS
from backend.services.rl_reward import CONTEXT_DIM

EPSILON_MIN = 0.05


class LinearUCB:
    """Disjoint LinUCB bandit with epsilon-greedy exploration."""

    def __init__(
        self,
        n_actions: int,
        context_dim: int,
        alpha: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        self.n_actions = n_actions
        self.context_dim = context_dim
        self.alpha = alpha
        self.update_count = 0
        self.A = [np.identity(context_dim) for _ in range(n_actions)]
        self.b = [np.zeros(context_dim) for _ in range(n_actions)]
        self.theta = [np.zeros(context_dim) for _ in range(n_actions)]
        self._rng = np.random.default_rng(seed)

    # --- policy ----------------------------------------------------------- #
    def epsilon(self) -> float:
        """Current exploration rate: max(0.05, 1/sqrt(update_count + 1))."""
        return max(EPSILON_MIN, 1.0 / math.sqrt(self.update_count + 1))

    def ucb_scores(self, context: np.ndarray) -> np.ndarray:
        """UCB score per action: theta·x + alpha·sqrt(xᵀ A⁻¹ x)."""
        x = np.asarray(context, dtype=float).reshape(-1)
        scores = np.zeros(self.n_actions)
        for a in range(self.n_actions):
            a_inv = np.linalg.inv(self.A[a])
            mean = float(self.theta[a] @ x)
            var = float(x @ a_inv @ x)
            scores[a] = mean + self.alpha * math.sqrt(max(var, 0.0))
        return scores

    def best_ucb_action(self, context: np.ndarray) -> int:
        """Pure (deterministic) LinUCB argmax action."""
        return int(np.argmax(self.ucb_scores(context)))

    def select_action(self, context: np.ndarray) -> int:
        """Epsilon-greedy action: explore randomly w.p. epsilon, else LinUCB."""
        if self._rng.random() < self.epsilon():
            return int(self._rng.integers(self.n_actions))
        return self.best_ucb_action(context)

    # --- learning --------------------------------------------------------- #
    def update(self, action_index: int, context: np.ndarray, reward: float) -> None:
        """Ridge-regression update for one observed (action, context, reward)."""
        if not 0 <= action_index < self.n_actions:
            raise IndexError(f"action_index {action_index} out of range")
        x = np.asarray(context, dtype=float).reshape(-1)
        self.A[action_index] += np.outer(x, x)
        self.b[action_index] += reward * x
        self.theta[action_index] = np.linalg.inv(self.A[action_index]) @ self.b[action_index]
        self.update_count += 1

    # --- persistence ------------------------------------------------------ #
    def to_dict(self) -> dict:
        return {
            "n_actions": self.n_actions,
            "context_dim": self.context_dim,
            "alpha": self.alpha,
            "update_count": self.update_count,
            "A": [m.tolist() for m in self.A],
            "b": [v.tolist() for v in self.b],
            "theta": [v.tolist() for v in self.theta],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LinearUCB":
        bandit = cls(
            n_actions=int(data["n_actions"]),
            context_dim=int(data["context_dim"]),
            alpha=float(data.get("alpha", 1.0)),
        )
        bandit.update_count = int(data.get("update_count", 0))
        bandit.A = [np.array(m, dtype=float) for m in data["A"]]
        bandit.b = [np.array(v, dtype=float) for v in data["b"]]
        bandit.theta = [np.array(v, dtype=float) for v in data["theta"]]
        return bandit

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh)

    @classmethod
    def load(cls, path: Path) -> "LinearUCB":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))


# --------------------------------------------------------------------------- #
# Module-level singleton management
# --------------------------------------------------------------------------- #
_BANDIT: Optional[LinearUCB] = None


def get_bandit() -> LinearUCB:
    """Return the process-wide bandit, loading from disk or creating fresh."""
    global _BANDIT
    if _BANDIT is not None:
        return _BANDIT
    path = settings.RL_BANDIT_PATH
    if path.exists():
        try:
            _BANDIT = LinearUCB.load(path)
            return _BANDIT
        except Exception as exc:  # pragma: no cover - corrupt file
            print(f"[rl] Could not load bandit weights ({exc}); starting fresh.")
    _BANDIT = LinearUCB(
        n_actions=N_ACTIONS, context_dim=CONTEXT_DIM, alpha=settings.RL_BANDIT_ALPHA
    )
    return _BANDIT


def save_bandit() -> None:
    """Persist the current bandit to :data:`settings.RL_BANDIT_PATH`."""
    if _BANDIT is not None:
        _BANDIT.save(settings.RL_BANDIT_PATH)


def load_bandit(force: bool = False) -> LinearUCB:
    """(Re)load the bandit from disk; ``force`` drops the in-memory cache first."""
    global _BANDIT
    if force:
        _BANDIT = None
    return get_bandit()


def reset_bandit() -> None:
    """Clear the in-memory bandit and delete the persisted weights (tests)."""
    global _BANDIT
    _BANDIT = None
    try:
        settings.RL_BANDIT_PATH.unlink(missing_ok=True)
    except OSError:  # pragma: no cover
        pass
