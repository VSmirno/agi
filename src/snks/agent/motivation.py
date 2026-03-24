"""IntrinsicMotivation: curiosity-driven action selection based on prediction error."""

from __future__ import annotations

import math
import random
from collections import defaultdict

from snks.agent.causal_model import _context_hash
from snks.daf.types import CausalAgentConfig

# Perceptual hash IDs live in [10000, ...), real DAF SKS IDs below 10000.
_PHASH_OFFSET = 10000

# Best params from grid search (Hypothesis A+B exhausted): denom=5.5, w=(0.92, 0.08)
_STATE_NOVELTY_DENOM = 5.5
_STATE_NOVELTY_WEIGHT = 0.92
_ACTION_NOVELTY_WEIGHT = 0.08


def _stable_context(sks: set[int]) -> int:
    """Hash only perceptual-hash IDs, ignoring noisy DAF SKS clusters.

    DAF produces slightly different cluster IDs on each cycle even for the
    same visual input, making full-set hashing useless for visit counting.
    Perceptual hash IDs (>= 10000) are deterministic and rotation-invariant.
    """
    stable = frozenset(s for s in sks if s >= _PHASH_OFFSET)
    return _context_hash(stable)


class IntrinsicMotivation:
    """Curiosity-driven action selection based on prediction error.

    Principle: choose action that maximizes expected prediction error
    (information gain). But with decay — already-learned transitions are boring.

    Formula:
        interest(a) = novelty(context, a) × uncertainty(context, a)
        novelty = 1 / (1 + visit_count(context, a))
        uncertainty = 1 - confidence(context, a)

    Epsilon-greedy: with probability epsilon choose random action.
    If curiosity_epsilon_horizon > 0, epsilon decays exponentially:
        epsilon(t) = epsilon_min + (epsilon_start - epsilon_min) * exp(-t / T)
    This allows chaotic initial exploration that becomes more systematic over time.
    """

    def __init__(self, config: CausalAgentConfig):
        self._epsilon_start = config.curiosity_epsilon
        self._epsilon_min = config.curiosity_epsilon_min
        self._epsilon_horizon = config.curiosity_epsilon_horizon
        # (context_hash, action) → visit count
        self._visit_counts: dict[tuple[int, int], int] = defaultdict(int)
        # state_hash → visit count (for state-level novelty)
        self._state_visits: dict[int, int] = defaultdict(int)
        self._total_steps: int = 0

    def _current_epsilon(self) -> float:
        """Compute current epsilon (fixed or decayed)."""
        if self._epsilon_horizon <= 0:
            return self._epsilon_start
        return self._epsilon_min + (self._epsilon_start - self._epsilon_min) * math.exp(
            -self._total_steps / self._epsilon_horizon
        )

    def select_action(
        self,
        current_sks: set[int],
        causal_model,  # unused, kept for compatibility
        n_actions: int,
    ) -> int:
        """Select action that maximizes expected exploration value.

        Pure count-based visitor counting: prefer untested actions.
        No prediction (causal_model.predict_effect uses coarsened hash, useless).
        Epsilon-greedy: with probability epsilon choose random action.
        """
        if random.random() < self._current_epsilon():
            return random.randint(0, n_actions - 1)

        # Use only stable perceptual hash IDs, ignoring noisy DAF clusters
        full_ctx = _stable_context(current_sks)
        best_action = 0
        best_interest = -1.0

        for a in range(n_actions):
            # Action novelty: prefer untested (context, action) pairs
            key = (full_ctx, a)
            visit_count = self._visit_counts[key]
            action_novelty = 1.0 / (1.0 + visit_count)

            # State novelty: prefer actions leading to less-visited states
            # denom=5.5: best param from grid search (Hypothesis A)
            state_novelty = 1.0 - (visit_count / (visit_count + _STATE_NOVELTY_DENOM))

            # Combined: w=(0.92, 0.08) — best from grid search (Hypothesis B)
            interest = _STATE_NOVELTY_WEIGHT * state_novelty + _ACTION_NOVELTY_WEIGHT * action_novelty

            if interest > best_interest:
                best_interest = interest
                best_action = a

        return best_action

    def update(
        self,
        context_sks: set[int],
        action: int,
        prediction_error: float,
    ) -> None:
        """Update visit counts."""
        full_hash = _stable_context(context_sks)
        key = (full_hash, action)
        self._visit_counts[key] += 1

        # Track state visits
        self._state_visits[full_hash] += 1
        self._total_steps += 1

    def get_visit_count(self, context_sks: set[int], action: int) -> int:
        """Get visit count for (context, action) pair."""
        return self._visit_counts.get((_stable_context(context_sks), action), 0)
