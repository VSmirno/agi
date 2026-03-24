"""IntrinsicMotivation: curiosity-driven action selection based on prediction error."""

from __future__ import annotations

import random
from collections import defaultdict

from snks.agent.causal_model import CausalWorldModel, _context_hash, _coarsen_sks
from snks.daf.types import CausalAgentConfig

# Perceptual hash IDs live in [10000, ...), real DAF SKS IDs below 10000.
_PHASH_OFFSET = 10000


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
    """

    def __init__(self, config: CausalAgentConfig):
        self.epsilon = config.curiosity_epsilon
        self.decay = config.curiosity_decay
        self._n_bins = config.causal_context_bins
        # (context_hash, action) → visit count
        self._visit_counts: dict[tuple[int, int], int] = defaultdict(int)
        # (context_hash, action) → current prediction error
        self._prediction_errors: dict[tuple[int, int], float] = {}
        # (context_hash, action) → smoothed learning progress (EMA of delta error)
        self._learning_progress: dict[tuple[int, int], float] = {}
        # state_hash → visit count (for state-level novelty)
        self._state_visits: dict[int, int] = defaultdict(int)

    def select_action(
        self,
        current_sks: set[int],
        causal_model: CausalWorldModel,
        n_actions: int,
    ) -> int:
        """Select action that maximizes expected exploration value.

        Count-based: prefer actions leading to least-visited states.
        Learning progress suppresses actions with no learning gain (noisy-TV).
        Epsilon-greedy: with probability epsilon choose random action.
        """
        if random.random() < self.epsilon:
            return random.randint(0, n_actions - 1)

        # Use only stable perceptual hash IDs, ignoring noisy DAF clusters
        full_ctx = _stable_context(current_sks)
        best_action = 0
        best_interest = -1.0

        for a in range(n_actions):
            key = (full_ctx, a)

            # Count-based: predict next state, prefer least-visited
            predicted_effect, confidence = causal_model.predict_effect(current_sks, a)
            predicted_next = current_sks | predicted_effect
            next_hash = _stable_context(predicted_next)
            state_novelty = 1.0 / (1.0 + self._state_visits.get(next_hash, 0))

            # Action novelty (also on full hash)
            visit_count = self._visit_counts[key]
            action_novelty = 1.0 / (1.0 + visit_count)

            # Learning progress: suppress actions with no learning gain
            lp = self._learning_progress.get(key, 1.0)  # unknown = high

            # Count-based interest with LP suppression
            interest = (0.6 * state_novelty + 0.4 * action_novelty) * (0.3 + 0.7 * lp)

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
        """Update visit counts and novelty estimates."""
        full_hash = _stable_context(context_sks)
        key = (full_hash, action)
        self._visit_counts[key] += 1

        # Learning progress: delta of prediction error (positive = still learning)
        prev_error = self._prediction_errors.get(key, 1.0)  # first time = max
        learning_progress = max(0.0, prev_error - prediction_error)

        # EMA smoothing of learning progress
        self._learning_progress[key] = (
            self._learning_progress.get(key, 1.0) * self.decay
            + learning_progress * (1.0 - self.decay)
        )

        # Update current error
        self._prediction_errors[key] = prediction_error

        # Track state visits
        self._state_visits[full_hash] += 1

    def get_visit_count(self, context_sks: set[int], action: int) -> int:
        """Get visit count for (context, action) pair."""
        return self._visit_counts.get((_stable_context(context_sks), action), 0)
