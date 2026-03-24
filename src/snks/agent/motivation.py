"""IntrinsicMotivation: curiosity-driven action selection based on prediction error."""

from __future__ import annotations

import random
from collections import defaultdict

from snks.agent.causal_model import CausalWorldModel, _context_hash
from snks.daf.types import CausalAgentConfig


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
        # (context_hash, action) → visit count
        self._visit_counts: dict[tuple[int, int], int] = defaultdict(int)
        # (context_hash, action) → cumulative prediction error
        self._prediction_errors: dict[tuple[int, int], float] = defaultdict(float)

    def select_action(
        self,
        current_sks: set[int],
        causal_model: CausalWorldModel,
        n_actions: int,
    ) -> int:
        """Select action that maximizes expected information gain.

        Epsilon-greedy: with probability epsilon choose random action.
        """
        if random.random() < self.epsilon:
            return random.randint(0, n_actions - 1)

        ctx_hash = _context_hash(frozenset(current_sks))
        best_action = 0
        best_interest = -1.0

        for a in range(n_actions):
            key = (ctx_hash, a)
            visit_count = self._visit_counts[key]
            novelty = 1.0 / (1.0 + visit_count)

            # Get uncertainty from causal model
            _, confidence = causal_model.predict_effect(current_sks, a)
            uncertainty = 1.0 - confidence

            interest = novelty * uncertainty

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
        ctx_hash = _context_hash(frozenset(context_sks))
        key = (ctx_hash, action)
        self._visit_counts[key] += 1
        self._prediction_errors[key] = (
            self._prediction_errors[key] * self.decay + prediction_error
        )

    def get_visit_count(self, context_sks: set[int], action: int) -> int:
        """Get visit count for (context, action) pair."""
        ctx_hash = _context_hash(frozenset(context_sks))
        return self._visit_counts.get((ctx_hash, action), 0)
