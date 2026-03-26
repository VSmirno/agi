"""TieredPlanner: hot/cold memory arbitration for action selection (Stage 16).

Hot memory  = CausalWorldModel (current-episode transition table).
Cold memory = ConsolidationScheduler (SSG causal layer, cross-session).

Cold overrides hot when SSG confidence > CausalWorldModel confidence.
"""

from __future__ import annotations

import random


class TieredPlanner:
    """Arbitrates between hot (CausalWorldModel) and cold (SSG) memory.

    Returns the best available action and its source tier.
    """

    def __init__(
        self,
        causal_model,           # CausalWorldModel
        scheduler,              # ConsolidationScheduler
        cold_threshold: float = 0.3,
        n_actions: int = 7,
    ) -> None:
        self.causal_model = causal_model
        self.scheduler = scheduler
        self.cold_threshold = cold_threshold
        self.n_actions = n_actions

    def plan(self, context_sks: set[int]) -> tuple[int, str]:
        """Return (action, source) where source ∈ {'hot', 'cold', 'random'}."""
        hot_action, hot_conf = self.causal_model.best_action(context_sks,
                                                              n_actions=self.n_actions)
        cold_action, cold_weight = self.scheduler.query(context_sks,
                                                        threshold=self.cold_threshold)
        if cold_action is not None and cold_weight > hot_conf:
            return cold_action, "cold"
        if hot_action is not None:
            return hot_action, "hot"
        return random.randint(0, self.n_actions - 1), "random"
