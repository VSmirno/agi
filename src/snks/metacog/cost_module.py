"""IntrinsicCostModule: unified intrinsic energy function (Stage 12).

Combines homeostatic, epistemic, and goal-directed signals into a single
total_cost ∈ [0, 1] used by Configurator and StochasticSimulator.

Formula:
    comfort       = 1 - homeostatic_cost        ∈ [0, 1]
    curiosity     = epistemic_value              ∈ [0, 1]
    goal_progress = 1 - goal_cost               ∈ [0, 1]

    value      = w_h * comfort + w_e * curiosity + w_g * goal_progress
    total_cost = 1 - value                       ∈ [0, 1]
"""

from __future__ import annotations

from snks.daf.types import CostModuleConfig, CostState


class IntrinsicCostModule:
    """Computes intrinsic cost from MetacogState + mean_firing_rate."""

    def __init__(self, config: CostModuleConfig) -> None:
        self.config = config
        self._goal_cost: float = 0.0

    def set_goal_cost(self, cost: float) -> None:
        """Set external goal cost ∈ [0, 1]. 0 = goal achieved / irrelevant."""
        self._goal_cost = float(max(0.0, min(1.0, cost)))

    def compute(
        self,
        metacog_state: object,  # MetacogState (avoid circular import)
        mean_firing_rate: float,
    ) -> CostState:
        """Compute intrinsic cost from current system state.

        Args:
            metacog_state: MetacogState with .winner_pe and .meta_pe fields.
            mean_firing_rate: mean v-component of FHN states (proxy for firing rate).

        Returns:
            CostState with all components.
        """
        cfg = self.config
        target = cfg.firing_rate_target if cfg.firing_rate_target is not None else 0.05

        # Homeostatic cost: deviation from target
        homeostatic = min(abs(mean_firing_rate - target) / max(target, 1e-8), 1.0)

        # Epistemic value: max of available PE signals (see spec §5.4)
        winner_pe = float(getattr(metacog_state, "winner_pe", 0.0))
        meta_pe = float(getattr(metacog_state, "meta_pe", 0.0))
        available = [x for x in [winner_pe, meta_pe] if x > 0.0]
        epistemic_value = max(available) if available else 0.0

        # Goal cost (external)
        goal_cost = self._goal_cost

        # Compute value and total cost
        comfort = 1.0 - homeostatic
        curiosity = epistemic_value
        goal_progress = 1.0 - goal_cost

        value = (cfg.w_homeostatic * comfort
                 + cfg.w_epistemic * curiosity
                 + cfg.w_goal * goal_progress)
        # Normalize by weight sum (safety guard if weights don't sum to 1)
        weight_sum = cfg.w_homeostatic + cfg.w_epistemic + cfg.w_goal
        if weight_sum > 1e-8:
            value /= weight_sum
        value = max(0.0, min(1.0, value))
        total_cost = 1.0 - value

        return CostState(
            total=total_cost,
            homeostatic=homeostatic,
            epistemic_value=epistemic_value,
            goal=goal_cost,
        )
