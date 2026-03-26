"""Experiment 28: Context switching speed (Stage 13).

Tests that Configurator transitions from CONSOLIDATE to EXPLORE in under 20 cycles
after a task change that raises cost and prediction error.

Gate: cycles_to_switch < 20 AND explore_reached == True

Cost math (w_homeostatic=0.5, w_epistemic=0.3, w_goal=0.2, firing_rate_target=0.05):

  CONSOLIDATE conditions (winner_pe=0.05, firing_rate=0.05):
    homeostatic = |0.05 - 0.05| / 0.05 = 0.0  → comfort = 1.0
    epistemic   = 0.05
    goal        = 0.0                           → goal_progress = 1.0
    value       = 0.5*1.0 + 0.3*0.05 + 0.2*1.0 = 0.715
    total_cost  = 0.285 < 0.40 ✓, stability=0.85 > 0.65 ✓ → CONSOLIDATE

  EXPLORE conditions (winner_pe=0.75, firing_rate=0.20):
    homeostatic = |0.20 - 0.05| / 0.05 = 3.0 → clamp 1.0 → comfort = 0.0
    epistemic   = 0.75
    goal        = 0.0                           → goal_progress = 1.0
    value       = 0.5*0.0 + 0.3*0.75 + 0.2*1.0 = 0.425
    total_cost  = 0.575 > 0.55 ✓, epistemic=0.75 > 0.35 ✓ → EXPLORE
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from snks.daf.types import (
    ConfiguratorConfig,
    CostModuleConfig,
    CostState,
    DafConfig,
    HACPredictionConfig,
    MetacogConfig,
)
from snks.metacog.configurator import Configurator
from snks.metacog.cost_module import IntrinsicCostModule


@dataclass
class MockMetacogState:
    """Minimal metacog state for Configurator.update()."""

    confidence: float = 0.5
    dominance: float = 0.8
    stability: float = 0.5
    pred_error: float = 0.1
    winner_pe: float = 0.3
    meta_pe: float = 0.0
    cost: Optional[CostState] = None
    winner_nodes: set = field(default_factory=set)


def run(
    device: str = "cpu",
    n_consolidate_cycles: int = 50,
    n_explore_cycles: int = 50,
) -> dict:
    """Run context switching experiment.

    Args:
        device: Torch device string (unused here, kept for CLI parity).
        n_consolidate_cycles: Cycles in Phase 1 (CONSOLIDATE conditions).
        n_explore_cycles: Cycles in Phase 2 (EXPLORE conditions).

    Returns:
        Dict with keys: passed, cycles_to_switch, consolidate_reached,
        explore_reached, mode_history.
    """
    # --- Configurator with fast hysteresis ---
    cfg = ConfiguratorConfig(
        hysteresis_cycles=5,
        max_explore_cycles=40,
        explore_cost_threshold=0.55,
        explore_epistemic_threshold=0.35,
        consolidate_cost_threshold=0.40,
        consolidate_stability_threshold=0.65,
        goal_cost_threshold=0.10,
    )

    daf_config = DafConfig()
    metacog_config = MetacogConfig()
    hac_pred_config = HACPredictionConfig()

    configurator = Configurator(
        config=cfg,
        daf_config=daf_config,
        metacog_config=metacog_config,
        hac_pred_config=hac_pred_config,
    )

    # Weights tuned so that:
    #   low-noise state  → total_cost ~0.285 (< 0.40 → CONSOLIDATE)
    #   high-noise state → total_cost ~0.575 (> 0.55 → EXPLORE)
    cost_module = IntrinsicCostModule(
        CostModuleConfig(
            w_homeostatic=0.5,
            w_epistemic=0.3,
            w_goal=0.2,
            firing_rate_target=0.05,
        )
    )

    # ------------------------------------------------------------------ #
    # Phase 1: CONSOLIDATE conditions                                     #
    # ------------------------------------------------------------------ #
    mode_history: list[str] = []
    consolidate_reached = False

    for _ in range(n_consolidate_cycles):
        state = MockMetacogState(
            winner_pe=0.05,
            stability=0.85,
            dominance=0.9,
            pred_error=0.05,
        )
        cost_state = cost_module.compute(state, mean_firing_rate=0.05)
        state.cost = cost_state

        action = configurator.update(state)
        mode = action.mode if action else "neutral"
        mode_history.append(mode)
        if mode == "consolidate":
            consolidate_reached = True

    # ------------------------------------------------------------------ #
    # Phase 2: task change → EXPLORE conditions                           #
    # ------------------------------------------------------------------ #
    explore_reached = False
    cycles_to_switch = -1

    for i in range(n_explore_cycles):
        state = MockMetacogState(
            winner_pe=0.75,
            stability=0.20,
            dominance=0.5,
            pred_error=0.75,
        )
        cost_state = cost_module.compute(state, mean_firing_rate=0.20)
        state.cost = cost_state

        action = configurator.update(state)
        mode = action.mode if action else "neutral"
        mode_history.append(mode)

        if mode == "explore" and not explore_reached:
            explore_reached = True
            cycles_to_switch = i + 1

    # Worst-case: never switched
    if cycles_to_switch == -1:
        cycles_to_switch = n_explore_cycles

    passed = explore_reached and cycles_to_switch < 20

    return {
        "passed": passed,
        "cycles_to_switch": cycles_to_switch,
        "consolidate_reached": consolidate_reached,
        "explore_reached": explore_reached,
        "mode_history": mode_history[:30] + (["..."] if len(mode_history) > 30 else []),
    }


if __name__ == "__main__":
    import sys

    device = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    result = run(device=device)
    print(result)
    sys.exit(0 if result["passed"] else 1)
