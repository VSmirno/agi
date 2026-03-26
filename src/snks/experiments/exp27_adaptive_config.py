"""Experiment 27: Adaptive vs fixed configuration (Stage 13).

Two-phase benchmark for the Configurator FSM:
  Phase 1 — novel environment: high PE, low stability → needs EXPLORE
             (higher stdp_a_plus = more plasticity = higher score)
  Phase 2 — familiar environment: low PE, high stability → needs CONSOLIDATE
             (lower stdp_a_plus = more stability = higher score)

Gate: adaptive_score > fixed_score * 1.1
"""
from __future__ import annotations

import statistics
import sys
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


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

@dataclass
class MockMetacogState:
    """Minimal metacognitive state accepted by Configurator and IntrinsicCostModule."""

    confidence: float = 0.5
    dominance: float = 0.8
    stability: float = 0.5
    pred_error: float = 0.1
    winner_pe: float = 0.3
    meta_pe: float = 0.0
    cost: Optional[CostState] = None
    winner_nodes: set = field(default_factory=set)


# ---------------------------------------------------------------------------
# Simulation core
# ---------------------------------------------------------------------------

def _simulate(use_configurator: bool, n_cycles: int) -> tuple[float, list[str]]:
    """Run two-phase simulation and return (total_score, mode_history).

    Args:
        use_configurator: Whether the Configurator FSM is active.
        n_cycles: Number of cycles per phase.

    Returns:
        Tuple of (mean score over both phases, list of mode strings recorded
        on each Configurator action).
    """
    daf_cfg = DafConfig(
        num_nodes=1000,
        stdp_a_plus=0.01,
        stdp_a_minus=0.012,
        device="cpu",
    )
    metacog_cfg = MetacogConfig(alpha=1 / 3, beta=1 / 3, gamma=1 / 3, delta=0.0)
    hac_cfg = HACPredictionConfig(memory_decay=0.95, enabled=True)
    cfg_cfg = ConfiguratorConfig(
        hysteresis_cycles=4,
        max_explore_cycles=30,
        explore_cost_threshold=0.55,
        explore_epistemic_threshold=0.35,
        consolidate_cost_threshold=0.40,
        consolidate_stability_threshold=0.65,
        goal_cost_threshold=0.1,
    )

    orig_stdp = daf_cfg.stdp_a_plus

    configurator: Optional[Configurator] = (
        Configurator(cfg_cfg, daf_cfg, metacog_cfg, hac_cfg)
        if use_configurator
        else None
    )
    # w_homeostatic dominates so that high firing_rate deviation → high total_cost.
    # This allows EXPLORE to trigger even when epistemic_value is high
    # (high curiosity reduces value only slightly via w_epistemic=0.1).
    # Phase 1: firing_rate=0.25 → homeostatic=1.0 → total_cost≈0.83 > 0.55 ✓
    # Phase 2: firing_rate=0.05 → homeostatic=0.0 → total_cost≈0.09 < 0.40 ✓
    cost_module = IntrinsicCostModule(
        CostModuleConfig(
            w_homeostatic=0.8,
            w_epistemic=0.1,
            w_goal=0.1,
            firing_rate_target=0.05,
        )
    )

    mode_history: list[str] = []
    phase1_scores: list[float] = []
    phase2_scores: list[float] = []

    # ------------------------------------------------------------------
    # Phase 1: novel environment — needs plasticity (high EXPLORE score)
    # Score = normalized plasticity ratio (higher stdp = better in phase 1)
    # ------------------------------------------------------------------
    for i in range(n_cycles):
        pe = 0.70 + 0.05 * ((i * 3) % 7) / 7
        stab = 0.15 + 0.05 * (i % 5) / 5
        state = MockMetacogState(
            winner_pe=pe,
            stability=stab,
            dominance=0.5,
            pred_error=pe,
        )
        # firing_rate=0.25: homeostatic=1.0, total_cost≈0.83 > 0.55, epistemic≈0.72 > 0.35 → EXPLORE
        state.cost = cost_module.compute(state, mean_firing_rate=0.25)

        if configurator is not None:
            action = configurator.update(state)
            if action is not None:
                mode_history.append(action.mode)

        # Phase 1: plasticity ratio maps [0.5, 2.0] → [0, 1] linearly
        plasticity = daf_cfg.stdp_a_plus / orig_stdp
        score = (plasticity - 0.5) / 1.5
        phase1_scores.append(max(0.0, min(score, 1.0)))

    # ------------------------------------------------------------------
    # Phase 2: familiar environment — needs stability (low plasticity score)
    # Score = inverse plasticity ratio (lower stdp = better in phase 2)
    # ------------------------------------------------------------------
    for i in range(n_cycles):
        pe = 0.05 + 0.03 * (i % 4) / 4
        stab = 0.80 + 0.05 * (i % 6) / 6
        state = MockMetacogState(
            winner_pe=pe,
            stability=stab,
            dominance=0.9,
            pred_error=pe,
        )
        # firing_rate=0.05: homeostatic=0.0, total_cost≈0.09 < 0.40, stability>0.65 → CONSOLIDATE
        state.cost = cost_module.compute(state, mean_firing_rate=0.05)

        if configurator is not None:
            action = configurator.update(state)
            if action is not None:
                mode_history.append(action.mode)

        # Phase 2: inverse plasticity maps [0.5, 2.0] → [1, 0] linearly
        plasticity = daf_cfg.stdp_a_plus / orig_stdp
        score = (2.0 - plasticity) / 1.5
        phase2_scores.append(max(0.0, min(score, 1.0)))

    total = statistics.mean(phase1_scores + phase2_scores)
    return total, mode_history


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(device: str = "cpu", n_cycles_per_phase: int = 60) -> dict:
    """Run Experiment 27 and return structured result.

    Args:
        device: Compute device string (currently unused; simulation is CPU-only).
        n_cycles_per_phase: Number of cycles per phase for both adaptive and
            fixed runs.

    Returns:
        Dictionary with keys:
            passed (bool): True when gate adaptive_score > fixed_score * 1.1.
            adaptive_score (float): Mean score with Configurator active.
            fixed_score (float): Mean score with fixed parameters.
            ratio (float): adaptive_score / fixed_score.
            mode_switches (list[str]): Sequence of modes emitted by Configurator.
    """
    adaptive_score, modes = _simulate(True, n_cycles_per_phase)
    fixed_score, _ = _simulate(False, n_cycles_per_phase)

    ratio = adaptive_score / max(fixed_score, 1e-8)
    passed = ratio > 1.1

    return {
        "passed": passed,
        "adaptive_score": adaptive_score,
        "fixed_score": fixed_score,
        "ratio": ratio,
        "mode_switches": modes,
    }


if __name__ == "__main__":
    device = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    result = run(device=device)
    print(result)
    sys.exit(0 if result["passed"] else 1)
