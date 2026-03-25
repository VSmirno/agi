"""Configurator: deterministic FSM meta-control (Stage 13).

Adapts DafConfig / MetacogConfig / HACPredictionConfig in-place
based on CostState + MetacogState, with hysteresis to prevent oscillation.

Modes:
    NEUTRAL      — default, all parameters at original values
    EXPLORE      — high cost + high epistemic → increase plasticity
    CONSOLIDATE  — low cost + high stability → decrease plasticity
    GOAL_SEEKING — goal_cost > threshold → focus on dominance + stability

Transition requires hysteresis_cycles consecutive cycles in the candidate mode
before switching. EXPLORE has a divergence safeguard: forced return to NEUTRAL
after max_explore_cycles.
"""

from __future__ import annotations

from enum import Enum

from snks.daf.types import (
    ConfiguratorAction,
    ConfiguratorConfig,
    CostState,
    DafConfig,
    HACPredictionConfig,
    MetacogConfig,
)


class ConfiguratorMode(str, Enum):
    NEUTRAL = "neutral"
    EXPLORE = "explore"
    CONSOLIDATE = "consolidate"
    GOAL_SEEKING = "goal_seeking"


class Configurator:
    """Deterministic FSM that adapts system parameters based on CostState.

    Parameters are modified in-place. Original values are saved at construction
    time for restoration when returning to NEUTRAL.

    Thread safety: SNKS is single-threaded; no locking needed.
    """

    def __init__(
        self,
        config: ConfiguratorConfig,
        daf_config: DafConfig,
        metacog_config: MetacogConfig,
        hac_pred_config: HACPredictionConfig,
    ) -> None:
        self._cfg = config
        self._daf = daf_config
        self._metacog = metacog_config
        self._hac_pred = hac_pred_config

        # Save originals for NEUTRAL restoration
        self._orig_stdp_a_plus = daf_config.stdp_a_plus
        self._orig_stdp_a_minus = daf_config.stdp_a_minus
        self._orig_memory_decay = hac_pred_config.memory_decay
        self._orig_alpha = metacog_config.alpha
        self._orig_beta = metacog_config.beta
        self._orig_gamma = metacog_config.gamma
        self._orig_delta = metacog_config.delta

        self._current_mode: ConfiguratorMode = ConfiguratorMode.NEUTRAL
        self._candidate_mode: ConfiguratorMode = ConfiguratorMode.NEUTRAL
        self._candidate_count: int = 0
        self._cycles_in_mode: int = 0

    def _determine_candidate(self, metacog_state: object, cost: CostState) -> ConfiguratorMode:
        cfg = self._cfg

        # GOAL_SEEKING has priority
        if cost.goal > cfg.goal_cost_threshold:
            return ConfiguratorMode.GOAL_SEEKING

        if (cost.total > cfg.explore_cost_threshold
                and cost.epistemic_value > cfg.explore_epistemic_threshold):
            return ConfiguratorMode.EXPLORE

        stability = float(getattr(metacog_state, "stability", 0.0))
        if (cost.total < cfg.consolidate_cost_threshold
                and stability > cfg.consolidate_stability_threshold):
            return ConfiguratorMode.CONSOLIDATE

        return ConfiguratorMode.NEUTRAL

    def _apply_mode(self, mode: ConfiguratorMode) -> dict[str, tuple[float, float]]:
        """Apply mode actions in-place, return dict of changes {param: (old, new)}."""
        changed: dict[str, tuple[float, float]] = {}
        orig_plus = self._orig_stdp_a_plus
        orig_minus = self._orig_stdp_a_minus

        if mode == ConfiguratorMode.EXPLORE:
            new_plus = min(orig_plus * 1.15, orig_plus * 2.0)
            new_minus = min(orig_minus * 1.10, orig_minus * 2.0)
            new_decay = 0.98
            if self._daf.stdp_a_plus != new_plus:
                changed["stdp_a_plus"] = (self._daf.stdp_a_plus, new_plus)
                self._daf.stdp_a_plus = new_plus
            if self._daf.stdp_a_minus != new_minus:
                changed["stdp_a_minus"] = (self._daf.stdp_a_minus, new_minus)
                self._daf.stdp_a_minus = new_minus
            if self._hac_pred.memory_decay != new_decay:
                changed["memory_decay"] = (self._hac_pred.memory_decay, new_decay)
                self._hac_pred.memory_decay = new_decay

        elif mode == ConfiguratorMode.CONSOLIDATE:
            new_plus = max(orig_plus * 0.85, orig_plus * 0.5)
            new_minus = max(orig_minus * 0.90, orig_minus * 0.5)
            new_decay = 0.90
            if self._daf.stdp_a_plus != new_plus:
                changed["stdp_a_plus"] = (self._daf.stdp_a_plus, new_plus)
                self._daf.stdp_a_plus = new_plus
            if self._daf.stdp_a_minus != new_minus:
                changed["stdp_a_minus"] = (self._daf.stdp_a_minus, new_minus)
                self._daf.stdp_a_minus = new_minus
            if self._hac_pred.memory_decay != new_decay:
                changed["memory_decay"] = (self._hac_pred.memory_decay, new_decay)
                self._hac_pred.memory_decay = new_decay

        elif mode == ConfiguratorMode.GOAL_SEEKING:
            if self._metacog.gamma != 0.0:
                changed["gamma"] = (self._metacog.gamma, 0.0)
                self._metacog.gamma = 0.0
            if self._metacog.delta != 0.0:
                changed["delta"] = (self._metacog.delta, 0.0)
                self._metacog.delta = 0.0
            if self._metacog.alpha != 0.5:
                changed["alpha"] = (self._metacog.alpha, 0.5)
                self._metacog.alpha = 0.5
            if self._metacog.beta != 0.5:
                changed["beta"] = (self._metacog.beta, 0.5)
                self._metacog.beta = 0.5

        elif mode == ConfiguratorMode.NEUTRAL:
            # Restore all originals
            if self._daf.stdp_a_plus != self._orig_stdp_a_plus:
                changed["stdp_a_plus"] = (self._daf.stdp_a_plus, self._orig_stdp_a_plus)
                self._daf.stdp_a_plus = self._orig_stdp_a_plus
            if self._daf.stdp_a_minus != self._orig_stdp_a_minus:
                changed["stdp_a_minus"] = (self._daf.stdp_a_minus, self._orig_stdp_a_minus)
                self._daf.stdp_a_minus = self._orig_stdp_a_minus
            if self._hac_pred.memory_decay != self._orig_memory_decay:
                changed["memory_decay"] = (self._hac_pred.memory_decay, self._orig_memory_decay)
                self._hac_pred.memory_decay = self._orig_memory_decay
            if self._metacog.alpha != self._orig_alpha:
                changed["alpha"] = (self._metacog.alpha, self._orig_alpha)
                self._metacog.alpha = self._orig_alpha
            if self._metacog.beta != self._orig_beta:
                changed["beta"] = (self._metacog.beta, self._orig_beta)
                self._metacog.beta = self._orig_beta
            if self._metacog.gamma != self._orig_gamma:
                changed["gamma"] = (self._metacog.gamma, self._orig_gamma)
                self._metacog.gamma = self._orig_gamma
            if self._metacog.delta != self._orig_delta:
                changed["delta"] = (self._metacog.delta, self._orig_delta)
                self._metacog.delta = self._orig_delta

        return changed

    def update(self, metacog_state: object) -> ConfiguratorAction | None:
        """Determine mode, apply actions if confirmed by hysteresis.

        Args:
            metacog_state: MetacogState with .cost (CostState | None) and .stability.

        Returns:
            ConfiguratorAction if mode switched or params changed, else None.
        """
        cost: CostState | None = getattr(metacog_state, "cost", None)
        if cost is None:
            # Stage 12 not active: stay NEUTRAL
            self._cycles_in_mode += 1
            return ConfiguratorAction(
                mode=self._current_mode.value,
                changed={},
                cycles_in_mode=self._cycles_in_mode,
            )

        candidate = self._determine_candidate(metacog_state, cost)

        # Divergence safeguard: force NEUTRAL if EXPLORE runs too long
        if (self._current_mode == ConfiguratorMode.EXPLORE
                and self._cycles_in_mode > self._cfg.max_explore_cycles):
            candidate = ConfiguratorMode.NEUTRAL

        if candidate == self._candidate_mode:
            self._candidate_count += 1
        else:
            self._candidate_mode = candidate
            self._candidate_count = 1

        # Switch mode if confirmed by hysteresis
        if (self._candidate_count >= self._cfg.hysteresis_cycles
                and candidate != self._current_mode):
            changed = self._apply_mode(candidate)
            self._current_mode = candidate
            self._cycles_in_mode = 0
            self._candidate_count = 0
            return ConfiguratorAction(
                mode=self._current_mode.value,
                changed=changed,
                cycles_in_mode=self._cycles_in_mode,
            )

        self._cycles_in_mode += 1

        # Return action even without change (for CycleResult tracking)
        return ConfiguratorAction(
            mode=self._current_mode.value,
            changed={},
            cycles_in_mode=self._cycles_in_mode,
        )
