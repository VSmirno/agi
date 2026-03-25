"""MetacogMonitor: observes system state, computes confidence."""

from __future__ import annotations

from dataclasses import dataclass, field

from torch import Tensor

from snks.daf.types import DafConfig, MetacogConfig
from snks.metacog.policies import NullPolicy, NoisePolicy, STDPPolicy, BroadcastPolicy


@dataclass
class MetacogState:
    confidence: float    # [0, 1]
    dominance: float     # from GWSState.dominance
    stability: float     # |winner_now & winner_prev| / winner_size in [0, 1]
    pred_error: float    # CycleResult.mean_prediction_error
    winner_pe: float = 0.0              # Stage 9: HAC PE for GWS winner ∈ [0, 1]
    winner_nodes: set[int] = field(default_factory=set)  # Stage 9: for BroadcastPolicy


class MetacogMonitor:
    """Observes system state, computes confidence.

    confidence = alpha * dominance + beta * stability + gamma * (1 - pe_for_confidence)

    pe_for_confidence:
      - If winner_pe > 0 (HACPredictionEngine active): uses winner_pe
      - Otherwise: fallback to pred_error_norm (global pred error)

    pred_error_norm = pred_error / max_observed_pred_error
    where max_observed_pred_error is running max over all time.
    Initialized to 1.0 (until first obs, pred_error_norm = pred_error).

    stability = |winner_now.nodes & winner_prev.nodes| / winner_now.size
    On first cycle (no previous winner): stability = 0.0.
    If gws_state is None -> confidence = 0.0, all components = 0.0.

    pred_error from CycleResult.mean_prediction_error --
    aggregate over all DAF nodes (not per-winner).
    """

    def __init__(self, config: MetacogConfig | None = None) -> None:
        if config is None:
            from snks.daf.types import MetacogConfig as _MC
            config = _MC()
        self.config = config
        self._prev_winner_nodes: set[int] | None = None
        self._max_pred_error: float = 1.0  # running max

        # Instantiate policy
        policy_name = config.policy.lower()
        strength = config.policy_strength
        if policy_name == "noise":
            self._policy: NullPolicy | NoisePolicy | STDPPolicy | BroadcastPolicy = NoisePolicy(strength=strength)
        elif policy_name == "stdp":
            self._policy = STDPPolicy(strength=strength)
        elif policy_name == "broadcast":
            threshold = getattr(config, "broadcast_threshold", 0.6)
            self._policy = BroadcastPolicy(strength=strength, threshold=threshold)
        else:
            self._policy = NullPolicy()

    def update(
        self,
        gws_state: "GWSState | None",  # noqa: F821
        cycle_result: "CycleResult",   # noqa: F821
    ) -> MetacogState:
        """Update prev_winner, max_pred_error; return MetacogState."""
        pred_error = float(cycle_result.mean_prediction_error)
        winner_pe = float(getattr(cycle_result, "winner_pe", 0.0))

        if gws_state is None:
            self._prev_winner_nodes = None
            return MetacogState(
                confidence=0.0,
                dominance=0.0,
                stability=0.0,
                pred_error=pred_error,
                winner_pe=winner_pe,
                winner_nodes=set(),
            )

        # Update running max pred_error
        if pred_error > self._max_pred_error:
            self._max_pred_error = pred_error

        # Compute pred_error_norm
        pred_error_norm = pred_error / self._max_pred_error if self._max_pred_error > 0.0 else pred_error

        # Stage 9: use winner_pe if available, else fallback to pred_error_norm
        pe_for_confidence = winner_pe if winner_pe > 0.0 else pred_error_norm

        # Compute stability
        winner_nodes = set(gws_state.winner_nodes)
        winner_size = gws_state.winner_size
        if self._prev_winner_nodes is None or winner_size == 0:
            stability = 0.0
        else:
            intersection = len(winner_nodes & self._prev_winner_nodes)
            stability = intersection / winner_size

        # Update prev winner
        self._prev_winner_nodes = winner_nodes

        # Compute confidence
        dominance = gws_state.dominance
        cfg = self.config
        confidence = cfg.alpha * dominance + cfg.beta * stability + cfg.gamma * (1.0 - pe_for_confidence)
        confidence = max(0.0, min(1.0, confidence))

        return MetacogState(
            confidence=confidence,
            dominance=dominance,
            stability=stability,
            pred_error=pred_error,
            winner_pe=winner_pe,
            winner_nodes=winner_nodes,
        )

    def apply_policy(self, state: MetacogState, config: DafConfig) -> None:
        """Delegate to active policy."""
        self._policy.apply(state, config)

    def get_broadcast_currents(self, n_nodes: int) -> Tensor | None:
        """Get broadcast currents from active policy (BroadcastPolicy only)."""
        return self._policy.get_broadcast_currents(n_nodes)
