"""Metacognitive policies that adapt engine parameters based on confidence."""

from __future__ import annotations

import torch
from torch import Tensor

from snks.daf.types import DafConfig


class NullPolicy:
    """Observation only. Does not change anything. Default."""

    def apply(self, state: "MetacogState", config: DafConfig) -> None:  # noqa: F821
        pass

    def get_broadcast_currents(self, n_nodes: int) -> Tensor | None:
        return None


class NoisePolicy:
    """Adapts noise_sigma based on confidence.

    noise_sigma = base_sigma * (1 + strength * (1 - confidence))

    confidence=1.0 -> noise = base_sigma        (stabilize pattern)
    confidence=0.0 -> noise = base_sigma * (1 + strength)  (explore)

    base_sigma is fixed on first apply() call from current config.noise_sigma.
    """

    def __init__(self, strength: float = 1.0) -> None:
        self.strength = strength
        self._base_sigma: float | None = None

    def apply(self, state: "MetacogState", config: DafConfig) -> None:  # noqa: F821
        if self._base_sigma is None:
            self._base_sigma = config.noise_sigma
        config.noise_sigma = self._base_sigma * (1.0 + self.strength * (1.0 - state.confidence))

    def get_broadcast_currents(self, n_nodes: int) -> Tensor | None:
        return None


class STDPPolicy:
    """Adapts stdp_a_plus based on confidence.

    a_plus = base_a_plus * (1 + strength * confidence)

    High confidence -> strengthen learning (consolidate pattern).
    Low confidence -> return to base value.
    base_a_plus is fixed on first apply() call.
    """

    def __init__(self, strength: float = 1.0) -> None:
        self.strength = strength
        self._base_a_plus: float | None = None

    def apply(self, state: "MetacogState", config: DafConfig) -> None:  # noqa: F821
        if self._base_a_plus is None:
            self._base_a_plus = config.stdp_a_plus
        config.stdp_a_plus = self._base_a_plus * (1.0 + self.strength * state.confidence)

    def get_broadcast_currents(self, n_nodes: int) -> Tensor | None:
        return None


class BroadcastPolicy:
    """Injects current into winner_nodes when confidence >= threshold.

    Implements global ignition: the GWS winner broadcasts back into the network,
    elevating activation of winner nodes in the next perception cycle.

    Broadcast is one-tick delayed: applied in the next cycle's step 1c.
    """

    def __init__(self, strength: float = 1.0, threshold: float = 0.6) -> None:
        self.strength = strength
        self.threshold = threshold
        self._pending_nodes: set[int] | None = None

    def apply(self, state: "MetacogState", config: DafConfig) -> None:  # noqa: F821
        if state.confidence >= self.threshold and state.winner_nodes:
            self._pending_nodes = set(state.winner_nodes)

    def get_broadcast_currents(self, n_nodes: int) -> Tensor | None:
        """Return current vector for winner_nodes, then clear pending.

        Returns None if no broadcast is pending.
        """
        if self._pending_nodes is None:
            return None
        currents = torch.zeros(n_nodes)
        for node in self._pending_nodes:
            if node < n_nodes:
                currents[node] = self.strength
        self._pending_nodes = None
        return currents
