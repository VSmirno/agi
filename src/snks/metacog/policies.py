"""Metacognitive policies that adapt engine parameters based on confidence."""

from __future__ import annotations

from snks.daf.types import DafConfig


class NullPolicy:
    """Observation only. Does not change anything. Default."""

    def apply(self, state: "MetacogState", config: DafConfig) -> None:  # noqa: F821
        pass


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
