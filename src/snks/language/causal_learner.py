"""CausalLearner: observes action effects for CausalWorldModel (Stage 25).

Thin adapter that captures before/after SKS state around each action
and feeds it to observe_transition(). The model computes symmetric_difference
internally.
"""

from __future__ import annotations

from snks.agent.causal_model import CausalWorldModel


class CausalLearner:
    """Observes action effects and updates CausalWorldModel."""

    def __init__(self, causal_model: CausalWorldModel) -> None:
        self._model = causal_model
        self._before: set[int] | None = None

    def before_action(self, current_sks: set[int]) -> None:
        """Snapshot state before action."""
        self._before = set(current_sks)

    def after_action(self, action: int, current_sks: set[int]) -> None:
        """Compare with snapshot, update causal model."""
        if self._before is None:
            return
        self._model.observe_transition(self._before, action, current_sks)
        self._before = None
