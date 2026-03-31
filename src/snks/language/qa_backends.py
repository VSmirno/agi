"""Real QA backends for CausalWorldModel and StochasticSimulator (Stage 24b).

Replace dict-based backends from Stage 22 with adapters over real components.
"""

from __future__ import annotations

from snks.agent.causal_model import CausalWorldModel
from snks.agent.stochastic_simulator import StochasticSimulator
from snks.language.grounding_map import GroundingMap
from snks.language.qa import QAResult, QuestionType


class CausalQABackend:
    """Factual QA via CausalWorldModel.get_causal_links().

    Scans causal links for ones matching the queried action/object.
    Returns the related SKS from the link's effect or context.
    """

    def __init__(
        self,
        causal_model: CausalWorldModel,
        grounding_map: GroundingMap,
        min_confidence: float = 0.3,
    ) -> None:
        self._causal = causal_model
        self._gmap = grounding_map
        self._min_conf = min_confidence

    def query(self, roles: dict[str, int]) -> QAResult | None:
        action_sks = roles.get("ACTION")
        object_sks = roles.get("OBJECT")

        links = self._causal.get_causal_links(self._min_conf)
        if not links:
            return None

        for link in links:
            # Match: action matches AND object is in context or effect.
            if action_sks is not None and link.action != action_sks:
                continue
            involved = link.context_sks | link.effect_sks
            if object_sks is not None and object_sks not in involved:
                continue

            # Found a match. Return the "other side" of the link.
            answer_sks = []
            search_in = link.effect_sks if object_sks in link.context_sks else link.context_sks
            for sks_id in sorted(search_in):
                if sks_id != object_sks and self._gmap.sks_to_word(sks_id) is not None:
                    answer_sks.append(sks_id)

            if answer_sks:
                return QAResult(
                    answer_sks=answer_sks,
                    confidence=link.strength,
                    source=QuestionType.FACTUAL,
                    metadata={"action": link.action},
                )

        return None


class SimulationQABackend:
    """Simulation QA via StochasticSimulator.sample_effect()."""

    def __init__(
        self,
        simulator: StochasticSimulator,
        grounding_map: GroundingMap,
        current_sks: set[int] | None = None,
    ) -> None:
        self._sim = simulator
        self._gmap = grounding_map
        self._current_sks = current_sks or set()

    def set_state(self, current_sks: set[int]) -> None:
        """Update current world state for simulation."""
        self._current_sks = current_sks

    def query(self, roles: dict[str, int]) -> QAResult | None:
        action_id = roles.get("ACTION")
        if action_id is None:
            return None

        effect_sks, confidence = self._sim.sample_effect(
            self._current_sks, action_id, temperature=0.1,
        )

        if not effect_sks:
            return None

        answer_sks = sorted(effect_sks)
        action_word = self._gmap.sks_to_word(action_id) or f"action_{action_id}"

        return QAResult(
            answer_sks=answer_sks,
            confidence=confidence,
            source=QuestionType.SIMULATION,
            metadata={"action": action_word},
        )
