"""InstructionPlanner: parsed instruction → action plan (Stage 24b).

Converts chunked BabyAI instructions into action sequences using
CausalWorldModel for prerequisite detection and StochasticSimulator
for plan search.
"""

from __future__ import annotations

from snks.agent.causal_model import CausalWorldModel
from snks.agent.stochastic_simulator import StochasticSimulator
from snks.language.chunker import Chunk
from snks.language.grounding_map import GroundingMap


class InstructionPlanner:
    """Converts parsed instruction chunks into action plan."""

    def __init__(
        self,
        grounding_map: GroundingMap,
        causal_model: CausalWorldModel,
        simulator: StochasticSimulator,
        action_names: dict[str, int],
    ) -> None:
        self._gmap = grounding_map
        self._causal = causal_model
        self._sim = simulator
        self._action_names = action_names  # "pick up" -> 3, "open" -> 5, etc.

    def plan(
        self,
        chunks: list[Chunk],
        current_sks: set[int],
    ) -> list[int]:
        """Convert instruction chunks to action sequence.

        Handles sequential instructions (SEQ_BREAK) and causal prerequisites.

        Args:
            chunks: output of chunker.chunk().
            current_sks: current world state (active SKS IDs).

        Returns:
            List of action IDs to execute.
        """
        sub_instructions = self._split_by_seq_break(chunks)
        actions: list[int] = []
        state = set(current_sks)

        for sub_chunks in sub_instructions:
            sub_actions = self._plan_single(sub_chunks, state)
            actions.extend(sub_actions)
            # Update state estimate after each sub-instruction
            for a in sub_actions:
                effect, _ = self._causal.predict_effect(state, a)
                if effect:
                    state = state | effect

        return actions

    def _plan_single(
        self, chunks: list[Chunk], current_sks: set[int],
    ) -> list[int]:
        """Plan a single (non-sequential) instruction."""
        action_name = None
        object_sks = None

        for chunk in chunks:
            if chunk.role == "ACTION":
                action_name = chunk.text
            elif chunk.role == "OBJECT":
                object_sks = self._gmap.word_to_sks(chunk.text)

        if action_name is None:
            return []

        action_id = self._action_names.get(action_name)
        if action_id is None:
            return []

        # Check if action has a prerequisite
        prereq_actions = self._find_prerequisites(action_id, object_sks, current_sks)

        return prereq_actions + [action_id]

    def _find_prerequisites(
        self,
        action_id: int,
        object_sks: int | None,
        current_sks: set[int],
    ) -> list[int]:
        """Find prerequisite actions via causal model.

        Example: open(door) requires key_held. If key not in current_sks,
        need pickup(key) first.

        Strategy: check if action has known effect. If not (confidence=0),
        scan causal links for what context enables this action to succeed.
        """
        # Try the action in current state
        effect, conf = self._causal.predict_effect(current_sks, action_id)
        if conf > 0:
            # Action works in current state, no prerequisites
            return []

        # Action doesn't work. Search for enabling conditions.
        # Look at all causal links for this action.
        links = self._causal.get_causal_links(min_confidence=0.1)
        enabling_contexts: list[frozenset[int]] = []
        for link in links:
            if link.action == action_id and link.strength > 0.1:
                enabling_contexts.append(link.context_sks)

        if not enabling_contexts:
            return []

        # Find what's missing from current state
        for ctx in enabling_contexts:
            missing = ctx - frozenset(current_sks)
            if not missing:
                # We already have the context, action should work
                return []

            # For each missing SKS, find an action that produces it
            prereqs: list[int] = []
            for m_sks in missing:
                producing_action = self._find_action_producing(m_sks, current_sks)
                if producing_action is not None:
                    prereqs.append(producing_action)
            if prereqs:
                return prereqs

        return []

    def _find_action_producing(
        self, target_sks: int, current_sks: set[int],
    ) -> int | None:
        """Find an action that produces target_sks in current state."""
        links = self._causal.get_causal_links(min_confidence=0.1)
        for link in links:
            if target_sks in link.effect_sks:
                return link.action
        return None

    @staticmethod
    def _split_by_seq_break(chunks: list[Chunk]) -> list[list[Chunk]]:
        """Split chunks at SEQ_BREAK markers."""
        result: list[list[Chunk]] = []
        current: list[Chunk] = []
        for chunk in chunks:
            if chunk.role == "SEQ_BREAK":
                if current:
                    result.append(current)
                    current = []
            else:
                current.append(chunk)
        if current:
            result.append(current)
        return result
