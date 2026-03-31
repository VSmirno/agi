"""Verbalizer: converts world model state into text (Stage 21).

Interface layer, not cognitive. Templates produce exact transmission
of world model content — no NLG, no "pretty" phrasing.
"""

from __future__ import annotations

from snks.agent.causal_model import CausalWorldModel
from snks.language.grounding_map import GroundingMap
from snks.language.templates import causal_template, describe_template, plan_template


class Verbalizer:
    """Verbalizes world model state, causal links, and plans.

    Args:
        grounding_map: bidirectional SKS ↔ word mapping.
        action_names: {action_id: human-readable name}, e.g. {3: "pick up"}.
    """

    def __init__(
        self,
        grounding_map: GroundingMap,
        action_names: dict[int, str],
    ) -> None:
        self._gmap = grounding_map
        self._action_names = action_names

    def describe_state(self, active_sks_ids: list[int]) -> str:
        """Verbalize active SKS as object list.

        Filters to only SKS that have a grounding label.
        Recall = fraction of grounded SKS mentioned.
        Precision = no false objects (guaranteed by design).
        """
        objects: list[str] = []
        for sks_id in active_sks_ids:
            word = self._gmap.sks_to_word(sks_id)
            if word is not None:
                objects.append(word)
        return describe_template(objects)

    def explain_causal(
        self, sks_id: int, causal_model: CausalWorldModel,
    ) -> str:
        """Find and verbalize causal links involving sks_id.

        Strategy: scan all causal links, find those where sks_id appears
        in context_sks or effect_sks. For each link, pick the "main" SKS
        (first one with a grounding label) from context and effect.

        Returns empty string if no relevant links found.
        """
        links = causal_model.get_causal_links(min_confidence=0.3)
        if not links:
            return ""

        results: list[str] = []
        for link in links:
            # Check if sks_id is involved in this link.
            in_context = sks_id in link.context_sks
            in_effect = sks_id in link.effect_sks
            if not in_context and not in_effect:
                continue

            action_name = self._action_names.get(link.action, f"action_{link.action}")
            obj_word = self._find_grounded(link.context_sks)
            effect_word = self._find_grounded(link.effect_sks)

            if obj_word and effect_word:
                results.append(causal_template(action_name, obj_word, effect_word))

        if not results:
            return ""
        return "; ".join(results)

    def verbalize_plan(
        self,
        action_ids: list[int],
        initial_sks: set[int],
        causal_model: CausalWorldModel,
    ) -> str:
        """Verbalize a plan (action sequence) by reconstructing SKS chain.

        For each action, uses causal_model.predict_effect to get the
        resulting SKS, then looks up grounding labels for the effects.
        """
        if not action_ids:
            return ""

        steps: list[tuple[str, str]] = []
        state = set(initial_sks)

        for action_id in action_ids:
            action_name = self._action_names.get(action_id, f"action_{action_id}")
            effect_sks, _conf = causal_model.predict_effect(state, action_id)

            # effect_sks is the symmetric difference (new/changed SKS).
            # Look for grounded word in the new effect first.
            effect_word = self._find_grounded(effect_sks)
            if effect_word is None:
                # Fallback: try context SKS (what the action acts on).
                effect_word = self._find_grounded(state) or "unknown"

            steps.append((action_name, effect_word))
            state = state | effect_sks

        return plan_template(steps)

    def _find_grounded(self, sks_ids: frozenset[int] | set[int]) -> str | None:
        """Find first SKS in the set that has a grounding label."""
        for sks_id in sorted(sks_ids):
            word = self._gmap.sks_to_word(sks_id)
            if word is not None:
                return word
        return None
