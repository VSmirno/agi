"""SkillLibrary: extract, store, compose, and match skills (Stage 27)."""

from __future__ import annotations

from snks.agent.causal_model import CausalWorldModel
from snks.language.grid_perception import (
    SKS_DOOR_LOCKED,
    SKS_DOOR_OPEN,
    SKS_GOAL_PRESENT,
    SKS_KEY_HELD,
    SKS_KEY_PRESENT,
)
from snks.language.skill import Skill

# State predicate range (stable across environments).
_STATE_RANGE = range(50, 100)

# "Positive" predicates — the desired outcomes of actions.
# Used to filter symmetric-difference effects from CausalLink.
_POSITIVE_PREDICATES = frozenset({SKS_KEY_HELD, SKS_DOOR_OPEN, SKS_GOAL_PRESENT})

# Action ID → (skill_name_verb, target_word).
_ACTION_SKILL_MAP: dict[int, tuple[str, str]] = {
    3: ("pickup", "key"),    # ACT_PICKUP
    5: ("toggle", "door"),   # ACT_TOGGLE
}

MAX_COMPOSITION_DEPTH = 3


class SkillLibrary:
    """Stores, extracts, and composes reusable skills."""

    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}

    def register(self, skill: Skill) -> None:
        self._skills[skill.name] = skill

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    @property
    def skills(self) -> list[Skill]:
        return list(self._skills.values())

    def extract_from_causal_model(
        self,
        model: CausalWorldModel,
        min_confidence: float = 0.5,
    ) -> int:
        """Extract skills from high-confidence causal links.

        Returns number of NEW skills extracted.
        """
        new_count = 0
        for link in model.get_causal_links(min_confidence):
            if link.action not in _ACTION_SKILL_MAP:
                continue

            verb, target = _ACTION_SKILL_MAP[link.action]

            # Preconditions: state predicates from context.
            preconditions = frozenset(
                s for s in link.context_sks if s in _STATE_RANGE
            )
            if not preconditions:
                continue

            # Effects: only positive predicates from symmetric difference.
            effects = frozenset(
                s for s in link.effect_sks if s in _POSITIVE_PREDICATES
            )
            if not effects:
                continue

            name = f"{verb}_{target}"

            if name in self._skills:
                # Update observation count.
                existing = self._skills[name]
                existing.attempt_count += link.count
                existing.success_count += link.count
                continue

            skill = Skill(
                name=name,
                preconditions=preconditions,
                effects=effects,
                terminal_action=link.action,
                target_word=target,
                success_count=link.count,
                attempt_count=link.count,
            )
            self._skills[name] = skill
            new_count += 1

        return new_count

    def compose_skills(self) -> int:
        """Create composite skills by chaining primitives.

        Chain A→B if A.effects ∩ B.preconditions ≠ ∅.
        Returns number of new composite skills.
        """
        primitives = [s for s in self._skills.values() if not s.is_composite]
        new_count = 0

        # Depth 2: A→B pairs.
        pairs: list[tuple[Skill, Skill]] = []
        for a in primitives:
            for b in primitives:
                if a.name == b.name:
                    continue
                if a.effects & b.preconditions:
                    pairs.append((a, b))

        for a, b in pairs:
            name = f"{a.name}+{b.name}"
            if name in self._skills:
                continue
            composite = Skill(
                name=name,
                preconditions=a.preconditions,
                effects=b.effects,
                terminal_action=None,
                target_word=b.target_word,
                sub_skills=[a.name, b.name],
            )
            self._skills[name] = composite
            new_count += 1

        # Depth 3: extend existing depth-2 composites.
        if new_count > 0:
            depth2 = [s for s in self._skills.values() if s.is_composite and len(s.sub_skills) == 2]
            for comp in depth2:
                for c in primitives:
                    if comp.effects & c.preconditions and c.name not in comp.sub_skills:
                        name = f"{comp.name}+{c.name}"
                        if name in self._skills:
                            continue
                        extended = Skill(
                            name=name,
                            preconditions=comp.preconditions,
                            effects=c.effects,
                            terminal_action=None,
                            target_word=c.target_word,
                            sub_skills=comp.sub_skills + [c.name],
                        )
                        self._skills[name] = extended
                        new_count += 1

        return new_count

    def find_applicable(
        self,
        current_sks: set[int],
        goal_sks: frozenset[int] | None = None,
    ) -> list[Skill]:
        """Find skills whose preconditions are met.

        If goal_sks given, prefer skills whose effects intersect goal.
        Returns sorted: composites first, then by success_rate.
        """
        applicable = []
        for skill in self._skills.values():
            if skill.preconditions <= frozenset(current_sks):
                applicable.append(skill)

        if goal_sks:
            # Split into goal-relevant and others.
            relevant = [s for s in applicable if s.effects & goal_sks]
            others = [s for s in applicable if not (s.effects & goal_sks)]
            # Composites first, then by success rate.
            relevant.sort(key=lambda s: (not s.is_composite, -s.success_rate))
            return relevant + others

        applicable.sort(key=lambda s: (not s.is_composite, -s.success_rate))
        return applicable
