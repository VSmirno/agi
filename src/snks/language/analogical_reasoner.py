"""AnalogicalReasoner: transfer skills by structural analogy (Stage 28).

Detects structural similarity between Skill predicates and adapts
known skills to new environments using ROLE_REGISTRY mappings.
"""

from __future__ import annotations

from dataclasses import dataclass

from snks.language.skill import Skill
from snks.language.skill_library import SkillLibrary
from snks.language.role_registry import ROLE_REGISTRY, SOURCE_TO_TARGET_SKS, WORD_ROLE_MAPPING


@dataclass
class AnalogyMap:
    """Represents an analogical mapping from a source skill to an adapted skill."""

    source_skill_name: str
    adapted_skill: Skill          # ready-to-use skill with mapped predicates
    sks_mapping: dict[int, int]   # {source_sks: target_sks}
    role_mapping: dict[str, str]  # {"key": "card", "door": "gate"}
    similarity: float             # fraction of source predicates with known analog


class AnalogicalReasoner:
    """Finds analogical mappings from a SkillLibrary to a new environment.

    Uses ROLE_REGISTRY to compute structural similarity between source skill
    predicates and target environment predicates, then adapts matching skills.
    """

    def __init__(self, threshold: float = 0.7) -> None:
        self._threshold = threshold

    def find_analogy(
        self,
        library: SkillLibrary,
        target_sks: set[int],
        threshold: float | None = None,
    ) -> list[AnalogyMap]:
        """Find skills that can be analogically applied to the current state.

        Args:
            library: Source skill library (trained on key/door world).
            target_sks: Current state predicates in target environment.
            threshold: Similarity threshold (default: self._threshold).

        Returns:
            List of AnalogyMap ordered by similarity (descending).
        """
        thr = threshold if threshold is not None else self._threshold
        analogies: list[AnalogyMap] = []

        for skill in library.skills:
            analogy = self._compute_analogy(skill, target_sks)
            if analogy is not None and analogy.similarity >= thr:
                analogies.append(analogy)

        # Sort by similarity descending, composites first.
        analogies.sort(
            key=lambda a: (a.adapted_skill.is_composite, a.similarity),
            reverse=True,
        )
        return analogies

    def _compute_analogy(
        self, skill: Skill, target_sks: set[int],
    ) -> AnalogyMap | None:
        """Compute analogical similarity between skill and target environment.

        Similarity = fraction of skill's predicates that have a known analog
        in SOURCE_TO_TARGET_SKS.
        """
        all_predicates = skill.preconditions | skill.effects
        if not all_predicates:
            return None

        matched = 0
        sks_mapping: dict[int, int] = {}
        for pred in all_predicates:
            if pred in SOURCE_TO_TARGET_SKS:
                sks_mapping[pred] = SOURCE_TO_TARGET_SKS[pred]
                matched += 1

        similarity = matched / len(all_predicates)
        if similarity == 0.0:
            return None

        # Build adapted skill with mapped predicates.
        adapted = self._adapt_skill(skill, sks_mapping)

        return AnalogyMap(
            source_skill_name=skill.name,
            adapted_skill=adapted,
            sks_mapping=sks_mapping,
            role_mapping=WORD_ROLE_MAPPING,
            similarity=similarity,
        )

    def _adapt_skill(self, skill: Skill, sks_mapping: dict[int, int]) -> Skill:
        """Create a new Skill with predicates remapped to target domain."""
        new_preconditions = frozenset(
            sks_mapping.get(p, p) for p in skill.preconditions
        )
        new_effects = frozenset(
            sks_mapping.get(e, e) for e in skill.effects
        )
        # Remap target_word via WORD_ROLE_MAPPING.
        new_target_word = WORD_ROLE_MAPPING.get(skill.target_word, skill.target_word)

        # For composite skills, adapt sub_skill names.
        new_sub_skills: list[str] | None = None
        if skill.sub_skills:
            new_sub_skills = [
                f"adapted_{s}" for s in skill.sub_skills
            ]

        return Skill(
            name=f"adapted_{skill.name}",
            preconditions=new_preconditions,
            effects=new_effects,
            terminal_action=skill.terminal_action,
            target_word=new_target_word,
            sub_skills=new_sub_skills,
        )
