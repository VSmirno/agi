"""Skill: reusable macro-action with preconditions and effects (Stage 27)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Skill:
    """A reusable macro-action extracted from causal knowledge."""

    name: str
    preconditions: frozenset[int]       # required SKS before execution
    effects: frozenset[int]             # expected SKS changes after
    terminal_action: int | None         # primitive action ID or None for composite
    target_word: str                    # "key", "door", "goal"
    sub_skills: list[str] | None = None # composite: ordered skill names
    success_count: int = 0
    attempt_count: int = 0

    @property
    def success_rate(self) -> float:
        return self.success_count / max(self.attempt_count, 1)

    @property
    def is_composite(self) -> bool:
        return self.sub_skills is not None and len(self.sub_skills) > 0
