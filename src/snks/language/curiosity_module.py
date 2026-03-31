"""CuriosityModule: count-based intrinsic motivation (Stage 29).

Tracks state visit counts and computes intrinsic reward as novelty signal.
Intrinsic reward decays with repeated visits: r_int = 1.0 / (1 + count).
"""

from __future__ import annotations

from collections import defaultdict


class CuriosityModule:
    """Count-based curiosity: novel states get high intrinsic reward.

    State key = frozenset(sks_predicates) | {pos_hash} to distinguish
    grid positions even when SKS predicates are identical (empty rooms).
    """

    _BONUS = 1.0  # base intrinsic reward for unseen state

    def __init__(self) -> None:
        self._counts: dict[frozenset, int] = defaultdict(int)

    @staticmethod
    def make_key(sks: set[int], agent_pos: tuple[int, int]) -> frozenset:
        """Create a hashable state key including position."""
        # Encode position as a single int to avoid hash collision issues.
        pos_token = 10000 + agent_pos[0] * 100 + agent_pos[1]
        return frozenset(sks | {pos_token})

    def observe(self, state_key: frozenset) -> float:
        """Record a state visit. Returns intrinsic reward before incrementing."""
        r = self._BONUS / (1 + self._counts[state_key])
        self._counts[state_key] += 1
        return r

    def intrinsic_reward(self, state_key: frozenset) -> float:
        """Peek: return expected intrinsic reward without updating counts."""
        return self._BONUS / (1 + self._counts[state_key])

    def n_distinct(self) -> int:
        """Number of distinct states seen."""
        return len(self._counts)

    def count(self, state_key: frozenset) -> int:
        """How many times a state has been visited."""
        return self._counts[state_key]

    def reset(self) -> None:
        """Reset all visit counts (new episode)."""
        self._counts.clear()
