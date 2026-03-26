"""AgentTransitionBuffer: fixed-capacity buffer for (pre_sks, action, post_sks, importance).

Used by ConsolidationScheduler and ReplayEngine (Stage 16).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class AgentTransition:
    """A single agent transition record."""
    pre_sks: set[int]
    action: int
    post_sks: set[int]
    importance: float
    pre_nodes: set[int] = field(default_factory=set)  # actual DAF node indices


class AgentTransitionBuffer:
    """Simple fixed-capacity buffer for (pre_sks, action, post_sks, importance) transitions."""

    def __init__(self, capacity: int = 200) -> None:
        self._buf: deque[AgentTransition] = deque(maxlen=capacity)

    def add(
        self,
        pre_sks: set[int],
        action: int,
        post_sks: set[int],
        importance: float,
        pre_nodes: set[int] | None = None,
    ) -> None:
        """Append a transition. Oldest entry is evicted when at capacity."""
        self._buf.append(AgentTransition(
            pre_sks, action, post_sks, importance,
            pre_nodes=pre_nodes or set(),
        ))

    def get_top_k(self, k: int, by: str = "importance") -> list[AgentTransition]:
        """Return top-k transitions sorted by the given field (descending)."""
        return sorted(self._buf, key=lambda t: t.importance, reverse=True)[:k]

    def __len__(self) -> int:
        return len(self._buf)
