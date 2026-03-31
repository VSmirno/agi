"""PlanNode: hierarchical plan representation (Stage 34).

A plan is a tree of PlanNodes at 3 levels:
- Level 2 (Strategic): abstract goals like "get_key", "unlock_door"
- Level 1 (Tactical): skill-level actions like "navigate_to", "pickup"
- Level 0 (Primitive): environment actions (forward, turn, pickup, toggle)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class PlanStatus(Enum):
    PENDING = "pending"
    ACTIVE = "active"
    DONE = "done"
    FAILED = "failed"
    REPLANNED = "replanned"


@dataclass
class PlanNode:
    """A node in a hierarchical plan tree."""

    level: int                                  # 0=primitive, 1=tactical, 2=strategic
    action: int | str                           # primitive action ID or skill/goal name
    preconditions: frozenset[int] = field(default_factory=frozenset)
    postconditions: frozenset[int] = field(default_factory=frozenset)
    children: list[PlanNode] = field(default_factory=list)
    estimated_steps: int = 1
    status: PlanStatus = PlanStatus.PENDING
    description: str = ""                       # human-readable (for logging)

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def is_done(self) -> bool:
        return self.status == PlanStatus.DONE

    @property
    def is_failed(self) -> bool:
        return self.status == PlanStatus.FAILED

    def total_estimated_steps(self) -> int:
        """Recursively sum estimated primitive steps."""
        if self.is_leaf:
            return self.estimated_steps
        return sum(c.total_estimated_steps() for c in self.children)

    def count_nodes(self) -> int:
        """Total nodes in subtree."""
        return 1 + sum(c.count_nodes() for c in self.children)

    def depth(self) -> int:
        """Max depth of subtree."""
        if self.is_leaf:
            return 0
        return 1 + max(c.depth() for c in self.children)

    def flatten_primitives(self) -> list[int]:
        """Extract ordered list of primitive actions from leaves."""
        if self.is_leaf and self.level == 0 and isinstance(self.action, int):
            return [self.action]
        result = []
        for child in self.children:
            result.extend(child.flatten_primitives())
        return result

    def validate_chain(self) -> float:
        """Check precondition/postcondition coherence through the plan.

        Returns fraction of transitions where child[i].postconditions
        satisfy child[i+1].preconditions.
        """
        if len(self.children) < 2:
            return 1.0

        valid = 0
        total = 0
        # Accumulate state through children.
        accumulated_state: set[int] = set(self.preconditions)
        for child in self.children:
            total += 1
            if child.preconditions <= frozenset(accumulated_state):
                valid += 1
            accumulated_state |= set(child.postconditions)

        return valid / total if total > 0 else 1.0


@dataclass
class PlanGraph:
    """Top-level plan container."""

    root: PlanNode
    goal_sks: frozenset[int] = field(default_factory=frozenset)
    initial_sks: frozenset[int] = field(default_factory=frozenset)

    @property
    def total_steps(self) -> int:
        return self.root.total_estimated_steps()

    @property
    def coherence(self) -> float:
        """Global plan coherence."""
        return self.root.validate_chain()

    @property
    def n_nodes(self) -> int:
        return self.root.count_nodes()

    def get_next_pending(self) -> PlanNode | None:
        """DFS for first pending leaf node."""
        return self._find_pending(self.root)

    def _find_pending(self, node: PlanNode) -> PlanNode | None:
        if node.status == PlanStatus.DONE or node.status == PlanStatus.FAILED:
            return None
        if node.is_leaf and node.status == PlanStatus.PENDING:
            return node
        for child in node.children:
            result = self._find_pending(child)
            if result is not None:
                return result
        return None

    def mark_complete_up(self, node: PlanNode) -> None:
        """Mark node done and propagate up if all siblings done."""
        node.status = PlanStatus.DONE
        # Walk up through root checking if parents are fully done.
        self._propagate(self.root)

    def _propagate(self, node: PlanNode) -> bool:
        if node.is_leaf:
            return node.is_done
        all_done = all(self._propagate(c) for c in node.children)
        if all_done and node.status != PlanStatus.DONE:
            node.status = PlanStatus.DONE
        return all_done
