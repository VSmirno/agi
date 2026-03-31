"""BlockingAnalyzer: obstacle detection + causal resolution (Stage 25).

Identifies what blocks a path (locked door) and queries CausalWorldModel
for actions that could remove the blocker (backward chaining).
"""

from __future__ import annotations

from dataclasses import dataclass

from snks.agent.causal_model import CausalWorldModel
from snks.language.grid_navigator import _bfs
from snks.language.grid_perception import (
    SKS_DOOR_LOCKED,
    SKS_DOOR_OPEN,
    SKS_KEY_HELD,
    SKS_KEY_PRESENT,
)

# Action IDs.
ACT_PICKUP = 3
ACT_TOGGLE = 5

# Max backward chaining depth.
MAX_CHAIN_DEPTH = 3


@dataclass
class Blocker:
    """An obstacle blocking a path."""

    cell_type: str          # "door"
    cell_color: str         # "yellow"
    pos: tuple[int, int]
    state: str              # "locked"
    sks_id: int             # e.g. SKS_DOOR_LOCKED


@dataclass
class SubGoal:
    """A sub-goal in a backward-chained plan."""

    action: str             # "pickup", "toggle"
    target_word: str        # "key", "door"
    target_sks: int         # SKS concept for the target state
    prerequisite: SubGoal | None  # recursive chain


class BlockingAnalyzer:
    """Identifies path blockers and suggests causal resolutions."""

    def find_blocker(self, grid, agent_pos, target_pos) -> Blocker | None:
        """Find what blocks the path from agent to target.

        If BFS succeeds, path is clear. If BFS fails, scan grid for
        locked doors (the most common blocker in MiniGrid).
        """
        path = _bfs(grid, agent_pos, target_pos)
        if path is not None:
            return None  # path is clear

        # Scan for locked doors.
        for j in range(grid.height):
            for i in range(grid.width):
                cell = grid.get(i, j)
                if cell is None:
                    continue
                if cell.type == "door" and getattr(cell, "is_locked", False):
                    return Blocker(
                        cell_type="door",
                        cell_color=cell.color,
                        pos=(i, j),
                        state="locked",
                        sks_id=SKS_DOOR_LOCKED,
                    )

        return None  # blocked but no identifiable blocker

    def suggest_resolution(
        self,
        blocker: Blocker,
        causal_model: CausalWorldModel,
        current_sks: set[int],
        depth: int = 0,
    ) -> SubGoal | None:
        """Query CausalWorldModel for action that removes blocker.

        Uses backward chaining: if toggle(door) requires key_held,
        and key_held is missing, recursively find how to get key_held.
        """
        if depth >= MAX_CHAIN_DEPTH:
            return None

        # What state do we want?
        if blocker.cell_type == "door" and blocker.state == "locked":
            desired = frozenset({SKS_DOOR_OPEN})
        else:
            return None

        results = causal_model.query_by_effect(desired)
        if not results:
            return None

        action_id, required_context, _confidence = results[0]

        # Map action_id to name.
        action_name = {ACT_PICKUP: "pickup", ACT_TOGGLE: "toggle"}.get(
            action_id, f"action_{action_id}"
        )

        # Check prerequisites: what's in required_context that we don't have?
        prerequisite = None
        missing = required_context - frozenset(current_sks)
        if SKS_KEY_HELD in missing:
            # Need key_held → find how to get it.
            key_results = causal_model.query_by_effect(frozenset({SKS_KEY_HELD}))
            if key_results:
                key_action, _key_ctx, _ = key_results[0]
                key_action_name = {ACT_PICKUP: "pickup", ACT_TOGGLE: "toggle"}.get(
                    key_action, f"action_{key_action}"
                )
                prerequisite = SubGoal(
                    action=key_action_name,
                    target_word="key",
                    target_sks=SKS_KEY_HELD,
                    prerequisite=None,
                )

        return SubGoal(
            action=action_name,
            target_word=blocker.cell_type,
            target_sks=SKS_DOOR_OPEN,
            prerequisite=prerequisite,
        )
