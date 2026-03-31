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
    SKS_GATE_LOCKED,
    SKS_GATE_OPEN,
    SKS_CARD_HELD,
    SKS_CARD_PRESENT,
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
    target_pos: tuple[int, int] | None = None  # specific position (for multi-object envs)


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

        # Scan for locked or closed doors, prioritize by distance to agent.
        blockers: list[tuple[int, Blocker]] = []
        ax, ay = agent_pos
        for j in range(grid.height):
            for i in range(grid.width):
                cell = grid.get(i, j)
                if cell is None:
                    continue
                if cell.type == "door":
                    is_locked = getattr(cell, "is_locked", False)
                    is_open = getattr(cell, "is_open", False)
                    if is_locked or not is_open:
                        state = "locked" if is_locked else "closed"
                        sks_id = SKS_GATE_LOCKED if cell.color == "purple" else SKS_DOOR_LOCKED
                        dist = abs(i - ax) + abs(j - ay)
                        blockers.append((dist, Blocker(
                            cell_type="door",
                            cell_color=cell.color,
                            pos=(i, j),
                            state=state,
                            sks_id=sks_id,
                        )))

        if blockers:
            # Return closest blocker to agent
            blockers.sort(key=lambda x: x[0])
            return blockers[0][1]

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
        is_gate = (blocker.cell_type == "door" and blocker.cell_color == "purple")
        if blocker.cell_type == "door" and blocker.state in ("locked", "closed"):
            desired = frozenset({SKS_GATE_OPEN if is_gate else SKS_DOOR_OPEN})
        else:
            return None

        # Closed (not locked) doors: just toggle, no prerequisites
        if blocker.state == "closed":
            return SubGoal(
                action="toggle",
                target_word="gate" if is_gate else blocker.cell_type,
                target_sks=SKS_GATE_OPEN if is_gate else SKS_DOOR_OPEN,
                prerequisite=None,
                target_pos=blocker.pos,
            )

        results = causal_model.query_by_effect(desired)
        if not results:
            # For gate, also try querying by DOOR_OPEN (analogically mapped).
            if is_gate:
                results = causal_model.query_by_effect(frozenset({SKS_DOOR_OPEN}))
            if not results:
                # No causal knowledge yet — return None so agent explores
                return None

        action_id, required_context, _confidence = results[0]

        # Map action_id to name.
        action_name = {ACT_PICKUP: "pickup", ACT_TOGGLE: "toggle"}.get(
            action_id, f"action_{action_id}"
        )

        # Target word depends on object type.
        blocker_word = "gate" if is_gate else blocker.cell_type
        done_sks = SKS_GATE_OPEN if is_gate else SKS_DOOR_OPEN

        # Check prerequisites: what's in required_context that we don't have?
        prerequisite = None
        missing = required_context - frozenset(current_sks)

        # Key/card held prerequisite.
        need_instrument = SKS_KEY_HELD in missing or SKS_CARD_HELD in missing
        if need_instrument:
            instrument_sks = SKS_CARD_HELD if is_gate else SKS_KEY_HELD
            instrument_word = "card" if is_gate else "key"
            # Find how to get instrument.
            inst_results = causal_model.query_by_effect(frozenset({instrument_sks}))
            if not inst_results and is_gate:
                # Fall back to key pickup (same action).
                inst_results = causal_model.query_by_effect(frozenset({SKS_KEY_HELD}))
            if inst_results:
                inst_action, _inst_ctx, _ = inst_results[0]
                inst_action_name = {ACT_PICKUP: "pickup", ACT_TOGGLE: "toggle"}.get(
                    inst_action, f"action_{inst_action}"
                )
                prerequisite = SubGoal(
                    action=inst_action_name,
                    target_word=instrument_word,
                    target_sks=instrument_sks,
                    prerequisite=None,
                )

        return SubGoal(
            action=action_name,
            target_word=blocker_word,
            target_sks=done_sks,
            prerequisite=prerequisite,
            target_pos=blocker.pos,
        )
