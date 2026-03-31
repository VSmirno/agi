"""GridPerception: MiniGrid grid state → SKS concept IDs (Stage 24c).

Scaffold replacing DAF visual encoder for grid-world environments.
Maps each unique (object_type, color) pair to a stable SKS concept ID,
and tracks agent position/direction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from snks.language.grounding_map import GroundingMap


# MiniGrid object types we care about (skip wall, empty, unseen, floor).
INTERACTIVE_OBJECTS = frozenset({"key", "ball", "box", "door", "goal"})

# Reserved SKS IDs for agent state.
SKS_AGENT = 1
SKS_AGENT_CARRYING = 2

# State-dependent SKS IDs (stable range 50-99, preserved by _split_context).
SKS_KEY_PRESENT = 50     # key visible on grid floor
SKS_KEY_HELD = 51        # agent carrying key
SKS_DOOR_LOCKED = 52     # door exists and is_locked
SKS_DOOR_OPEN = 53       # door exists and is_open
SKS_GOAL_PRESENT = 54    # goal cell visible


@dataclass
class GridObject:
    """A detected object in the grid."""

    obj_type: str
    color: str
    pos: tuple[int, int]
    sks_id: int


class GridPerception:
    """Extracts SKS concepts directly from MiniGrid grid state.

    Each unique (object_type, color) pair gets a stable SKS ID.
    Registers words in GroundingMap for bidirectional lookup.
    """

    def __init__(self, grounding_map: GroundingMap, sdr_size: int = 256) -> None:
        self._gmap = grounding_map
        self._sdr_size = sdr_size
        self._type_color_to_sks: dict[tuple[str, str], int] = {}
        self._next_sks_id: int = 100  # start above reserved IDs
        self._objects: list[GridObject] = []  # last perceive() result

    def register_object(self, obj_type: str, color: str) -> int:
        """Register a MiniGrid object as a grounded concept.

        Creates bidirectional mapping:
        - (obj_type, color) → sks_id
        - "{color} {obj_type}" → sks_id (via GroundingMap)
        - "{obj_type}" → sks_id (bare noun, if not already registered)

        Returns:
            Assigned SKS ID.
        """
        key = (obj_type, color)
        if key in self._type_color_to_sks:
            return self._type_color_to_sks[key]

        sks_id = self._next_sks_id
        self._next_sks_id += 1
        self._type_color_to_sks[key] = sks_id

        # Create a dummy SDR for GroundingMap registration.
        sdr = torch.zeros(self._sdr_size)
        sdr[sks_id % self._sdr_size] = 1.0

        # Register bare noun first (if not yet mapped), then composite.
        # GroundingMap.register overwrites sks_to_word, so composite goes last
        # to ensure sks_to_word[sks_id] == "red key" (not just "key").
        if self._gmap.word_to_sks(obj_type) is None:
            self._gmap.register(obj_type, sks_id, sdr)

        # Register composite word "red key" — this overwrites sks_to_word.
        composite_word = f"{color} {obj_type}"
        self._gmap.register(composite_word, sks_id, sdr)

        # Also register color as attribute concept.
        if self._gmap.word_to_sks(color) is None:
            color_sks = self._next_sks_id
            self._next_sks_id += 1
            color_sdr = torch.zeros(self._sdr_size)
            color_sdr[color_sks % self._sdr_size] = 1.0
            self._gmap.register(color, color_sks, color_sdr)

        return sks_id

    def perceive(self, grid, agent_pos, agent_dir, carrying=None) -> set[int]:
        """Extract active SKS IDs from current grid state.

        Scans the full grid (scaffolding — bypasses partial observability).
        Also emits state-dependent predicates (key_held, door_locked, etc.).

        Args:
            grid: MiniGrid Grid object.
            agent_pos: (x, y) agent position.
            agent_dir: agent direction (0=right, 1=down, 2=left, 3=up).
            carrying: env.unwrapped.carrying (WorldObject or None).

        Returns:
            Set of active SKS IDs.
        """
        active_sks: set[int] = {SKS_AGENT}
        self._objects = []
        has_key_on_floor = False

        for j in range(grid.height):
            for i in range(grid.width):
                cell = grid.get(i, j)
                if cell is None:
                    continue
                if cell.type not in INTERACTIVE_OBJECTS:
                    continue

                sks_id = self.register_object(cell.type, cell.color)
                active_sks.add(sks_id)
                self._objects.append(GridObject(
                    obj_type=cell.type,
                    color=cell.color,
                    pos=(i, j),
                    sks_id=sks_id,
                ))

                # State predicates.
                if cell.type == "key":
                    has_key_on_floor = True
                    active_sks.add(SKS_KEY_PRESENT)
                elif cell.type == "door":
                    if getattr(cell, "is_locked", False):
                        active_sks.add(SKS_DOOR_LOCKED)
                    if getattr(cell, "is_open", False):
                        active_sks.add(SKS_DOOR_OPEN)
                elif cell.type == "goal":
                    active_sks.add(SKS_GOAL_PRESENT)

        # Agent carrying state.
        if carrying is not None and getattr(carrying, "type", None) == "key":
            active_sks.add(SKS_KEY_HELD)

        return active_sks

    def find_object(self, word: str) -> GridObject | None:
        """Find a perceived object by word (e.g. 'red key', 'key', 'ball').

        Searches last perceive() results.
        """
        # Try exact composite match first.
        for obj in self._objects:
            composite = f"{obj.color} {obj.obj_type}"
            if composite == word:
                return obj

        # Try bare noun match.
        for obj in self._objects:
            if obj.obj_type == word:
                return obj

        # Try color + any object.
        parts = word.split()
        if len(parts) == 2:
            color, obj_type = parts
            for obj in self._objects:
                if obj.color == color and obj.obj_type == obj_type:
                    return obj

        return None

    def find_object_at(self, pos: tuple[int, int]) -> GridObject | None:
        """Find a perceived object at a specific grid position."""
        for obj in self._objects:
            if obj.pos == pos:
                return obj
        return None

    def find_object_by_sks(self, sks_id: int) -> GridObject | None:
        """Find a perceived object by SKS ID."""
        for obj in self._objects:
            if obj.sks_id == sks_id:
                return obj
        return None

    @property
    def objects(self) -> list[GridObject]:
        """Last perceived objects."""
        return list(self._objects)
