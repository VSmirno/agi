"""GridPerception: MiniGrid grid state → SKS concept IDs (Stage 24c).

Scaffold replacing DAF visual encoder for grid-world environments.
Maps each unique (object_type, color) pair to a stable SKS concept ID,
and tracks agent position/direction.

Extended in Stage 28 with card/gate analogical predicates (55-58).
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

# Analogical predicates — card/gate world (Stage 28).
# purple key → "card"; purple door → "gate".
SKS_CARD_PRESENT = 55    # card (purple key) visible on grid floor
SKS_CARD_HELD = 56       # agent carrying card (purple key)
SKS_GATE_LOCKED = 57     # gate (purple door) exists and is_locked
SKS_GATE_OPEN = 58       # gate (purple door) exists and is_open


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

        for j in range(grid.height):
            for i in range(grid.width):
                cell = grid.get(i, j)
                if cell is None:
                    continue
                if cell.type not in INTERACTIVE_OBJECTS:
                    continue

                sks_id = self.register_object(cell.type, cell.color)
                active_sks.add(sks_id)

                # For purple key/door, also register under alias word.
                obj_type = cell.type
                color = cell.color
                if obj_type == "key" and color == "purple":
                    # Register "card" as alias for purple key.
                    self._register_alias("card", sks_id)
                elif obj_type == "door" and color == "purple":
                    # Register "gate" as alias for purple door.
                    self._register_alias("gate", sks_id)

                self._objects.append(GridObject(
                    obj_type=obj_type,
                    color=color,
                    pos=(i, j),
                    sks_id=sks_id,
                ))

                # State predicates — with card/gate analogy (Stage 28).
                if obj_type == "key":
                    if color == "purple":
                        active_sks.add(SKS_CARD_PRESENT)
                    else:
                        active_sks.add(SKS_KEY_PRESENT)
                elif obj_type == "door":
                    locked = getattr(cell, "is_locked", False)
                    opened = getattr(cell, "is_open", False)
                    if color == "purple":
                        if locked:
                            active_sks.add(SKS_GATE_LOCKED)
                        if opened:
                            active_sks.add(SKS_GATE_OPEN)
                    else:
                        if locked:
                            active_sks.add(SKS_DOOR_LOCKED)
                        if opened:
                            active_sks.add(SKS_DOOR_OPEN)
                elif obj_type == "goal":
                    active_sks.add(SKS_GOAL_PRESENT)

        # Agent carrying state.
        if carrying is not None:
            c_type = getattr(carrying, "type", None)
            c_color = getattr(carrying, "color", None)
            if c_type == "key":
                if c_color == "purple":
                    active_sks.add(SKS_CARD_HELD)
                else:
                    active_sks.add(SKS_KEY_HELD)

        return active_sks

    def _register_alias(self, word: str, sks_id: int) -> None:
        """Register an alias word for an existing SKS ID (e.g. 'card' for purple key)."""
        if self._gmap.word_to_sks(word) is None:
            sdr = torch.zeros(self._sdr_size)
            sdr[sks_id % self._sdr_size] = 1.0
            self._gmap.register(word, sks_id, sdr)

    def find_object(self, word: str) -> GridObject | None:
        """Find a perceived object by word (e.g. 'red key', 'key', 'ball', 'card', 'gate').

        Supports Stage 28 analogical aliases:
          - 'card' → purple key
          - 'gate' → purple door

        Searches last perceive() results.
        """
        # Analogical aliases (Stage 28).
        if word == "card":
            for obj in self._objects:
                if obj.obj_type == "key" and obj.color == "purple":
                    return obj
            return None
        if word == "gate":
            for obj in self._objects:
                if obj.obj_type == "door" and obj.color == "purple":
                    return obj
            return None

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
