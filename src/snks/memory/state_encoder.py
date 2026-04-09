"""Stage 76 Phase 2: StateEncoder — raw state → SDR.

Projects agent's raw perception (inventory, visible field, spatial map,
body state) into a sparse binary vector (4096 bits, ~200 active).

NO derived features. Only bucket encoding of scalars, fixed SDRs for
presence, and spatial bit allocation for concept×distance bindings.

Bit layout (4096 total):
  [0,    400)   body stats: 4 variables × 100 bits (window 40)
                — actual variables come from tracker.observed_variables()
                — allocator pattern below handles dynamic set
  [400,  1000)  inventory scalars: lazily allocated from observed inv keys
                — up to 6 scalar items × 100 bits (wood, stone, coal, ...)
  [1000, 1400)  inventory presence: fixed SDRs for binary items
                (wood_sword, wood_pickaxe, table, etc.)
  [1400, 2400)  visible concepts × distance (SpatialRangeAllocator)
                — up to 10 concepts × 100 bits
  [2400, 3400)  spatial_map known concepts × distance (SpatialRangeAllocator)
                — up to 10 concepts × 100 bits (wider distance range)
  [3400, 4096)  reserved / padding

Dynamic allocation: any previously-unseen variable or concept auto-allocates
its bit range on first observation. Allows the encoder to work across
environments without hardcoded class lists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from snks.memory.sdr_encoder import (
    bucket_encode,
    FixedSDRRegistry,
    SpatialRangeAllocator,
)


# Bit layout constants
TOTAL_BITS = 4096

BODY_START = 0
BODY_END = 400
BODY_BITS_PER_VAR = 100
BODY_WINDOW = 40

INV_SCALAR_START = 400
INV_SCALAR_END = 1000
INV_SCALAR_BITS_PER_VAR = 100
INV_SCALAR_WINDOW = 40

INV_PRESENCE_START = 1000
INV_PRESENCE_END = 1400
PRESENCE_BITS_PER_ITEM = 40

VIS_START = 1400
VIS_END = 2400
VIS_BITS_PER_CONCEPT = 100
VIS_WINDOW = 40
VIS_DIST_MIN = 0
VIS_DIST_MAX = 9

KNOWN_START = 2400
KNOWN_END = 3400
KNOWN_BITS_PER_CONCEPT = 100
KNOWN_WINDOW = 40
KNOWN_DIST_MIN = 0
KNOWN_DIST_MAX = 30


@dataclass
class StateEncoder:
    """Deterministic projection of raw state → sparse binary SDR.

    Dynamically allocates bit ranges per observed variable / concept.
    Similarity-preserving by construction: similar states → overlapping bits.
    """

    total_bits: int = TOTAL_BITS

    # Allocators (initialized in __post_init__)
    body_alloc: SpatialRangeAllocator = field(init=False)
    inv_scalar_alloc: SpatialRangeAllocator = field(init=False)
    inv_presence_registry: FixedSDRRegistry = field(init=False)
    visible_alloc: SpatialRangeAllocator = field(init=False)
    known_alloc: SpatialRangeAllocator = field(init=False)

    def __post_init__(self) -> None:
        # Body: 4 variables max by default, each 100 bits
        self.body_alloc = SpatialRangeAllocator(
            start_bit=BODY_START,
            end_bit=BODY_END,
            bits_per_concept=BODY_BITS_PER_VAR,
        )
        # Inventory scalars: 6 variables max
        self.inv_scalar_alloc = SpatialRangeAllocator(
            start_bit=INV_SCALAR_START,
            end_bit=INV_SCALAR_END,
            bits_per_concept=INV_SCALAR_BITS_PER_VAR,
        )
        # Inventory presence: fixed SDRs within dedicated domain
        # The registry uses full TOTAL_BITS domain but we constrain output
        # by masking the relevant range. Simpler: keep registry with its
        # own small domain.
        self.inv_presence_registry = FixedSDRRegistry(
            total_bits=INV_PRESENCE_END - INV_PRESENCE_START,
            bits_per_concept=PRESENCE_BITS_PER_ITEM,
        )
        # Visible: 10 concepts × 100 bits
        self.visible_alloc = SpatialRangeAllocator(
            start_bit=VIS_START,
            end_bit=VIS_END,
            bits_per_concept=VIS_BITS_PER_CONCEPT,
        )
        # Known (spatial_map): 10 concepts × 100 bits
        self.known_alloc = SpatialRangeAllocator(
            start_bit=KNOWN_START,
            end_bit=KNOWN_END,
            bits_per_concept=KNOWN_BITS_PER_CONCEPT,
        )

    def encode(
        self,
        inventory: dict[str, int],
        visible_field: Any,  # snks.agent.perception.VisualField
        spatial_map: Any,    # snks.agent.crafter_spatial_map.CrafterSpatialMap
        player_pos: tuple[int, int],
        body_variables: set[str] | None = None,
    ) -> np.ndarray:
        """Encode raw state into a 4096-bit SDR.

        Args:
            inventory: dict of inventory items → counts (includes body stats).
            visible_field: VisualField with detections list.
            spatial_map: CrafterSpatialMap for known object positions.
            player_pos: (world_x, world_y).
            body_variables: explicit set of body variable names to encode as
                body-stat bucket (e.g., {"health", "food", "drink", "energy"}).
                If None, infer from inventory: any int value with max ≤ 9 is
                treated as body stat. Other ints are inventory scalars.
                Cleaner: pass tracker.observed_variables() from caller.

        Returns:
            Boolean array shape (4096,), ~200 active bits expected.
        """
        bits = np.zeros(self.total_bits, dtype=bool)

        # Infer body variables if not given (caller should pass explicit set)
        if body_variables is None:
            # Heuristic: body stats have typical range 0-9 in Crafter.
            # This is a fallback; production code should pass tracker.observed_variables().
            body_variables = {
                k for k, v in inventory.items() if isinstance(v, int) and 0 <= v <= 9
            }

        # === Body stats (bucket encoded) ===
        for var in sorted(body_variables):
            if var not in inventory:
                continue
            try:
                start, end = self.body_alloc.get_range(var)
            except ValueError:
                continue  # body range exhausted; skip
            value = inventory[var]
            var_bits = bucket_encode(
                value=float(value),
                value_min=0.0,
                value_max=9.0,
                start_bit=start,
                end_bit=end,
                window=BODY_WINDOW,
            )
            bits[start:end] |= var_bits

        # === Inventory: split into scalars and presence indicators ===
        # Scalars: items that take multiple values (wood, stone, coal, etc.)
        # Presence: binary items (wood_sword, wood_pickaxe, table)
        # Heuristic: presence if value ∈ {0, 1} AND name doesn't indicate count
        # Better: pass explicit lists from caller. For now, use simple rule:
        # - If var in body_variables: skip (already encoded above)
        # - Else: treat as inventory scalar OR presence based on max observed
        for item, value in inventory.items():
            if item in body_variables:
                continue
            if value is None:
                continue
            if not isinstance(value, (int, float)):
                continue

            # Presence items: name suggests binary tool/crafted item
            # (wood_sword, stone_sword, wood_pickaxe, etc.)
            # Scalar items: wood, stone_item, coal_item, etc.
            # We use the same rule: if value > 1 ever occurs, it's a scalar;
            # else presence. Since we can't know history here, use heuristic
            # on name: "_sword", "_pickaxe", "table" → presence.
            is_presence = self._is_presence_item(item)

            if is_presence:
                if value > 0:
                    pattern = self.inv_presence_registry.get(item)
                    # pattern is in local domain [0, INV_PRESENCE_END-INV_PRESENCE_START)
                    bits[INV_PRESENCE_START:INV_PRESENCE_END] |= pattern
            else:
                # Scalar count (wood, stone_item, ...)
                try:
                    start, end = self.inv_scalar_alloc.get_range(item)
                except ValueError:
                    continue
                # Cap value at a reasonable max for bucket encoding
                var_bits = bucket_encode(
                    value=float(min(value, 10)),
                    value_min=0.0,
                    value_max=10.0,
                    start_bit=start,
                    end_bit=end,
                    window=INV_SCALAR_WINDOW,
                )
                bits[start:end] |= var_bits

        # === Visible concepts × distance ===
        # For each unique detected concept, encode the MIN distance to
        # nearest instance (closest occurrence most relevant).
        vis_center = _viewport_center()  # (cr, cc) for 7×9 grid

        concept_min_dist: dict[str, int] = {}
        for detection in getattr(visible_field, "detections", []):
            cid, _, gy, gx = detection
            if cid == "empty":
                continue
            dist = abs(gy - vis_center[0]) + abs(gx - vis_center[1])
            if cid not in concept_min_dist or dist < concept_min_dist[cid]:
                concept_min_dist[cid] = dist

        for cid, dist in concept_min_dist.items():
            try:
                start, end, var_bits = self.visible_alloc.encode(
                    cid,
                    value=float(dist),
                    value_min=float(VIS_DIST_MIN),
                    value_max=float(VIS_DIST_MAX),
                    window=VIS_WINDOW,
                )
                bits[start:end] |= var_bits
            except ValueError:
                continue  # visible allocator exhausted

        # === Spatial map known concepts × world distance ===
        if spatial_map is not None:
            # Iterate over known concepts in spatial_map
            for concept in _spatial_map_concepts(spatial_map):
                nearest = spatial_map.find_nearest(concept, player_pos)
                if nearest is None:
                    continue
                dist = abs(nearest[0] - player_pos[0]) + abs(nearest[1] - player_pos[1])
                try:
                    start, end, var_bits = self.known_alloc.encode(
                        concept,
                        value=float(min(dist, KNOWN_DIST_MAX)),
                        value_min=float(KNOWN_DIST_MIN),
                        value_max=float(KNOWN_DIST_MAX),
                        window=KNOWN_WINDOW,
                    )
                    bits[start:end] |= var_bits
                except ValueError:
                    continue

        return bits

    def _is_presence_item(self, item: str) -> bool:
        """Heuristic: is this a binary presence item (tool/crafted) vs a scalar count?

        Not hardcoded to Crafter specifics — uses name suffix pattern. If this
        proves insufficient, caller can override via explicit presence set.
        """
        presence_suffixes = ("_sword", "_pickaxe")
        presence_exact = {"table", "furnace"}
        if item in presence_exact:
            return True
        for suffix in presence_suffixes:
            if item.endswith(suffix):
                return True
        return False


def _viewport_center() -> tuple[int, int]:
    """Center of 7×9 viewport grid (from Stage 75 tile_head_trainer)."""
    from snks.encoder.tile_head_trainer import VIEWPORT_ROWS, VIEWPORT_COLS
    return (VIEWPORT_ROWS // 2, VIEWPORT_COLS // 2)


def _spatial_map_concepts(spatial_map: Any) -> set[str]:
    """Extract the set of known concept types from a CrafterSpatialMap.

    CrafterSpatialMap stores (x, y) → concept_name in `_map` attribute.
    """
    if not hasattr(spatial_map, "_map"):
        return set()
    concepts = set()
    for concept_name in spatial_map._map.values():
        if concept_name and concept_name != "empty":
            concepts.add(concept_name)
    return concepts
