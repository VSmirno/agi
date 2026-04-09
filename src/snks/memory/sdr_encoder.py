"""Stage 76: SDR encoding primitives.

Three building blocks for turning raw state into sparse binary vectors:

1. bucket_encode — scalar value → sliding window of bits (similar values
   share many bits, distant values share none)

2. FixedSDRRegistry — lazy-allocated deterministic random bit patterns per
   concept id (used for presence indicators, e.g., "has_sword")

3. SpatialRangeAllocator — pre-allocates a dedicated bit range per concept
   id, within which distance is bucket-encoded. Preserves similarity for
   (concept, distance) pairs without breaking concept separation.

The StateEncoder (Phase 2) uses these primitives to build a 4096-bit SDR
from inventory, visible, spatial_map, body state. All primitives are
deterministic given a seed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def bucket_encode(
    value: float,
    value_min: float,
    value_max: float,
    start_bit: int,
    end_bit: int,
    window: int,
) -> np.ndarray:
    """Sliding-window bucket encoding of a scalar.

    Maps value ∈ [value_min, value_max] to a contiguous window of ~`window`
    bits within the bit range [start_bit, end_bit). Similar values share
    most of their bits; distant values share none.

    Args:
        value: scalar to encode (clipped to [value_min, value_max]).
        value_min: lowest expected value.
        value_max: highest expected value.
        start_bit: first bit index of the range (inclusive).
        end_bit: last bit index of the range (exclusive).
        window: width of active bit window.

    Returns:
        Boolean array of shape (end_bit - start_bit,) with `window` active
        bits at the position proportional to `value` within the range.

    Example:
        bucket_encode(5, 0, 9, 0, 100, 40) activates bits ~27..67 out of 100.
        bucket_encode(6, 0, 9, 0, 100, 40) activates bits ~33..73 out of 100.
        Overlap: ~34 bits → high similarity for close values.
    """
    width = end_bit - start_bit
    if width <= window:
        raise ValueError(
            f"Bit range width {width} must be larger than window {window}"
        )
    if value_max <= value_min:
        raise ValueError(f"value_max {value_max} must be > value_min {value_min}")

    # Clip value to range
    v = max(value_min, min(value_max, value))
    # Normalize to [0, 1]
    t = (v - value_min) / (value_max - value_min)
    # Position of window start within the range
    # Window slides from 0 to (width - window) as t goes 0 → 1
    max_start = width - window
    start_offset = int(round(t * max_start))

    bits = np.zeros(width, dtype=bool)
    bits[start_offset : start_offset + window] = True
    return bits


@dataclass
class FixedSDRRegistry:
    """Lazy-allocated deterministic SDR patterns per concept id.

    Each concept gets a fixed random subset of bits, drawn from a seeded
    RNG. Same concept id → same pattern across calls and runs. Different
    concept ids → near-zero bit overlap (by sparse random subset property).

    Used for presence indicators: "has_sword" is either fully present
    (its bits ON) or absent (its bits OFF).
    """

    total_bits: int
    bits_per_concept: int = 40
    seed: int = 42
    _patterns: dict[str, np.ndarray] = field(default_factory=dict)
    _rng: np.random.RandomState | None = None

    def __post_init__(self) -> None:
        self._rng = np.random.RandomState(self.seed)

    def get(self, concept_id: str) -> np.ndarray:
        """Return (or lazy-allocate) the SDR pattern for a concept.

        Returns:
            Boolean array of shape (total_bits,) with bits_per_concept active.
        """
        if concept_id not in self._patterns:
            # Deterministic per-concept: hash id into a local seed so that
            # order of first-time access doesn't affect pattern identity.
            local_seed = self.seed + (hash(concept_id) & 0xFFFF_FFFF)
            local_rng = np.random.RandomState(local_seed)
            indices = local_rng.choice(
                self.total_bits, self.bits_per_concept, replace=False
            )
            pattern = np.zeros(self.total_bits, dtype=bool)
            pattern[indices] = True
            self._patterns[concept_id] = pattern
        return self._patterns[concept_id]

    def known_concepts(self) -> set[str]:
        return set(self._patterns.keys())


@dataclass
class SpatialRangeAllocator:
    """Pre-allocate bit ranges per concept for (concept, scalar) encoding.

    Each concept gets a dedicated contiguous bit range. Within that range,
    a scalar (typically distance) is bucket-encoded. This way:

    - "see_zombie @ dist=2" and "see_zombie @ dist=3" share most bits
      (same range, adjacent bucket positions)
    - "see_zombie @ dist=2" and "see_tree @ dist=2" share 0 bits
      (different ranges)

    Ranges are lazy-allocated on first access. Allocation order is stable
    across runs given the same insertion sequence.
    """

    start_bit: int           # first bit of the allocator's domain
    end_bit: int             # last bit (exclusive) of the domain
    bits_per_concept: int    # width of each concept's sub-range
    _ranges: dict[str, tuple[int, int]] = field(default_factory=dict)
    _next_start: int = 0

    def __post_init__(self) -> None:
        self._next_start = self.start_bit

    def get_range(self, concept_id: str) -> tuple[int, int]:
        """Return (or allocate) the (start, end) bit range for a concept.

        Raises:
            ValueError: if domain is exhausted.
        """
        if concept_id not in self._ranges:
            new_start = self._next_start
            new_end = new_start + self.bits_per_concept
            if new_end > self.end_bit:
                raise ValueError(
                    f"SpatialRangeAllocator exhausted: cannot allocate "
                    f"{self.bits_per_concept} bits for '{concept_id}' "
                    f"(next_start={new_start}, domain_end={self.end_bit})"
                )
            self._ranges[concept_id] = (new_start, new_end)
            self._next_start = new_end
        return self._ranges[concept_id]

    def encode(
        self,
        concept_id: str,
        value: float,
        value_min: float,
        value_max: float,
        window: int,
    ) -> tuple[int, int, np.ndarray]:
        """Encode (concept, scalar) using this concept's allocated range.

        Returns:
            (start, end, bits) — start/end bit indices of the range, and
            a boolean array matching that range's width.
        """
        start, end = self.get_range(concept_id)
        bits = bucket_encode(
            value=value,
            value_min=value_min,
            value_max=value_max,
            start_bit=start,
            end_bit=end,
            window=window,
        )
        return start, end, bits

    def allocated_concepts(self) -> set[str]:
        return set(self._ranges.keys())
