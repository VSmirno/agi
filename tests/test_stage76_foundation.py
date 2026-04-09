"""Stage 76 Phase 1: foundation tests.

Covers:
- HomeostaticTracker.observed_max / observed_variables (tracker extensions)
- bucket_encode (similarity-preserving scalar encoding)
- FixedSDRRegistry (deterministic per-concept patterns)
- SpatialRangeAllocator (per-concept bit range allocation)
"""

from __future__ import annotations

import numpy as np
import pytest

from snks.agent.perception import HomeostaticTracker
from snks.memory.sdr_encoder import (
    bucket_encode,
    FixedSDRRegistry,
    SpatialRangeAllocator,
)


# ---------------------------------------------------------------------------
# HomeostaticTracker extensions (task 1.1)
# ---------------------------------------------------------------------------


class TestTrackerExtensions:
    def test_observed_max_starts_empty(self):
        tracker = HomeostaticTracker()
        assert tracker.observed_max == {}
        assert tracker.observed_variables() == set()

    def test_observed_max_updates_from_inventory(self):
        tracker = HomeostaticTracker()
        tracker.update(
            inv_before={"health": 9, "food": 9},
            inv_after={"health": 8, "food": 9},
            visible_concepts=set(),
        )
        assert tracker.observed_max["health"] == 9
        assert tracker.observed_max["food"] == 9

    def test_observed_max_monotonic(self):
        tracker = HomeostaticTracker()
        # First observation
        tracker.update(
            inv_before={"wood": 0},
            inv_after={"wood": 1},
            visible_concepts=set(),
        )
        assert tracker.observed_max["wood"] == 1

        # Grow
        tracker.update(
            inv_before={"wood": 2},
            inv_after={"wood": 3},
            visible_concepts=set(),
        )
        assert tracker.observed_max["wood"] == 3

        # Drop below max — max stays at max
        tracker.update(
            inv_before={"wood": 3},
            inv_after={"wood": 1},
            visible_concepts=set(),
        )
        assert tracker.observed_max["wood"] == 3  # max does not decrease

    def test_observed_variables_dynamic(self):
        """Variable set grows from observation, no hardcoded list."""
        tracker = HomeostaticTracker()
        assert tracker.observed_variables() == set()

        tracker.update(
            inv_before={"health": 9},
            inv_after={"health": 9},
            visible_concepts=set(),
        )
        assert tracker.observed_variables() == {"health"}

        tracker.update(
            inv_before={"health": 9, "food": 5, "novel_var": 2},
            inv_after={"health": 9, "food": 4, "novel_var": 3},
            visible_concepts=set(),
        )
        assert tracker.observed_variables() == {"health", "food", "novel_var"}
        # Novel var also gets observed_max
        assert tracker.observed_max["novel_var"] == 3

    def test_observed_max_tracks_both_before_and_after(self):
        """If inv_before has a higher value than inv_after, both should be scanned."""
        tracker = HomeostaticTracker()
        tracker.update(
            inv_before={"stone": 5},
            inv_after={"stone": 3},
            visible_concepts=set(),
        )
        assert tracker.observed_max["stone"] == 5


# ---------------------------------------------------------------------------
# bucket_encode (task 1.3)
# ---------------------------------------------------------------------------


class TestBucketEncode:
    def test_output_shape(self):
        bits = bucket_encode(5, 0, 9, 0, 100, 40)
        assert bits.shape == (100,)
        assert bits.dtype == np.bool_

    def test_active_bit_count_matches_window(self):
        bits = bucket_encode(5, 0, 9, 0, 100, 40)
        assert bits.sum() == 40

    def test_consecutive_values_share_most_bits(self):
        """HP=5 and HP=6 should share ~80% bits (close values → similar patterns)."""
        b5 = bucket_encode(5, 0, 9, 0, 100, 40)
        b6 = bucket_encode(6, 0, 9, 0, 100, 40)
        overlap = np.logical_and(b5, b6).sum()
        # Window slides by ~max_start/9 = 60/9 ≈ 7 bits per unit
        # So HP=5 and HP=6 differ by ~7, overlap is 40-7 = 33
        assert overlap >= 30
        assert overlap <= 40

    def test_distant_values_share_zero_bits(self):
        """HP=0 and HP=9 should have no overlap."""
        b0 = bucket_encode(0, 0, 9, 0, 100, 40)
        b9 = bucket_encode(9, 0, 9, 0, 100, 40)
        overlap = np.logical_and(b0, b9).sum()
        assert overlap == 0

    def test_boundary_extremes_valid(self):
        b_min = bucket_encode(0, 0, 9, 0, 100, 40)
        b_max = bucket_encode(9, 0, 9, 0, 100, 40)
        assert b_min.sum() == 40
        assert b_max.sum() == 40
        # Extreme window sits at start of range for min
        assert b_min[0] == True
        # And at end of range for max
        assert b_max[-1] == True

    def test_clip_out_of_range(self):
        """Values outside [value_min, value_max] should be clipped."""
        b_low = bucket_encode(-100, 0, 9, 0, 100, 40)
        b_min = bucket_encode(0, 0, 9, 0, 100, 40)
        assert np.array_equal(b_low, b_min)

        b_high = bucket_encode(1000, 0, 9, 0, 100, 40)
        b_max = bucket_encode(9, 0, 9, 0, 100, 40)
        assert np.array_equal(b_high, b_max)

    def test_deterministic(self):
        """Same inputs produce same output."""
        b1 = bucket_encode(5, 0, 9, 0, 100, 40)
        b2 = bucket_encode(5, 0, 9, 0, 100, 40)
        assert np.array_equal(b1, b2)

    def test_window_larger_than_range_errors(self):
        with pytest.raises(ValueError):
            bucket_encode(5, 0, 9, 0, 30, 40)  # window > width


# ---------------------------------------------------------------------------
# FixedSDRRegistry (task 1.4)
# ---------------------------------------------------------------------------


class TestFixedSDRRegistry:
    def test_registration_is_lazy(self):
        reg = FixedSDRRegistry(total_bits=4096, bits_per_concept=40)
        assert reg.known_concepts() == set()

        _ = reg.get("tree")
        assert reg.known_concepts() == {"tree"}

    def test_pattern_shape_and_density(self):
        reg = FixedSDRRegistry(total_bits=4096, bits_per_concept=40)
        pattern = reg.get("zombie")
        assert pattern.shape == (4096,)
        assert pattern.dtype == np.bool_
        assert pattern.sum() == 40

    def test_same_concept_same_pattern(self):
        reg = FixedSDRRegistry(total_bits=4096)
        p1 = reg.get("wood")
        p2 = reg.get("wood")
        assert np.array_equal(p1, p2)

    def test_different_concepts_near_zero_overlap(self):
        """Random sparse patterns have negligible overlap."""
        reg = FixedSDRRegistry(total_bits=4096, bits_per_concept=40)
        p_tree = reg.get("tree")
        p_stone = reg.get("stone")
        overlap = np.logical_and(p_tree, p_stone).sum()
        # Expected overlap of 2 random 40-bit subsets in 4096 bits:
        # ~ 40 * 40 / 4096 = 0.39 → very low
        assert overlap <= 5  # conservative upper bound

    def test_deterministic_across_instances(self):
        """Same seed + concept id → same pattern even across registry instances."""
        reg1 = FixedSDRRegistry(total_bits=4096, seed=42)
        reg2 = FixedSDRRegistry(total_bits=4096, seed=42)
        p1 = reg1.get("tree")
        p2 = reg2.get("tree")
        assert np.array_equal(p1, p2)

    def test_insertion_order_independence(self):
        """Pattern for 'wood' should not depend on order of other insertions."""
        reg_a = FixedSDRRegistry(total_bits=4096, seed=42)
        _ = reg_a.get("stone")
        _ = reg_a.get("tree")
        p_wood_a = reg_a.get("wood")

        reg_b = FixedSDRRegistry(total_bits=4096, seed=42)
        _ = reg_b.get("tree")
        _ = reg_b.get("stone")
        p_wood_b = reg_b.get("wood")

        assert np.array_equal(p_wood_a, p_wood_b)


# ---------------------------------------------------------------------------
# SpatialRangeAllocator (task 1.5)
# ---------------------------------------------------------------------------


class TestSpatialRangeAllocator:
    def test_ranges_are_distinct(self):
        alloc = SpatialRangeAllocator(start_bit=0, end_bit=1000, bits_per_concept=100)
        r_zombie = alloc.get_range("zombie")
        r_tree = alloc.get_range("tree")
        assert r_zombie != r_tree
        # Non-overlapping
        assert r_zombie[1] <= r_tree[0] or r_tree[1] <= r_zombie[0]

    def test_range_size_matches_bits_per_concept(self):
        alloc = SpatialRangeAllocator(start_bit=0, end_bit=1000, bits_per_concept=100)
        start, end = alloc.get_range("water")
        assert end - start == 100

    def test_range_persistence(self):
        """Same concept returns same range across calls."""
        alloc = SpatialRangeAllocator(start_bit=0, end_bit=1000, bits_per_concept=100)
        r1 = alloc.get_range("coal")
        r2 = alloc.get_range("coal")
        assert r1 == r2

    def test_exhaustion_raises(self):
        """Allocating more concepts than fit should raise."""
        alloc = SpatialRangeAllocator(start_bit=0, end_bit=200, bits_per_concept=100)
        alloc.get_range("a")
        alloc.get_range("b")
        with pytest.raises(ValueError):
            alloc.get_range("c")  # no room

    def test_encode_similarity_within_concept(self):
        """dist=2 and dist=3 for same concept share most bits."""
        alloc = SpatialRangeAllocator(
            start_bit=0, end_bit=1000, bits_per_concept=100
        )
        _, _, bits_2 = alloc.encode("zombie", value=2, value_min=0, value_max=9, window=40)
        _, _, bits_3 = alloc.encode("zombie", value=3, value_min=0, value_max=9, window=40)
        overlap = np.logical_and(bits_2, bits_3).sum()
        assert overlap >= 30  # close values → high overlap

    def test_encode_no_overlap_across_concepts(self):
        """Same distance for different concepts → different bit positions."""
        alloc = SpatialRangeAllocator(
            start_bit=0, end_bit=1000, bits_per_concept=100
        )
        start_z, _, _ = alloc.encode("zombie", value=2, value_min=0, value_max=9, window=40)
        start_t, _, _ = alloc.encode("tree", value=2, value_min=0, value_max=9, window=40)
        # Different starting positions → disjoint ranges
        assert start_z != start_t
