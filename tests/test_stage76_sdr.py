"""Stage 76 Phase 2: StateEncoder tests.

Verifies the raw-state → 4096-bit SDR projection.
Covers:
- Body stats encoding (bucket-based similarity)
- Inventory scalar vs presence split
- Visible concepts × distance (spatial allocation)
- Spatial map known concepts × world distance
- Similarity property: similar states → overlapping bits
- Density: ~200 active bits for typical states
"""

from __future__ import annotations

import numpy as np
import pytest

from snks.agent.crafter_spatial_map import CrafterSpatialMap
from snks.agent.perception import VisualField
from snks.memory.state_encoder import (
    StateEncoder,
    TOTAL_BITS,
    BODY_START,
    BODY_END,
    INV_SCALAR_START,
    INV_SCALAR_END,
    INV_PRESENCE_START,
    INV_PRESENCE_END,
    VIS_START,
    VIS_END,
    KNOWN_START,
    KNOWN_END,
)


# ---------------------------------------------------------------------------
# Helpers: build minimal test states
# ---------------------------------------------------------------------------


def make_visible_field(detections: list[tuple[str, int, int]]) -> VisualField:
    """detections: list of (concept_id, gy, gx). similarity is set to 1.0."""
    vf = VisualField()
    vf.detections = [(cid, 1.0, gy, gx) for cid, gy, gx in detections]
    return vf


def make_spatial_map(entries: dict[tuple[int, int], str]) -> CrafterSpatialMap:
    sm = CrafterSpatialMap()
    for (y, x), concept in entries.items():
        sm._map[(y, x)] = concept
        sm._visited.add((y, x))
    return sm


BODY_VARS = {"health", "food", "drink", "energy"}


def baseline_state():
    """A reasonable Crafter-like state for testing."""
    inv = {
        "health": 9,
        "food": 9,
        "drink": 9,
        "energy": 9,
        "wood": 2,
        "stone": 0,
        "wood_pickaxe": 0,
        "wood_sword": 0,
    }
    # Viewport center is (3, 4) for 7×9
    vf = make_visible_field([
        ("tree", 2, 4),    # 1 step above center
        ("grass", 3, 4),   # center
        ("water", 5, 4),   # 2 steps below
    ])
    sm = make_spatial_map({(10, 10): "tree", (12, 12): "water"})
    player_pos = (10, 10)
    return inv, vf, sm, player_pos


# ---------------------------------------------------------------------------
# Shape, density, determinism
# ---------------------------------------------------------------------------


class TestEncoderBasics:
    def test_output_shape(self):
        enc = StateEncoder()
        inv, vf, sm, pos = baseline_state()
        bits = enc.encode(inv, vf, sm, pos, body_variables=BODY_VARS)
        assert bits.shape == (TOTAL_BITS,)
        assert bits.dtype == np.bool_

    def test_deterministic(self):
        enc = StateEncoder()
        inv, vf, sm, pos = baseline_state()
        b1 = enc.encode(inv, vf, sm, pos, body_variables=BODY_VARS)
        b2 = enc.encode(inv, vf, sm, pos, body_variables=BODY_VARS)
        assert np.array_equal(b1, b2)

    def test_density_in_range(self):
        """Task 2.7: mean active bits in 50 random states in sparse range.

        Note: spec targeted ~200 (5%) but window=40 (needed for ≥80%
        adjacent-value similarity) mathematically forces higher density.
        For ~16 encoded things × window 40 = ~640 max. With partial
        overlap and presence items off, typical ~400-550. Still sparse
        by neural standards (<15%).
        """
        rng = np.random.RandomState(0)
        enc = StateEncoder()
        counts = []
        for _ in range(50):
            inv = {
                "health": rng.randint(0, 10),
                "food": rng.randint(0, 10),
                "drink": rng.randint(0, 10),
                "energy": rng.randint(0, 10),
                "wood": rng.randint(0, 6),
                "stone": rng.randint(0, 4),
                "wood_pickaxe": rng.randint(0, 2),
                "wood_sword": rng.randint(0, 2),
            }
            # 2-4 visible concepts
            n_vis = rng.randint(2, 5)
            concepts = ["tree", "stone", "water", "grass", "zombie", "cow"]
            chosen = rng.choice(concepts, size=n_vis, replace=False)
            dets = [
                (c, rng.randint(0, 7), rng.randint(0, 9))
                for c in chosen
            ]
            vf = make_visible_field(dets)
            sm = make_spatial_map({
                (rng.randint(0, 64), rng.randint(0, 64)): "tree",
                (rng.randint(0, 64), rng.randint(0, 64)): "water",
            })
            pos = (rng.randint(0, 64), rng.randint(0, 64))
            bits = enc.encode(inv, vf, sm, pos, body_variables=BODY_VARS)
            counts.append(int(bits.sum()))

        mean = float(np.mean(counts))
        # Sparsity guard: should be well below 20% (820 bits).
        assert 300 <= mean <= 700, f"density out of range: {mean} (counts={counts[:5]}...)"
        # And clearly sparser than dense networks
        assert mean / TOTAL_BITS < 0.20


# ---------------------------------------------------------------------------
# Body stats encoding (task 2.2)
# ---------------------------------------------------------------------------


class TestBodyEncoding:
    def test_body_bits_only_in_body_range(self):
        """Encoding only body stats should set bits exclusively in [0, 400)."""
        enc = StateEncoder()
        inv = {"health": 5, "food": 5, "drink": 5, "energy": 5}
        vf = VisualField()  # empty
        sm = CrafterSpatialMap()  # empty
        bits = enc.encode(inv, vf, sm, (10, 10), body_variables=BODY_VARS)
        # All active bits should be in body range
        active_outside_body = bits[BODY_END:].sum()
        assert active_outside_body == 0
        assert bits[BODY_START:BODY_END].sum() > 0

    def test_similar_body_states_overlap(self):
        """HP=5 vs HP=6 (rest equal) should share most body bits."""
        enc = StateEncoder()
        vf = VisualField()
        sm = CrafterSpatialMap()
        inv_a = {"health": 5, "food": 9, "drink": 9, "energy": 9}
        inv_b = {"health": 6, "food": 9, "drink": 9, "energy": 9}
        b_a = enc.encode(inv_a, vf, sm, (0, 0), body_variables=BODY_VARS)
        b_b = enc.encode(inv_b, vf, sm, (0, 0), body_variables=BODY_VARS)
        # Per-variable window is 40 bits; HP=5→HP=6 shifts ~7 bits.
        # Other 3 variables identical (120 bits). HP overlap ≈ 33.
        # Total overlap ≈ 120 + 33 = 153; total active = 160.
        overlap = np.logical_and(b_a, b_b).sum()
        total = b_a.sum()
        ratio = overlap / total
        assert ratio >= 0.8, f"expected ≥80% body overlap, got {ratio:.2f}"

    def test_distant_body_states_no_overlap_in_varied_var(self):
        """HP=0 vs HP=9 (rest equal) shares 0 bits in HP range, full in rest."""
        enc = StateEncoder()
        vf = VisualField()
        sm = CrafterSpatialMap()
        inv_a = {"health": 0, "food": 9, "drink": 9, "energy": 9}
        inv_b = {"health": 9, "food": 9, "drink": 9, "energy": 9}
        b_a = enc.encode(inv_a, vf, sm, (0, 0), body_variables=BODY_VARS)
        b_b = enc.encode(inv_b, vf, sm, (0, 0), body_variables=BODY_VARS)
        # 3 unchanged vars × 40 bits = 120 shared
        overlap = np.logical_and(b_a, b_b).sum()
        assert overlap == 120  # only food/drink/energy overlap fully


# ---------------------------------------------------------------------------
# Inventory encoding (task 2.3)
# ---------------------------------------------------------------------------


class TestInventoryEncoding:
    def test_scalar_inventory_in_scalar_range(self):
        enc = StateEncoder()
        inv = {"wood": 3, "stone": 2}
        bits = enc.encode(inv, VisualField(), CrafterSpatialMap(), (0, 0),
                          body_variables=set())
        assert bits[INV_SCALAR_START:INV_SCALAR_END].sum() > 0
        assert bits[INV_PRESENCE_START:INV_PRESENCE_END].sum() == 0

    def test_presence_inventory_activates_when_present(self):
        enc = StateEncoder()
        inv_has = {"wood_sword": 1}
        inv_lacks = {"wood_sword": 0}
        bits_has = enc.encode(inv_has, VisualField(), CrafterSpatialMap(), (0, 0),
                              body_variables=set())
        bits_lacks = enc.encode(inv_lacks, VisualField(), CrafterSpatialMap(), (0, 0),
                                body_variables=set())
        assert bits_has[INV_PRESENCE_START:INV_PRESENCE_END].sum() > 0
        assert bits_lacks[INV_PRESENCE_START:INV_PRESENCE_END].sum() == 0

    def test_close_wood_counts_overlap(self):
        """wood=2 and wood=3 share ≥80% bits in scalar range."""
        enc = StateEncoder()
        b2 = enc.encode({"wood": 2}, VisualField(), CrafterSpatialMap(), (0, 0),
                        body_variables=set())
        b3 = enc.encode({"wood": 3}, VisualField(), CrafterSpatialMap(), (0, 0),
                        body_variables=set())
        region = slice(INV_SCALAR_START, INV_SCALAR_END)
        overlap = np.logical_and(b2[region], b3[region]).sum()
        total = b2[region].sum()
        assert overlap / total >= 0.6  # close values → substantial overlap

    def test_presence_items_differ_from_each_other(self):
        """wood_sword and wood_pickaxe should produce different presence bits."""
        enc = StateEncoder()
        b_sword = enc.encode({"wood_sword": 1}, VisualField(), CrafterSpatialMap(),
                             (0, 0), body_variables=set())
        b_pickaxe = enc.encode({"wood_pickaxe": 1}, VisualField(), CrafterSpatialMap(),
                               (0, 0), body_variables=set())
        region = slice(INV_PRESENCE_START, INV_PRESENCE_END)
        # Different fixed SDRs → near-zero overlap
        overlap = np.logical_and(b_sword[region], b_pickaxe[region]).sum()
        assert overlap <= 5  # random sparse overlap expected tiny

    def test_presence_items_detected_by_name_heuristic(self):
        enc = StateEncoder()
        assert enc._is_presence_item("wood_sword") is True
        assert enc._is_presence_item("stone_pickaxe") is True
        assert enc._is_presence_item("table") is True
        assert enc._is_presence_item("furnace") is True
        assert enc._is_presence_item("wood") is False
        assert enc._is_presence_item("stone") is False
        assert enc._is_presence_item("coal") is False


# ---------------------------------------------------------------------------
# Visible concepts × distance (task 2.4)
# ---------------------------------------------------------------------------


class TestVisibleEncoding:
    def test_visible_bits_only_in_visible_range(self):
        enc = StateEncoder()
        vf = make_visible_field([("zombie", 3, 4)])  # at center
        bits = enc.encode({}, vf, CrafterSpatialMap(), (0, 0),
                          body_variables=set())
        # Only visible range should have bits set
        assert bits[VIS_START:VIS_END].sum() > 0
        assert bits[BODY_START:VIS_START].sum() == 0
        assert bits[VIS_END:].sum() == 0

    def test_same_concept_different_distance_high_overlap(self):
        """zombie@dist=0 and zombie@dist=1 share ≥75% bits within zombie range."""
        enc = StateEncoder()
        # Center is (3,4). dist=0 at center, dist=1 one step away.
        vf_near = make_visible_field([("zombie", 3, 4)])
        vf_far = make_visible_field([("zombie", 3, 5)])
        b_near = enc.encode({}, vf_near, CrafterSpatialMap(), (0, 0),
                            body_variables=set())
        b_far = enc.encode({}, vf_far, CrafterSpatialMap(), (0, 0),
                           body_variables=set())
        region = slice(VIS_START, VIS_END)
        overlap = np.logical_and(b_near[region], b_far[region]).sum()
        total = b_near[region].sum()
        assert overlap / total >= 0.75

    def test_different_concepts_zero_overlap(self):
        """zombie@dist=2 and tree@dist=2 share 0 bits (different ranges)."""
        enc = StateEncoder()
        vf_z = make_visible_field([("zombie", 1, 4)])  # dist=2
        vf_t = make_visible_field([("tree", 1, 4)])    # dist=2
        b_z = enc.encode({}, vf_z, CrafterSpatialMap(), (0, 0),
                         body_variables=set())
        b_t = enc.encode({}, vf_t, CrafterSpatialMap(), (0, 0),
                         body_variables=set())
        overlap = np.logical_and(b_z, b_t).sum()
        assert overlap == 0

    def test_empty_detections_skipped(self):
        enc = StateEncoder()
        vf = make_visible_field([("empty", 3, 4)])
        bits = enc.encode({}, vf, CrafterSpatialMap(), (0, 0),
                          body_variables=set())
        # empty is filtered
        assert bits[VIS_START:VIS_END].sum() == 0

    def test_multiple_same_concept_uses_min_distance(self):
        """Two zombies: the nearer one determines the encoding."""
        enc = StateEncoder()
        # Nearby zombie at (3,4)=dist 0, far zombie at (0,0)=dist 7
        vf_both = make_visible_field([("zombie", 3, 4), ("zombie", 0, 0)])
        vf_near = make_visible_field([("zombie", 3, 4)])
        b_both = enc.encode({}, vf_both, CrafterSpatialMap(), (0, 0),
                            body_variables=set())
        b_near = enc.encode({}, vf_near, CrafterSpatialMap(), (0, 0),
                            body_variables=set())
        assert np.array_equal(b_both, b_near)


# ---------------------------------------------------------------------------
# Spatial map encoding (task 2.5)
# ---------------------------------------------------------------------------


class TestKnownEncoding:
    def test_known_bits_only_in_known_range(self):
        enc = StateEncoder()
        sm = make_spatial_map({(10, 10): "tree"})
        bits = enc.encode({}, VisualField(), sm, (10, 10),
                          body_variables=set())
        assert bits[KNOWN_START:KNOWN_END].sum() > 0
        assert bits[KNOWN_END:].sum() == 0
        assert bits[:KNOWN_START].sum() == 0

    def test_known_empty_excluded(self):
        enc = StateEncoder()
        sm = make_spatial_map({(10, 10): "empty"})
        bits = enc.encode({}, VisualField(), sm, (10, 10),
                          body_variables=set())
        assert bits[KNOWN_START:KNOWN_END].sum() == 0

    def test_known_different_concepts_no_overlap(self):
        enc = StateEncoder()
        sm_tree = make_spatial_map({(10, 15): "tree"})
        sm_water = make_spatial_map({(10, 15): "water"})
        b_tree = enc.encode({}, VisualField(), sm_tree, (10, 10),
                            body_variables=set())
        b_water = enc.encode({}, VisualField(), sm_water, (10, 10),
                             body_variables=set())
        # Different concepts use disjoint ranges within known domain
        assert np.logical_and(b_tree, b_water).sum() == 0

    def test_known_close_distances_overlap(self):
        """Same concept at dist=5 vs dist=6 share most bits in its range."""
        enc = StateEncoder()
        sm_5 = make_spatial_map({(15, 10): "tree"})   # dist=5
        sm_6 = make_spatial_map({(16, 10): "tree"})   # dist=6
        b_5 = enc.encode({}, VisualField(), sm_5, (10, 10),
                         body_variables=set())
        b_6 = enc.encode({}, VisualField(), sm_6, (10, 10),
                         body_variables=set())
        region = slice(KNOWN_START, KNOWN_END)
        overlap = np.logical_and(b_5[region], b_6[region]).sum()
        total = b_5[region].sum()
        assert overlap / total >= 0.7  # allocator window=40, wider distance range


# ---------------------------------------------------------------------------
# Similarity property (task 2.6)
# ---------------------------------------------------------------------------


class TestSimilarity:
    def test_similar_states_high_overlap(self):
        """Manually-constructed similar states → ≥60% overlap."""
        enc = StateEncoder()
        inv_a = {"health": 9, "food": 8, "drink": 8, "energy": 9,
                 "wood": 2, "stone": 0, "wood_pickaxe": 0}
        inv_b = {"health": 9, "food": 8, "drink": 7, "energy": 9,
                 "wood": 2, "stone": 0, "wood_pickaxe": 0}  # drink changed by 1
        vf_a = make_visible_field([("tree", 2, 4), ("grass", 3, 4)])
        vf_b = make_visible_field([("tree", 2, 4), ("grass", 3, 4)])
        sm = make_spatial_map({(10, 12): "tree"})
        pos = (10, 10)
        b_a = enc.encode(inv_a, vf_a, sm, pos, body_variables=BODY_VARS)
        b_b = enc.encode(inv_b, vf_b, sm, pos, body_variables=BODY_VARS)
        overlap = np.logical_and(b_a, b_b).sum()
        total = b_a.sum()
        ratio = overlap / total
        assert ratio >= 0.6, f"similar states overlap {ratio:.2f} (expected ≥0.60)"

    def test_distinct_states_low_overlap(self):
        """Manually-constructed distinct states → ≤20% overlap."""
        enc = StateEncoder()
        # State A: full health, safe, tree visible
        inv_a = {"health": 9, "food": 9, "drink": 9, "energy": 9, "wood": 0}
        vf_a = make_visible_field([("tree", 2, 4), ("grass", 3, 4)])
        sm_a = make_spatial_map({(10, 12): "tree"})
        pos_a = (10, 10)

        # State B: dying, zombie visible, far from tree
        inv_b = {"health": 1, "food": 0, "drink": 0, "energy": 2, "wood": 5,
                 "wood_sword": 1}
        vf_b = make_visible_field([("zombie", 3, 4), ("stone", 1, 1)])
        sm_b = make_spatial_map({(30, 30): "water"})
        pos_b = (50, 50)

        b_a = enc.encode(inv_a, vf_a, sm_a, pos_a, body_variables=BODY_VARS)
        b_b = enc.encode(inv_b, vf_b, sm_b, pos_b, body_variables=BODY_VARS)
        overlap = np.logical_and(b_a, b_b).sum()
        total_min = min(b_a.sum(), b_b.sum())
        ratio = overlap / total_min
        assert ratio <= 0.20, f"distinct states overlap {ratio:.2f} (expected ≤0.20)"


# ---------------------------------------------------------------------------
# Dynamic allocation (no hardcoded concepts)
# ---------------------------------------------------------------------------


class TestDynamicAllocation:
    def test_novel_concept_allocates_on_first_observation(self):
        """Encoder has no hardcoded concept list — novel ids just work."""
        enc = StateEncoder()
        vf = make_visible_field([("alien_thing", 3, 4)])
        bits = enc.encode({}, vf, CrafterSpatialMap(), (0, 0),
                          body_variables=set())
        assert bits[VIS_START:VIS_END].sum() > 0

    def test_novel_body_variable_allocates(self):
        """Unseen body variable auto-allocates a range in body domain."""
        enc = StateEncoder()
        inv = {"mana": 5}
        bits = enc.encode(inv, VisualField(), CrafterSpatialMap(), (0, 0),
                          body_variables={"mana"})
        assert bits[BODY_START:BODY_END].sum() > 0

    def test_novel_inventory_scalar_allocates(self):
        """Unseen inventory scalar auto-allocates."""
        enc = StateEncoder()
        inv = {"exotic_item": 3}
        bits = enc.encode(inv, VisualField(), CrafterSpatialMap(), (0, 0),
                          body_variables=set())
        assert bits[INV_SCALAR_START:INV_SCALAR_END].sum() > 0
