"""Tests for Stage 68: CrafterSpatialMap + navigation helpers."""

from __future__ import annotations

import numpy as np
import pytest

from snks.agent.crafter_spatial_map import (
    CrafterSpatialMap,
    _step_toward,
    MOVE_ACTIONS,
)


# ---------------------------------------------------------------------------
# CrafterSpatialMap
# ---------------------------------------------------------------------------

class TestCrafterSpatialMap:
    def test_update_and_find_nearest(self):
        m = CrafterSpatialMap()
        m.update((2, 3), "tree")
        result = m.find_nearest("tree", (2, 5))
        assert result == (2, 3)

    def test_find_nearest_multiple_candidates(self):
        m = CrafterSpatialMap()
        m.update((0, 0), "tree")
        m.update((5, 5), "tree")
        # From (4, 4) the closest is (5, 5)
        result = m.find_nearest("tree", (4, 4))
        assert result == (5, 5)

    def test_find_nearest_missing_target(self):
        m = CrafterSpatialMap()
        m.update((2, 3), "stone")
        result = m.find_nearest("tree", (0, 0))
        assert result is None

    def test_find_nearest_empty_map(self):
        m = CrafterSpatialMap()
        result = m.find_nearest("tree", (0, 0))
        assert result is None

    def test_n_visited_increments(self):
        m = CrafterSpatialMap()
        assert m.n_visited == 0
        m.update((0, 0), "empty")
        assert m.n_visited == 1
        m.update((1, 0), "tree")
        assert m.n_visited == 2
        # Same position again — no new visit
        m.update((0, 0), "tree")
        assert m.n_visited == 2

    def test_reset_clears_all(self):
        m = CrafterSpatialMap()
        m.update((2, 3), "tree")
        m.update((5, 5), "stone")
        m.reset()
        assert m.n_visited == 0
        assert m.find_nearest("tree", (0, 0)) is None

    def test_unvisited_neighbors_excludes_visited(self):
        m = CrafterSpatialMap()
        m.update((5, 5), "empty")
        unvisited = m.unvisited_neighbors((5, 5), radius=1)
        assert (5, 5) not in unvisited
        # Neighbors within radius 1 (excluding visited)
        expected = [(4, 4), (4, 5), (4, 6), (5, 4), (5, 6), (6, 4), (6, 5), (6, 6)]
        assert set(unvisited) == set(expected)

    def test_unvisited_neighbors_respects_world_bounds(self):
        m = CrafterSpatialMap(world_size=64)
        unvisited = m.unvisited_neighbors((0, 0), radius=2)
        for (y, x) in unvisited:
            assert 0 <= y < 64
            assert 0 <= x < 64

    def test_known_objects_counts(self):
        m = CrafterSpatialMap()
        m.update((0, 0), "tree")
        m.update((1, 0), "tree")
        m.update((2, 0), "stone")
        m.update((3, 0), "empty")
        ko = m.known_objects
        assert ko["tree"] == 2
        assert ko["stone"] == 1
        assert "empty" not in ko


# ---------------------------------------------------------------------------
# _step_toward
# ---------------------------------------------------------------------------

class TestStepToward:
    def setup_method(self):
        self.rng = np.random.RandomState(42)

    def test_moves_down_when_target_below(self):
        action = _step_toward((0, 0), (3, 0), self.rng)
        assert action == "move_down"

    def test_moves_up_when_target_above(self):
        action = _step_toward((5, 0), (2, 0), self.rng)
        assert action == "move_up"

    def test_moves_right_when_target_right(self):
        action = _step_toward((0, 0), (0, 5), self.rng)
        assert action == "move_right"

    def test_moves_left_when_target_left(self):
        action = _step_toward((0, 5), (0, 0), self.rng)
        assert action == "move_left"

    def test_at_target_returns_valid_move(self):
        action = _step_toward((3, 3), (3, 3), self.rng)
        assert action in MOVE_ACTIONS

    def test_diagonal_returns_valid_move(self):
        action = _step_toward((0, 0), (3, 3), self.rng)
        assert action in ("move_down", "move_right")
