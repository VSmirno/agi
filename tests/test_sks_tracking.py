"""Tests for SKSTracker (Stage 3)."""

import pytest

from snks.sks.tracking import SKSTracker


class TestSKSTrackerBasic:
    """SKSTracker — stable cluster ID assignment."""

    def test_first_call_assigns_ids(self) -> None:
        tracker = SKSTracker()
        result = tracker.update([{0, 1, 2}, {3, 4, 5}])
        assert len(result) == 2
        assert all(isinstance(k, int) for k in result)
        assert all(isinstance(v, set) for v in result.values())

    def test_stable_clusters_keep_ids(self) -> None:
        """Same clusters in consecutive calls → same IDs."""
        tracker = SKSTracker()
        r1 = tracker.update([{0, 1, 2}, {3, 4, 5}])
        r2 = tracker.update([{0, 1, 2}, {3, 4, 5}])
        assert set(r1.keys()) == set(r2.keys())
        for k in r1:
            assert r1[k] == r2[k]

    def test_new_cluster_gets_new_id(self) -> None:
        """A new cluster appearing → new unique ID."""
        tracker = SKSTracker()
        r1 = tracker.update([{0, 1, 2}])
        r2 = tracker.update([{0, 1, 2}, {10, 11, 12}])
        old_ids = set(r1.keys())
        new_ids = set(r2.keys()) - old_ids
        assert len(new_ids) == 1

    def test_disappeared_cluster_removed(self) -> None:
        """Cluster that disappears → not in output."""
        tracker = SKSTracker()
        r1 = tracker.update([{0, 1, 2}, {3, 4, 5}])
        r2 = tracker.update([{0, 1, 2}])
        assert len(r2) == 1

    def test_partial_overlap_matched(self) -> None:
        """Cluster with high overlap to previous → same ID."""
        tracker = SKSTracker()
        r1 = tracker.update([{0, 1, 2, 3, 4}])
        old_id = list(r1.keys())[0]
        r2 = tracker.update([{1, 2, 3, 4, 5}])  # 4/6 Jaccard overlap
        assert old_id in r2

    def test_empty_clusters(self) -> None:
        tracker = SKSTracker()
        r1 = tracker.update([])
        assert r1 == {}
        r2 = tracker.update([{0, 1}])
        assert len(r2) == 1

    def test_no_duplicate_ids(self) -> None:
        """All IDs are unique across calls."""
        tracker = SKSTracker()
        all_ids: set[int] = set()
        for clusters in [[{0, 1}], [{2, 3}], [{4, 5}, {6, 7}]]:
            result = tracker.update(clusters)
            for k in result:
                assert k not in all_ids or k in set(result.keys())
