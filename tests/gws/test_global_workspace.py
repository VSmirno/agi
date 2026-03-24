"""Tests for GlobalWorkspace."""

import pytest
import torch

from snks.gws.workspace import GlobalWorkspace, GWSState
from snks.daf.types import GWSConfig


def make_gws(**kwargs) -> GlobalWorkspace:
    return GlobalWorkspace(GWSConfig(**kwargs))


class TestGlobalWorkspaceWinner:
    def test_winner_is_largest_cluster(self):
        gws = make_gws(w_size=1.0)
        clusters = {0: {1, 2, 3}, 1: {4, 5}, 2: {6}}
        result = gws.select_winner(clusters, fired_history=None)
        assert result is not None
        assert result.winner_id == 0
        assert result.winner_size == 3

    def test_dominance_formula(self):
        gws = make_gws()
        # cluster 0: 3 nodes, cluster 1: 2 nodes; union = 5 unique nodes
        clusters = {0: {1, 2, 3}, 1: {4, 5}}
        result = gws.select_winner(clusters, fired_history=None)
        assert result is not None
        # winner is cluster 0 (size 3), total unique = 5
        assert result.dominance == pytest.approx(3 / 5)

    def test_returns_none_when_no_clusters(self):
        gws = make_gws()
        result = gws.select_winner({}, fired_history=None)
        assert result is None

    def test_winner_score_respects_weights(self):
        # With w_size=1.0, cluster 0 wins (size 3 > 2)
        gws = make_gws(w_size=1.0)
        clusters = {0: {1, 2, 3}, 1: {4, 5}}
        result = gws.select_winner(clusters, fired_history=None)
        assert result.winner_id == 0
        assert result.winner_score == pytest.approx(3.0)

    def test_fired_history_none_with_clusters(self):
        """fired_history=None with w_coherence=0 should work normally."""
        gws = make_gws(w_coherence=0.0)
        clusters = {0: {1, 2, 3}, 1: {4}}
        result = gws.select_winner(clusters, fired_history=None)
        assert result is not None
        assert result.winner_id == 0

    def test_single_cluster(self):
        """Single cluster → dominance=1.0."""
        gws = make_gws()
        clusters = {42: {1, 2, 3, 4}}
        result = gws.select_winner(clusters, fired_history=None)
        assert result is not None
        assert result.winner_id == 42
        assert result.dominance == pytest.approx(1.0)

    def test_overlapping_clusters_total_active(self):
        """Overlapping clusters: total_active = len(union)."""
        gws = make_gws()
        # clusters share node 5
        clusters = {0: {1, 2, 5}, 1: {5, 6, 7, 8}}
        result = gws.select_winner(clusters, fired_history=None)
        assert result is not None
        # union = {1,2,5,6,7,8} = 6 unique nodes
        # winner is cluster 1 (size 4), dominance = 4/6
        assert result.winner_id == 1
        assert result.dominance == pytest.approx(4 / 6)
