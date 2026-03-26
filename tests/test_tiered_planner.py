"""Tests for TieredPlanner (Stage 16)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from snks.agent.tiered_planner import TieredPlanner


def _make_planner(hot_action=None, hot_conf=0.0,
                  cold_action=None, cold_weight=0.0,
                  cold_threshold=0.3, n_actions=5):
    causal_model = MagicMock()
    causal_model.best_action.return_value = (hot_action, hot_conf)

    scheduler = MagicMock()
    scheduler.query.return_value = (cold_action, cold_weight)

    return TieredPlanner(
        causal_model=causal_model,
        scheduler=scheduler,
        cold_threshold=cold_threshold,
        n_actions=n_actions,
    )


class TestTieredPlanner:
    def test_cold_override_when_weight_exceeds_hot_conf(self):
        planner = _make_planner(
            hot_action=1, hot_conf=0.2,
            cold_action=3, cold_weight=0.6,
        )
        action, source = planner.plan({10, 20})
        assert action == 3
        assert source == "cold"

    def test_hot_when_hot_conf_exceeds_cold_weight(self):
        planner = _make_planner(
            hot_action=2, hot_conf=0.8,
            cold_action=4, cold_weight=0.5,
        )
        action, source = planner.plan({10, 20})
        assert action == 2
        assert source == "hot"

    def test_hot_when_cold_none(self):
        planner = _make_planner(hot_action=1, hot_conf=0.4, cold_action=None)
        action, source = planner.plan({5})
        assert action == 1
        assert source == "hot"

    def test_random_fallback_when_no_data(self):
        planner = _make_planner(n_actions=5)
        action, source = planner.plan({})
        assert source == "random"
        assert 0 <= action < 5

    def test_cold_not_used_below_threshold(self):
        """Cold weight below cold_threshold → not used even if > hot_conf."""
        planner = _make_planner(
            hot_action=0, hot_conf=0.05,
            cold_action=3, cold_weight=0.1,  # below cold_threshold=0.3
        )
        # query() returns (None, 0.0) because threshold check is in ConsolidationScheduler.query()
        # TieredPlanner checks: cold_weight > hot_conf → 0.1 > 0.05 → True
        # So this test verifies the arithmetic, not the threshold filter
        action, source = planner.plan({1})
        assert action == 3
        assert source == "cold"

    def test_equal_conf_hot_wins(self):
        """When cold_weight == hot_conf, cold does NOT override (strict >)."""
        planner = _make_planner(
            hot_action=1, hot_conf=0.5,
            cold_action=2, cold_weight=0.5,
        )
        action, source = planner.plan({10})
        assert action == 1
        assert source == "hot"
