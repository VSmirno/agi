"""Tests for CuriosityModule (Stage 29)."""

from __future__ import annotations

import pytest
from snks.language.curiosity_module import CuriosityModule


class TestCuriosityModule:

    def test_initial_reward_is_one(self):
        """First visit to any state gives r_int = 1.0."""
        cm = CuriosityModule()
        key = frozenset({50, 54, 10101})
        assert cm.intrinsic_reward(key) == pytest.approx(1.0)

    def test_observe_returns_reward_before_increment(self):
        """observe() returns reward THEN increments count."""
        cm = CuriosityModule()
        key = frozenset({50, 10101})
        r1 = cm.observe(key)
        assert r1 == pytest.approx(1.0)  # first visit
        r2 = cm.observe(key)
        assert r2 == pytest.approx(0.5)  # second visit: 1/(1+1)

    def test_repeated_visit_reward_decreases(self):
        """Reward decreases monotonically with visit count."""
        cm = CuriosityModule()
        key = frozenset({50, 10101})
        rewards = [cm.observe(key) for _ in range(5)]
        for i in range(len(rewards) - 1):
            assert rewards[i] > rewards[i + 1]

    def test_intrinsic_reward_peek_no_update(self):
        """intrinsic_reward() does not change count."""
        cm = CuriosityModule()
        key = frozenset({50})
        _ = cm.intrinsic_reward(key)
        _ = cm.intrinsic_reward(key)
        assert cm.count(key) == 0

    def test_n_distinct_increases(self):
        """n_distinct grows with new states."""
        cm = CuriosityModule()
        k1 = frozenset({50, 10101})
        k2 = frozenset({51, 10102})
        cm.observe(k1)
        assert cm.n_distinct() == 1
        cm.observe(k2)
        assert cm.n_distinct() == 2
        cm.observe(k1)  # revisit
        assert cm.n_distinct() == 2  # no change

    def test_make_key_includes_position(self):
        """Different positions produce different keys even with same SKS."""
        sks = {50, 54}
        k1 = CuriosityModule.make_key(sks, (1, 1))
        k2 = CuriosityModule.make_key(sks, (2, 2))
        assert k1 != k2

    def test_make_key_same_pos_same_key(self):
        """Same SKS + same position → same key."""
        sks = {50, 54}
        k1 = CuriosityModule.make_key(sks, (3, 3))
        k2 = CuriosityModule.make_key(sks, (3, 3))
        assert k1 == k2

    def test_reset_clears_counts(self):
        """reset() starts fresh."""
        cm = CuriosityModule()
        key = frozenset({50})
        cm.observe(key)
        cm.observe(key)
        cm.reset()
        assert cm.n_distinct() == 0
        assert cm.count(key) == 0
        assert cm.intrinsic_reward(key) == pytest.approx(1.0)

    def test_formula_exact(self):
        """r_int = 1.0 / (1 + n_visits)."""
        cm = CuriosityModule()
        key = frozenset({50})
        for n in range(5):
            assert cm.intrinsic_reward(key) == pytest.approx(1.0 / (1 + n))
            cm.observe(key)
