"""Stage 77a Commit 3: Tests for HomeostaticTracker innate/observed split.

Verifies the new Bayesian combination of innate + observed rates, the new
init_from_textbook path, and that legacy behavior (init_from_body_rules,
get_rate with visible_concepts) still works during the staged transition.
"""

from __future__ import annotations

import pytest

from snks.agent.concept_store import ConceptStore
from snks.agent.crafter_textbook import CrafterTextbook
from snks.agent.perception import HomeostaticTracker


# ---------------------------------------------------------------------------
# Legacy init_from_body_rules backward compat
# ---------------------------------------------------------------------------


class TestLegacyInit:
    def test_legacy_init_populates_both_paths(self):
        t = HomeostaticTracker()
        t.init_from_body_rules([
            {"concept": "_background", "variable": "food", "rate": -0.04},
            {"concept": "zombie", "variable": "health", "rate": -2.0},
        ])
        # Legacy fields populated
        assert t.rates["food"] == -0.04
        assert t.conditional_rates[("zombie", "health")] == -2.0
        # New path — innate_rates gets the _background rules
        assert t.innate_rates["food"] == -0.04
        # Conditional rates NOT in innate_rates (different mechanism)
        assert "zombie" not in t.innate_rates

    def test_legacy_init_idempotent(self):
        """Second call should be skipped (as before)."""
        t = HomeostaticTracker()
        t.init_from_body_rules([
            {"concept": "_background", "variable": "food", "rate": -0.04},
        ])
        # Change the values then re-init
        t.rates["food"] = -99.0
        t.init_from_body_rules([
            {"concept": "_background", "variable": "food", "rate": -0.01},
        ])
        # Should NOT have overwritten because _initialized is True
        assert t.rates["food"] == -99.0


# ---------------------------------------------------------------------------
# New init_from_textbook
# ---------------------------------------------------------------------------


class TestInitFromTextbook:
    def _load_store_and_block(self):
        tb = CrafterTextbook("configs/crafter_textbook.yaml")
        store = ConceptStore()
        tb.load_into(store)
        return tb.body_block, store.passive_rules

    def test_populates_prior_strength(self):
        body, rules = self._load_store_and_block()
        t = HomeostaticTracker()
        t.init_from_textbook(body, rules)
        assert t.prior_strength == 20

    def test_populates_reference_bounds(self):
        body, rules = self._load_store_and_block()
        t = HomeostaticTracker()
        t.init_from_textbook(body, rules)
        assert t.reference_min == {
            "health": 0.0, "food": 0.0, "drink": 0.0, "energy": 0.0,
        }
        assert t.reference_max == {
            "health": 9.0, "food": 9.0, "drink": 9.0, "energy": 9.0,
        }

    def test_populates_innate_rates(self):
        body, rules = self._load_store_and_block()
        t = HomeostaticTracker()
        t.init_from_textbook(body, rules)
        # Rates updated to match real Crafter decay (measured via diag)
        assert t.innate_rates["food"] == -0.04
        assert t.innate_rates["drink"] == -0.05
        assert t.innate_rates["energy"] == -0.033

    def test_populates_observed_max_initial(self):
        body, rules = self._load_store_and_block()
        t = HomeostaticTracker()
        t.init_from_textbook(body, rules)
        assert t.observed_max["health"] == 9
        assert t.observed_max["food"] == 9

    def test_idempotent(self):
        body, rules = self._load_store_and_block()
        t = HomeostaticTracker()
        t.init_from_textbook(body, rules)
        t.innate_rates["food"] = -99.0
        t.init_from_textbook(body, rules)
        assert t.innate_rates["food"] == -99.0  # not overwritten


# ---------------------------------------------------------------------------
# Bayesian combination via get_rate(var)
# ---------------------------------------------------------------------------


class TestBayesianGetRate:
    def test_pure_innate_with_zero_observations(self):
        t = HomeostaticTracker()
        t.innate_rates["food"] = -0.04
        t.prior_strength = 20
        # n=0 → w=1.0 → rate = innate
        assert t.get_rate("food") == pytest.approx(-0.04)

    def test_half_weight_at_prior_strength_observations(self):
        t = HomeostaticTracker()
        t.innate_rates["food"] = -0.04
        t.prior_strength = 20
        # Simulate 20 observations of -1.0 each
        t.observed_rates["food"] = -1.0
        t.observation_counts["food"] = 20
        # w = 20/40 = 0.5 → rate = 0.5 * -0.04 + 0.5 * -1.0 = -0.52
        assert t.get_rate("food") == pytest.approx(-0.52)

    def test_dominated_by_observed_with_many_observations(self):
        t = HomeostaticTracker()
        t.innate_rates["food"] = -0.04
        t.prior_strength = 20
        # n=200 → w = 20/220 ≈ 0.091
        t.observed_rates["food"] = -1.0
        t.observation_counts["food"] = 200
        expected = (20 / 220) * -0.04 + (200 / 220) * -1.0
        assert t.get_rate("food") == pytest.approx(expected)

    def test_no_innate_no_observations_falls_back_to_legacy_rates(self):
        """If neither innate nor observed populated, fall back to legacy rates."""
        t = HomeostaticTracker()
        t.rates["food"] = -99.0
        # No innate, no observations
        assert t.get_rate("food") == -99.0

    def test_missing_var_returns_zero(self):
        t = HomeostaticTracker()
        assert t.get_rate("nonexistent") == 0.0


class TestLegacyGetRateStillWorks:
    """Legacy path with visible_concepts — used by select_goal (dies in Commit 8)."""

    def test_visible_concepts_returns_worst_conditional(self):
        t = HomeostaticTracker()
        t.rates["health"] = 0.0
        t.conditional_rates[("zombie", "health")] = -2.0
        t.conditional_rates[("skeleton", "health")] = -1.0
        # Worst (most negative) among visible
        assert t.get_rate("health", visible_concepts={"zombie", "skeleton"}) == -2.0

    def test_empty_visible_set_uses_rates(self):
        t = HomeostaticTracker()
        t.rates["food"] = -0.04
        # Empty visible set (falsy) → falls through to new path;
        # because legacy rates populated, the new path uses fallback
        assert t.get_rate("food", visible_concepts=set()) == -0.04

    def test_none_visible_uses_new_bayesian(self):
        """None signals 'new path', NOT 'no context' — these differ."""
        t = HomeostaticTracker()
        t.innate_rates["food"] = -0.04
        # visible_concepts=None → new path (Bayesian)
        assert t.get_rate("food", visible_concepts=None) == pytest.approx(-0.04)


# ---------------------------------------------------------------------------
# Running mean behavior
# ---------------------------------------------------------------------------


class TestRunningMean:
    def test_update_accumulates_running_mean(self):
        t = HomeostaticTracker()
        inv_before = {"food": 9}
        inv_after = {"food": 8}
        t.update(inv_before, inv_after, set())
        assert t.observed_rates["food"] == -1.0
        assert t.observation_counts["food"] == 1

    def test_running_mean_stable(self):
        """Running mean converges to true rate over many observations."""
        t = HomeostaticTracker()
        for _ in range(100):
            t.update({"food": 9}, {"food": 8}, set())  # delta = -1 each time
        assert t.observed_rates["food"] == pytest.approx(-1.0)
        assert t.observation_counts["food"] == 100

    def test_running_mean_bias_free(self):
        """Unlike EMA, running mean is exact average."""
        t = HomeostaticTracker()
        deltas = [-1, -2, -3, -4, -5]  # mean = -3
        for d in deltas:
            t.update({"food": 10}, {"food": 10 + d}, set())
        assert t.observed_rates["food"] == pytest.approx(-3.0)


# ---------------------------------------------------------------------------
# observed_variables now includes innate_rates
# ---------------------------------------------------------------------------


class TestObservedVariables:
    def test_includes_observed_max(self):
        t = HomeostaticTracker()
        t.observed_max["food"] = 9
        assert "food" in t.observed_variables()

    def test_includes_innate_before_observation(self):
        """New: agent knows about vars the teacher taught even before seeing them."""
        t = HomeostaticTracker()
        t.innate_rates["food"] = -0.04
        assert "food" in t.observed_variables()


# ---------------------------------------------------------------------------
# Full integration: textbook → tracker → forward-sim-style query
# ---------------------------------------------------------------------------


class TestTextbookIntegration:
    def test_full_pipeline(self):
        tb = CrafterTextbook("configs/crafter_textbook.yaml")
        store = ConceptStore()
        tb.load_into(store)

        t = HomeostaticTracker()
        t.init_from_textbook(tb.body_block, store.passive_rules)

        # reference_min used by SimState.is_dead check
        assert t.reference_min["health"] == 0.0

        # Innate food decay
        assert t.get_rate("food") == pytest.approx(-0.04)

        # After observing steady decay, running mean matches innate
        for _ in range(200):
            t.update({"food": 9}, {"food": 9}, set())  # no change
        # observed = 0 (no delta), innate = -0.04
        # w = 20/220 ≈ 0.091; effective = 0.091 * -0.04 + 0.909 * 0 ≈ -0.0036
        result = t.get_rate("food")
        assert -0.01 < result < 0  # mostly observed (near zero) with some innate pull
