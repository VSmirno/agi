"""Tests for Stage 12: IntrinsicCostModule."""

from __future__ import annotations

import math
import pytest
from dataclasses import dataclass

from snks.metacog.cost_module import IntrinsicCostModule
from snks.daf.types import CostModuleConfig, CostState


@dataclass
class _MS:
    winner_pe: float = 0.0
    meta_pe: float = 0.0


def _make_module(
    w_homeostatic: float = 0.3,
    w_epistemic: float = 0.4,
    w_goal: float = 0.3,
    firing_rate_target: float = 0.05,
) -> IntrinsicCostModule:
    cfg = CostModuleConfig(
        enabled=True,
        w_homeostatic=w_homeostatic,
        w_epistemic=w_epistemic,
        w_goal=w_goal,
        firing_rate_target=firing_rate_target,
    )
    return IntrinsicCostModule(cfg)


# ---------------------------------------------------------------------------
# Test 1: basic smoke test — compute() returns CostState
# ---------------------------------------------------------------------------

def test_compute_returns_cost_state():
    mod = _make_module()
    state = mod.compute(_MS(), mean_firing_rate=0.05)
    assert isinstance(state, CostState)
    assert hasattr(state, "total")
    assert hasattr(state, "homeostatic")
    assert hasattr(state, "epistemic_value")
    assert hasattr(state, "goal")


# ---------------------------------------------------------------------------
# Test 2: total ∈ [0, 1] for random inputs
# ---------------------------------------------------------------------------

def test_total_cost_in_range():
    import random
    random.seed(42)
    mod = _make_module()
    for _ in range(50):
        rate = random.uniform(0.0, 1.0)
        winner_pe = random.uniform(0.0, 1.0)
        meta_pe = random.uniform(0.0, 1.0)
        state = mod.compute(_MS(winner_pe=winner_pe, meta_pe=meta_pe), mean_firing_rate=rate)
        assert 0.0 <= state.total <= 1.0, (
            f"total={state.total} out of range for rate={rate}, "
            f"winner_pe={winner_pe}, meta_pe={meta_pe}"
        )


# ---------------------------------------------------------------------------
# Test 3: mean_firing_rate == target → homeostatic ≈ 0.0
# ---------------------------------------------------------------------------

def test_homeostatic_zero_at_target():
    target = 0.05
    mod = _make_module(firing_rate_target=target)
    state = mod.compute(_MS(), mean_firing_rate=target)
    assert state.homeostatic == pytest.approx(0.0, abs=1e-7)


# ---------------------------------------------------------------------------
# Test 4: mean_firing_rate = 10 * target → homeostatic ≈ 1.0
# ---------------------------------------------------------------------------

def test_homeostatic_high_when_far():
    target = 0.05
    mod = _make_module(firing_rate_target=target)
    # deviation = |10*target - target| / target = 9*target / target = 9 → clamped to 1.0
    state = mod.compute(_MS(), mean_firing_rate=10 * target)
    assert state.homeostatic == pytest.approx(1.0, abs=1e-7)


# ---------------------------------------------------------------------------
# Test 5: epistemic_value = max(winner_pe, meta_pe)
# ---------------------------------------------------------------------------

def test_epistemic_value_max_of_pe_signals():
    mod = _make_module()
    state = mod.compute(_MS(winner_pe=0.3, meta_pe=0.7), mean_firing_rate=0.05)
    assert state.epistemic_value == pytest.approx(0.7, abs=1e-7)


# ---------------------------------------------------------------------------
# Test 6: zero winner_pe, non-zero meta_pe → epistemic_value = meta_pe
# ---------------------------------------------------------------------------

def test_epistemic_value_ignores_zero_signals():
    mod = _make_module()
    state = mod.compute(_MS(winner_pe=0.0, meta_pe=0.5), mean_firing_rate=0.05)
    assert state.epistemic_value == pytest.approx(0.5, abs=1e-7)


# ---------------------------------------------------------------------------
# Test 7: both zero → epistemic_value = 0.0
# ---------------------------------------------------------------------------

def test_epistemic_value_zero_when_no_signals():
    mod = _make_module()
    state = mod.compute(_MS(winner_pe=0.0, meta_pe=0.0), mean_firing_rate=0.05)
    assert state.epistemic_value == pytest.approx(0.0, abs=1e-7)


# ---------------------------------------------------------------------------
# Test 8: set_goal_cost(1.0) increases total vs set_goal_cost(0.0)
# ---------------------------------------------------------------------------

def test_set_goal_cost_affects_total():
    ms = _MS(winner_pe=0.0, meta_pe=0.0)
    rate = 0.05  # at target → homeostatic=0

    mod_low = _make_module()
    mod_low.set_goal_cost(0.0)
    low = mod_low.compute(ms, mean_firing_rate=rate)

    mod_high = _make_module()
    mod_high.set_goal_cost(1.0)
    high = mod_high.compute(ms, mean_firing_rate=rate)

    assert high.total > low.total, (
        f"expected high goal cost to raise total: low={low.total}, high={high.total}"
    )


# ---------------------------------------------------------------------------
# Test 9: manually compute expected total and compare
# ---------------------------------------------------------------------------

def test_total_cost_formula():
    w_h, w_e, w_g = 0.3, 0.4, 0.3
    target = 0.05
    rate = 0.10          # homeostatic = |0.10 - 0.05| / 0.05 = 1.0 → clamped 1.0
    winner_pe = 0.0
    meta_pe = 0.6        # epistemic_value = 0.6
    goal_cost = 0.2

    mod = _make_module(w_homeostatic=w_h, w_epistemic=w_e, w_goal=w_g,
                       firing_rate_target=target)
    mod.set_goal_cost(goal_cost)
    state = mod.compute(_MS(winner_pe=winner_pe, meta_pe=meta_pe), mean_firing_rate=rate)

    homeostatic = min(abs(rate - target) / target, 1.0)  # = 1.0
    epistemic_value = 0.6
    comfort = 1.0 - homeostatic          # = 0.0
    curiosity = epistemic_value          # = 0.6
    goal_progress = 1.0 - goal_cost      # = 0.8
    weight_sum = w_h + w_e + w_g        # = 1.0
    value = (w_h * comfort + w_e * curiosity + w_g * goal_progress) / weight_sum
    value = max(0.0, min(1.0, value))
    expected_total = 1.0 - value

    assert state.total == pytest.approx(expected_total, abs=1e-6)
    assert state.homeostatic == pytest.approx(homeostatic, abs=1e-6)
    assert state.epistemic_value == pytest.approx(epistemic_value, abs=1e-6)
    assert state.goal == pytest.approx(goal_cost, abs=1e-6)
