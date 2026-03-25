"""Tests for Stage 13: Configurator FSM."""

from __future__ import annotations

import pytest
from dataclasses import dataclass, field

from snks.metacog.configurator import Configurator, ConfiguratorMode
from snks.daf.types import (
    ConfiguratorConfig,
    ConfiguratorAction,
    CostState,
    DafConfig,
    HACPredictionConfig,
    MetacogConfig,
)


# ---------------------------------------------------------------------------
# Mock MetacogState
# ---------------------------------------------------------------------------

@dataclass
class _MS:
    stability: float = 0.5
    cost: CostState | None = None


def _explore_state() -> _MS:
    """MetacogState that satisfies EXPLORE transition conditions."""
    return _MS(
        stability=0.3,
        cost=CostState(total=0.7, homeostatic=0.3, epistemic_value=0.5, goal=0.0),
    )


def _consolidate_state() -> _MS:
    """MetacogState that satisfies CONSOLIDATE transition conditions."""
    return _MS(
        stability=0.9,
        cost=CostState(total=0.2, homeostatic=0.1, epistemic_value=0.1, goal=0.0),
    )


def _neutral_state() -> _MS:
    """MetacogState that falls through to NEUTRAL candidate."""
    return _MS(
        stability=0.5,
        cost=CostState(total=0.5, homeostatic=0.1, epistemic_value=0.2, goal=0.0),
    )


def _goal_state() -> _MS:
    """MetacogState that triggers GOAL_SEEKING (goal > threshold)."""
    return _MS(
        stability=0.3,
        # goal=0.5 > threshold=0.10; explore conditions also satisfied but GOAL wins
        cost=CostState(total=0.7, homeostatic=0.3, epistemic_value=0.5, goal=0.5),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

K = 3   # hysteresis_cycles for tests
MAX_EX = 6  # max_explore_cycles for tests


def _make_cfg(**kwargs) -> ConfiguratorConfig:
    defaults = dict(
        enabled=True,
        hysteresis_cycles=K,
        max_explore_cycles=MAX_EX,
        explore_cost_threshold=0.65,
        explore_epistemic_threshold=0.45,
        consolidate_cost_threshold=0.35,
        consolidate_stability_threshold=0.70,
        goal_cost_threshold=0.10,
    )
    defaults.update(kwargs)
    return ConfiguratorConfig(**defaults)


def _make_configurator(**cfg_kwargs):
    daf = DafConfig(stdp_a_plus=0.01, stdp_a_minus=0.012)
    metacog = MetacogConfig(alpha=1/3, beta=1/3, gamma=1/3, delta=0.0)
    hac_pred = HACPredictionConfig(memory_decay=0.95)
    cfg = _make_cfg(**cfg_kwargs)
    return Configurator(cfg, daf, metacog, hac_pred), daf, metacog, hac_pred


def _push_mode(configurator: Configurator, state_fn, n: int):
    """Call configurator.update(state_fn()) n times."""
    for _ in range(n):
        configurator.update(state_fn())


# ---------------------------------------------------------------------------
# Test 1: freshly created Configurator starts in NEUTRAL
# ---------------------------------------------------------------------------

def test_initial_mode_neutral():
    conf, *_ = _make_configurator()
    assert conf._current_mode == ConfiguratorMode.NEUTRAL


# ---------------------------------------------------------------------------
# Test 2: K-1 explore cycles → still NEUTRAL (hysteresis not yet satisfied)
# ---------------------------------------------------------------------------

def test_hysteresis_prevents_immediate_switch():
    conf, *_ = _make_configurator()
    _push_mode(conf, _explore_state, K - 1)
    assert conf._current_mode == ConfiguratorMode.NEUTRAL


# ---------------------------------------------------------------------------
# Test 3: exactly K explore cycles → switches to EXPLORE
# ---------------------------------------------------------------------------

def test_switch_to_explore_after_hysteresis():
    conf, *_ = _make_configurator()
    _push_mode(conf, _explore_state, K)
    assert conf._current_mode == ConfiguratorMode.EXPLORE


# ---------------------------------------------------------------------------
# Test 4: after EXPLORE switch, stdp_a_plus is increased
# ---------------------------------------------------------------------------

def test_explore_increases_plasticity():
    conf, daf, *_ = _make_configurator()
    original = daf.stdp_a_plus
    _push_mode(conf, _explore_state, K)
    assert conf._current_mode == ConfiguratorMode.EXPLORE
    assert daf.stdp_a_plus > original, (
        f"expected stdp_a_plus > {original}, got {daf.stdp_a_plus}"
    )


# ---------------------------------------------------------------------------
# Test 5: K consolidate cycles from NEUTRAL → stdp_a_plus decreases
# ---------------------------------------------------------------------------

def test_consolidate_decreases_plasticity():
    conf, daf, *_ = _make_configurator()
    original = daf.stdp_a_plus
    _push_mode(conf, _consolidate_state, K)
    assert conf._current_mode == ConfiguratorMode.CONSOLIDATE
    assert daf.stdp_a_plus < original, (
        f"expected stdp_a_plus < {original}, got {daf.stdp_a_plus}"
    )


# ---------------------------------------------------------------------------
# Test 6: EXPLORE → NEUTRAL restores original parameters
# ---------------------------------------------------------------------------

def test_neutral_restores_originals():
    conf, daf, metacog, hac_pred = _make_configurator()
    orig_plus = daf.stdp_a_plus
    orig_minus = daf.stdp_a_minus
    orig_decay = hac_pred.memory_decay
    orig_alpha = metacog.alpha
    orig_beta = metacog.beta
    orig_gamma = metacog.gamma

    # Switch to EXPLORE
    _push_mode(conf, _explore_state, K)
    assert conf._current_mode == ConfiguratorMode.EXPLORE

    # Now push neutral conditions for K cycles
    _push_mode(conf, _neutral_state, K)
    assert conf._current_mode == ConfiguratorMode.NEUTRAL

    assert daf.stdp_a_plus == pytest.approx(orig_plus, rel=1e-9)
    assert daf.stdp_a_minus == pytest.approx(orig_minus, rel=1e-9)
    assert hac_pred.memory_decay == pytest.approx(orig_decay, rel=1e-9)
    assert metacog.alpha == pytest.approx(orig_alpha, rel=1e-9)
    assert metacog.beta == pytest.approx(orig_beta, rel=1e-9)
    assert metacog.gamma == pytest.approx(orig_gamma, rel=1e-9)


# ---------------------------------------------------------------------------
# Test 7: GOAL_SEEKING has priority over EXPLORE
# ---------------------------------------------------------------------------

def test_goal_seeking_priority():
    conf, *_ = _make_configurator()
    # goal_state has both explore conditions AND goal > threshold
    _push_mode(conf, _goal_state, K)
    assert conf._current_mode == ConfiguratorMode.GOAL_SEEKING, (
        f"expected GOAL_SEEKING, got {conf._current_mode}"
    )


# ---------------------------------------------------------------------------
# Test 8: GOAL_SEEKING sets alpha=0.5, beta=0.5, gamma=0.0
# ---------------------------------------------------------------------------

def test_goal_seeking_changes_metacog_weights():
    conf, _, metacog, _ = _make_configurator()
    _push_mode(conf, _goal_state, K)
    assert conf._current_mode == ConfiguratorMode.GOAL_SEEKING
    assert metacog.alpha == pytest.approx(0.5, abs=1e-9)
    assert metacog.beta == pytest.approx(0.5, abs=1e-9)
    assert metacog.gamma == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Test 9: divergence safeguard — forced back to NEUTRAL after max_explore_cycles
# ---------------------------------------------------------------------------

def test_divergence_safeguard():
    conf, *_ = _make_configurator()

    # Enter EXPLORE
    _push_mode(conf, _explore_state, K)
    assert conf._current_mode == ConfiguratorMode.EXPLORE

    # Keep pushing explore conditions; after max_explore_cycles + K cycles the
    # safeguard fires (cycles_in_mode > MAX_EX) and forces candidate=NEUTRAL,
    # which then satisfies hysteresis after K consecutive neutral-forced cycles.
    # We push MAX_EX + K more cycles to be safe.
    _push_mode(conf, _explore_state, MAX_EX + K + 1)
    assert conf._current_mode == ConfiguratorMode.NEUTRAL, (
        f"expected NEUTRAL after divergence safeguard, got {conf._current_mode}"
    )


# ---------------------------------------------------------------------------
# Test 10: cycles_in_mode increments each call (no mode switch)
# ---------------------------------------------------------------------------

def test_action_cycles_in_mode_counter():
    conf, *_ = _make_configurator()
    # Stay in NEUTRAL with neutral states (no switch)
    actions = []
    for _ in range(5):
        action = conf.update(_neutral_state())
        actions.append(action)

    # cycles_in_mode should increment each call
    for i, action in enumerate(actions):
        assert action.cycles_in_mode == i + 1, (
            f"expected cycles_in_mode={i + 1}, got {action.cycles_in_mode}"
        )
