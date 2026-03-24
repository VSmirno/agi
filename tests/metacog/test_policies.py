"""Tests for metacognitive policies."""

import pytest
from dataclasses import dataclass

from snks.metacog.policies import NullPolicy, NoisePolicy, STDPPolicy
from snks.metacog.monitor import MetacogState
from snks.daf.types import DafConfig


def make_state(confidence: float) -> MetacogState:
    return MetacogState(confidence=confidence, dominance=0.5, stability=0.5, pred_error=0.1)


def make_config(**kwargs) -> DafConfig:
    cfg = DafConfig()
    for k, v in kwargs.items():
        setattr(cfg, k, v)
    return cfg


class TestNullPolicy:
    def test_null_policy_no_change(self):
        policy = NullPolicy()
        cfg = make_config(noise_sigma=0.01, stdp_a_plus=0.01)
        state = make_state(0.5)
        policy.apply(state, cfg)
        assert cfg.noise_sigma == pytest.approx(0.01)
        assert cfg.stdp_a_plus == pytest.approx(0.01)


class TestNoisePolicy:
    def test_noise_policy_high_confidence(self):
        """confidence=1.0 -> noise_sigma ~= base_sigma."""
        policy = NoisePolicy(strength=1.0)
        cfg = make_config(noise_sigma=0.01)
        state = make_state(1.0)
        policy.apply(state, cfg)
        assert cfg.noise_sigma == pytest.approx(0.01)

    def test_noise_policy_low_confidence(self):
        """confidence=0.0 -> noise_sigma = base * (1 + strength)."""
        policy = NoisePolicy(strength=1.0)
        cfg = make_config(noise_sigma=0.01)
        state = make_state(0.0)
        policy.apply(state, cfg)
        assert cfg.noise_sigma == pytest.approx(0.02)

    def test_noise_policy_mid_confidence(self):
        """confidence=0.5 -> noise_sigma = base * (1 + 0.5 * strength)."""
        policy = NoisePolicy(strength=1.0)
        cfg = make_config(noise_sigma=0.01)
        state = make_state(0.5)
        policy.apply(state, cfg)
        assert cfg.noise_sigma == pytest.approx(0.015)

    def test_noise_policy_base_fixed_after_first_call(self):
        """base_sigma is captured from first call, not mutated config."""
        policy = NoisePolicy(strength=1.0)
        cfg = make_config(noise_sigma=0.01)
        # First call with confidence=0.0: noise becomes 0.02
        policy.apply(make_state(0.0), cfg)
        assert cfg.noise_sigma == pytest.approx(0.02)
        # Second call with confidence=1.0: should return to base (0.01), not 0.02
        policy.apply(make_state(1.0), cfg)
        assert cfg.noise_sigma == pytest.approx(0.01)


class TestSTDPPolicy:
    def test_stdp_policy_high_confidence(self):
        """confidence=1.0 -> a_plus = base * (1 + strength)."""
        policy = STDPPolicy(strength=1.0)
        cfg = make_config(stdp_a_plus=0.01)
        state = make_state(1.0)
        policy.apply(state, cfg)
        assert cfg.stdp_a_plus == pytest.approx(0.02)

    def test_stdp_policy_low_confidence(self):
        """confidence=0.0 -> a_plus ~= base."""
        policy = STDPPolicy(strength=1.0)
        cfg = make_config(stdp_a_plus=0.01)
        state = make_state(0.0)
        policy.apply(state, cfg)
        assert cfg.stdp_a_plus == pytest.approx(0.01)

    def test_stdp_policy_base_fixed_after_first_call(self):
        """base_a_plus captured from first call."""
        policy = STDPPolicy(strength=1.0)
        cfg = make_config(stdp_a_plus=0.01)
        # First call: confidence=1.0 -> a_plus = 0.02
        policy.apply(make_state(1.0), cfg)
        assert cfg.stdp_a_plus == pytest.approx(0.02)
        # Second call: confidence=0.0 -> a_plus = base = 0.01
        policy.apply(make_state(0.0), cfg)
        assert cfg.stdp_a_plus == pytest.approx(0.01)
