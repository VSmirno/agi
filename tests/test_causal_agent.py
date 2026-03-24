"""Integration tests for agent/agent.py — CausalAgent."""

import numpy as np
import pytest

from snks.agent.agent import CausalAgent
from snks.daf.types import CausalAgentConfig, DafConfig, EncoderConfig, PipelineConfig, SKSConfig


def make_small_config() -> CausalAgentConfig:
    """Small config for fast tests."""
    return CausalAgentConfig(
        pipeline=PipelineConfig(
            daf=DafConfig(
                num_nodes=5000,
                avg_degree=10,
                oscillator_model="fhn",
                dt=0.01,
                noise_sigma=0.005,
                fhn_I_base=0.0,
                device="cpu",
            ),
            encoder=EncoderConfig(
                sdr_size=4096,
                sdr_sparsity=0.04,
            ),
            sks=SKSConfig(
                coherence_mode="rate",
                min_cluster_size=5,
                dbscan_min_samples=5,
            ),
            steps_per_cycle=50,
            device="cpu",
        ),
        motor_sdr_size=200,
        causal_min_observations=1,
        curiosity_epsilon=0.3,
    )


class TestCausalAgent:
    def test_creation(self):
        config = make_small_config()
        agent = CausalAgent(config)
        assert agent.pipeline is not None
        assert agent.causal_model is not None
        assert agent.simulator is not None

    def test_step_returns_valid_action(self):
        config = make_small_config()
        agent = CausalAgent(config)
        obs = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        action = agent.step(obs)
        assert 0 <= action < 5

    def test_observe_result(self):
        config = make_small_config()
        agent = CausalAgent(config)
        obs1 = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        action = agent.step(obs1)
        obs2 = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        pe = agent.observe_result(obs2)
        assert isinstance(pe, float)

    def test_step_observe_loop(self):
        config = make_small_config()
        agent = CausalAgent(config)
        for _ in range(3):
            obs = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
            action = agent.step(obs)
            obs2 = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
            agent.observe_result(obs2)
        assert agent._step_count == 3

    def test_motor_zone_valid(self):
        config = make_small_config()
        agent = CausalAgent(config)
        assert agent.motor.motor_zone_start > 0
        assert agent.motor.motor_zone_start + agent.motor.motor_zone_size == config.pipeline.daf.num_nodes
