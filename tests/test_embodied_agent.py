"""Tests for EmbodiedAgent (Stage 14)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from snks.agent.embodied_agent import EmbodiedAgent, EmbodiedAgentConfig
from snks.daf.types import (
    CausalAgentConfig,
    ConfiguratorAction,
    ConfiguratorConfig,
    CostModuleConfig,
    DafConfig,
    EncoderConfig,
    PipelineConfig,
    SKSConfig,
)


def _make_small_config(
    configurator_enabled: bool = True,
    cost_module_enabled: bool = True,
) -> EmbodiedAgentConfig:
    """Minimal config for fast unit tests."""
    pipeline = PipelineConfig(
        daf=DafConfig(
            num_nodes=1000,
            avg_degree=8,
            oscillator_model="fhn",
            dt=0.01,
            noise_sigma=0.005,
            fhn_I_base=0.0,
            device="cpu",
        ),
        encoder=EncoderConfig(sdr_size=512, sdr_sparsity=0.04),
        sks=SKSConfig(coherence_mode="rate", min_cluster_size=5, dbscan_min_samples=5),
        steps_per_cycle=10,
        device="cpu",
        cost_module=CostModuleConfig(enabled=cost_module_enabled),
        configurator=ConfiguratorConfig(enabled=configurator_enabled),
    )
    causal = CausalAgentConfig(
        pipeline=pipeline,
        motor_sdr_size=100,
        causal_min_observations=1,
        curiosity_epsilon=0.5,
    )
    return EmbodiedAgentConfig(
        causal=causal,
        use_stochastic_planner=True,
        n_plan_samples=2,
        max_plan_depth=3,
    )


def _random_obs() -> np.ndarray:
    return np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)


class TestStepObserveResultCycle:
    """test_step_observe_result_cycle: one full step()+observe_result() cycle completes."""

    def test_step_returns_valid_action(self):
        agent = EmbodiedAgent(_make_small_config())
        obs = _random_obs()
        action = agent.step(obs)
        assert isinstance(action, int)
        assert 0 <= action < agent.n_actions

    def test_observe_result_returns_float(self):
        agent = EmbodiedAgent(_make_small_config())
        obs1 = _random_obs()
        agent.step(obs1)
        obs2 = _random_obs()
        pe = agent.observe_result(obs2)
        assert isinstance(pe, float)
        assert 0.0 <= pe <= 1.0

    def test_full_cycle_no_error(self):
        """Full step → env_step → observe_result cycle completes without error."""
        agent = EmbodiedAgent(_make_small_config())
        obs = _random_obs()
        action = agent.step(obs)
        obs_next = _random_obs()
        pe = agent.observe_result(obs_next)
        assert isinstance(action, int)
        assert isinstance(pe, float)

    def test_last_cycle_result_cached(self):
        """Pipeline.last_cycle_result is populated after step()."""
        agent = EmbodiedAgent(_make_small_config())
        obs = _random_obs()
        agent.step(obs)
        assert agent.causal_agent.pipeline.last_cycle_result is not None


class TestAblationFlags:
    """test_ablation_flags: each config combination instantiates and runs one step."""

    @pytest.mark.parametrize(
        "configurator_enabled,cost_module_enabled",
        [
            (True, True),    # Full stack
            (False, True),   # No Configurator
            (False, False),  # No ICM + No Configurator (Stage 6 baseline)
        ],
    )
    def test_ablation_combination(self, configurator_enabled, cost_module_enabled):
        config = _make_small_config(
            configurator_enabled=configurator_enabled,
            cost_module_enabled=cost_module_enabled,
        )
        agent = EmbodiedAgent(config)
        obs = _random_obs()
        action = agent.step(obs)
        assert 0 <= action < agent.n_actions

    def test_stochastic_planner_disabled(self):
        config = _make_small_config()
        config.use_stochastic_planner = False
        agent = EmbodiedAgent(config)
        obs = _random_obs()
        action = agent.step(obs)
        assert 0 <= action < agent.n_actions


class TestGoalSeekingCallsSimulator:
    """test_goal_seeking_calls_simulator: when mode=goal_seeking and goal_sks set,
    find_plan_stochastic is called."""

    def test_simulator_called_on_goal_seeking(self):
        agent = EmbodiedAgent(_make_small_config())
        obs = _random_obs()

        # First step to populate last_cycle_result
        agent.step(obs)

        # Inject a mock last_cycle_result with mode=goal_seeking
        from snks.pipeline.runner import CycleResult

        mock_result = MagicMock(spec=CycleResult)
        mock_result.sks_clusters = {1: {10, 11}, 2: {20, 21}}
        mock_conf_action = MagicMock(spec=ConfiguratorAction)
        mock_conf_action.mode = "goal_seeking"
        mock_result.configurator_action = mock_conf_action

        agent.causal_agent.pipeline.last_cycle_result = mock_result

        # Set goal SKS so planner path is triggered
        agent.set_goal_sks({99, 100})

        # Patch simulator to track calls and return a valid plan
        with patch.object(
            agent.simulator,
            "find_plan_stochastic",
            return_value=([2], 0.8),
        ) as mock_plan:
            # Patch CausalAgent.step to avoid running full pipeline again
            with patch.object(agent.causal_agent, "step", return_value=0):
                returned_action = agent.step(obs)

            mock_plan.assert_called_once()
            call_kwargs = mock_plan.call_args
            assert call_kwargs is not None
            # Returned action should be first element of plan (action 2)
            assert returned_action == 2

    def test_no_simulator_when_goal_sks_none(self):
        """When _goal_sks is None, simulator is NOT called even in goal_seeking mode."""
        agent = EmbodiedAgent(_make_small_config())
        obs = _random_obs()
        agent.step(obs)

        from snks.pipeline.runner import CycleResult

        mock_result = MagicMock(spec=CycleResult)
        mock_result.sks_clusters = {}
        mock_conf_action = MagicMock(spec=ConfiguratorAction)
        mock_conf_action.mode = "goal_seeking"
        mock_result.configurator_action = mock_conf_action
        agent.causal_agent.pipeline.last_cycle_result = mock_result

        # goal_sks is None
        agent.set_goal_sks(None)

        with patch.object(agent.simulator, "find_plan_stochastic") as mock_plan:
            with patch.object(agent.causal_agent, "step", return_value=1):
                agent.step(obs)
            mock_plan.assert_not_called()

    def test_explore_mode_returns_random_action(self):
        """In explore mode, action is random (not necessarily CausalAgent default)."""
        agent = EmbodiedAgent(_make_small_config())
        obs = _random_obs()
        agent.step(obs)

        from snks.pipeline.runner import CycleResult

        mock_result = MagicMock(spec=CycleResult)
        mock_result.sks_clusters = {}
        mock_conf_action = MagicMock(spec=ConfiguratorAction)
        mock_conf_action.mode = "explore"
        mock_result.configurator_action = mock_conf_action
        agent.causal_agent.pipeline.last_cycle_result = mock_result

        actions = set()
        with patch.object(agent.causal_agent, "step", return_value=0):
            for _ in range(50):
                a = agent.step(obs)
                assert 0 <= a < agent.n_actions
                actions.add(a)
        # With 50 samples from 5 actions, expect > 1 unique action
        assert len(actions) > 1
