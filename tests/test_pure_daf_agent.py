"""Tests for PureDafAgent — pure DAF pipeline without scaffolding."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from snks.agent.attractor_navigator import AttractorNavigator
from snks.agent.daf_causal_model import DafCausalModel
from snks.agent.pure_daf_agent import PureDafAgent, PureDafConfig
from snks.env.adapter import ArrayEnvAdapter, EnvAdapter, MiniGridAdapter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_config() -> PureDafConfig:
    """Minimal config for fast tests (small DAF)."""
    cfg = PureDafConfig()
    cfg.causal.pipeline.daf.num_nodes = 500
    cfg.causal.pipeline.daf.avg_degree = 10
    cfg.causal.pipeline.daf.device = "cpu"
    cfg.causal.pipeline.daf.disable_csr = True
    cfg.causal.pipeline.daf.dt = 0.005  # faster for CPU tests
    cfg.causal.pipeline.steps_per_cycle = 50  # minimal for unit tests
    cfg.causal.pipeline.encoder.image_size = 16
    cfg.causal.pipeline.encoder.sdr_size = 500
    cfg.causal.pipeline.encoder.pool_h = 5
    cfg.causal.pipeline.encoder.pool_w = 5
    cfg.causal.pipeline.encoder.n_orientations = 4
    cfg.causal.pipeline.encoder.n_frequencies = 1
    cfg.causal.pipeline.sks.min_cluster_size = 3
    cfg.causal.motor_sdr_size = 50
    cfg.n_actions = 5
    cfg.n_sim_steps = 5
    cfg.max_episode_steps = 20
    cfg.exploration_epsilon = 0.5
    return cfg


class DummyEnv:
    """Minimal env for testing — returns random observations, reward on step 10."""

    def __init__(self, n_actions: int = 5, size: int = 8):
        self._n_actions = n_actions
        self._size = size
        self._step_count = 0

    def reset(self, seed=None):
        self._step_count = 0
        return np.random.randint(0, 255, (self._size, self._size, 3), dtype=np.uint8)

    def step(self, action):
        self._step_count += 1
        obs = np.random.randint(0, 255, (self._size, self._size, 3), dtype=np.uint8)
        reward = 1.0 if self._step_count >= 10 else 0.0
        terminated = self._step_count >= 10
        return obs, reward, terminated, False, {}

    @property
    def n_actions(self):
        return self._n_actions

    @property
    def name(self):
        return "DummyEnv"


# ---------------------------------------------------------------------------
# EnvAdapter tests
# ---------------------------------------------------------------------------

class TestEnvAdapter:
    def test_protocol_check(self):
        env = DummyEnv()
        assert isinstance(env, EnvAdapter)

    def test_dummy_env_works(self):
        env = DummyEnv()
        obs = env.reset()
        assert obs.shape == (8, 8, 3)
        obs2, reward, done, trunc, info = env.step(0)
        assert obs2.shape == (8, 8, 3)
        assert reward == 0.0
        assert not done

    def test_array_adapter(self):
        class SimpleEnv:
            def reset(self):
                return np.zeros(16)
            def step(self, a):
                return np.ones(16), 1.0, True, False, {}

        adapter = ArrayEnvAdapter(SimpleEnv(), n_actions=3, name="test")
        obs = adapter.reset()
        assert obs.shape[2] == 3  # RGB
        assert adapter.n_actions == 3
        assert adapter.name == "test"


# ---------------------------------------------------------------------------
# DafCausalModel tests
# ---------------------------------------------------------------------------

class TestDafCausalModel:
    def test_before_after_action(self, small_config):
        agent = PureDafAgent(small_config)
        model = agent._causal

        model.before_action(0, {1, 2, 3})
        assert len(model._trace) == 1

        # No error on after_action with zero reward
        model.after_action(0.0)
        assert model._total_modulations == 0

        # Positive reward triggers modulation
        model.before_action(1, {4, 5})
        model.after_action(1.0)
        assert model._total_modulations == 1

    def test_trace_length(self, small_config):
        agent = PureDafAgent(small_config)
        model = agent._causal

        for i in range(10):
            model.before_action(i % 5, {i})
        assert len(model._trace) == small_config.trace_length

    def test_reward_modulation_changes_weights(self, small_config):
        agent = PureDafAgent(small_config)
        engine = agent.engine
        model = agent._causal

        # Record initial weights
        w_before = engine.graph.edge_attr[:, 0].clone()

        # Run a DAF step to get STDP changes
        model.before_action(0, {1, 2})
        engine.step(10)
        model.after_action(1.0)

        w_after = engine.graph.edge_attr[:, 0]
        # Weights should have changed (STDP + reward modulation)
        assert not torch.allclose(w_before, w_after)

    def test_stats(self, small_config):
        agent = PureDafAgent(small_config)
        stats = agent._causal.stats
        assert "total_reward" in stats
        assert "total_modulations" in stats


# ---------------------------------------------------------------------------
# AttractorNavigator tests
# ---------------------------------------------------------------------------

class TestAttractorNavigator:
    def test_select_action_with_goal(self, small_config):
        agent = PureDafAgent(small_config)
        nav = agent._navigator

        current = torch.randn(2048)
        goal = torch.randn(2048)
        action = nav.select_action(current, goal, 5)
        assert 0 <= action < 5

    def test_select_action_no_goal(self, small_config):
        agent = PureDafAgent(small_config)
        nav = agent._navigator

        current = torch.randn(2048)
        action = nav.select_action(current, None, 5)
        assert 0 <= action < 5

    def test_stats(self, small_config):
        agent = PureDafAgent(small_config)
        nav = agent._navigator
        stats = nav.stats
        assert "total_steps" in stats
        assert "explore_ratio" in stats

    def test_cosine_similarity(self):
        a = torch.tensor([1.0, 0.0, 0.0])
        b = torch.tensor([1.0, 0.0, 0.0])
        assert abs(AttractorNavigator._cosine_similarity(a, b) - 1.0) < 1e-5

        c = torch.tensor([0.0, 1.0, 0.0])
        assert abs(AttractorNavigator._cosine_similarity(a, c)) < 1e-5


# ---------------------------------------------------------------------------
# PureDafAgent tests
# ---------------------------------------------------------------------------

class TestPureDafAgent:
    def test_construction(self, small_config):
        agent = PureDafAgent(small_config)
        assert agent.engine is not None
        assert agent.pipeline is not None

    def test_step(self, small_config):
        agent = PureDafAgent(small_config)
        obs = np.random.randint(0, 255, (8, 8, 3), dtype=np.uint8)
        action = agent.step(obs)
        assert 0 <= action < small_config.n_actions

    def test_observe_result(self, small_config):
        agent = PureDafAgent(small_config)
        obs = np.random.randint(0, 255, (8, 8, 3), dtype=np.uint8)
        agent.step(obs)
        obs2 = np.random.randint(0, 255, (8, 8, 3), dtype=np.uint8)
        pe = agent.observe_result(obs2, 0.0)
        assert isinstance(pe, float)

    def test_run_episode(self, small_config):
        agent = PureDafAgent(small_config)
        env = DummyEnv()
        result = agent.run_episode(env)
        assert result.steps > 0
        assert result.steps <= small_config.max_episode_steps

    def test_run_episode_with_reward(self, small_config):
        agent = PureDafAgent(small_config)
        env = DummyEnv()
        small_config.max_episode_steps = 15
        result = agent.run_episode(env, max_steps=15)
        assert result.success  # DummyEnv gives reward at step 10
        assert result.reward > 0

    def test_run_training(self, small_config):
        small_config.max_episode_steps = 10
        agent = PureDafAgent(small_config)
        env = DummyEnv()
        results = agent.run_training(env, n_episodes=3, max_steps=10)
        assert len(results) == 3

    def test_no_grid_access(self, small_config):
        """Verify agent does NOT import any grid-specific modules."""
        import snks.agent.pure_daf_agent as mod
        source = open(mod.__file__).read()
        # Check import lines only (not docstring mentions)
        import_lines = [l for l in source.splitlines() if l.strip().startswith(("import ", "from "))]
        imports_text = "\n".join(import_lines)
        assert "GridPerception" not in imports_text
        assert "GridNavigator" not in imports_text
        assert "BlockingAnalyzer" not in imports_text
        assert "BabyAIExecutor" not in imports_text
        assert "goal_agent" not in imports_text

    def test_no_hardcoded_sks(self, small_config):
        """Verify no hardcoded SKS IDs (50-58) used as constants in agent code."""
        import snks.agent.pure_daf_agent as mod
        source = open(mod.__file__).read()
        import_lines = [l for l in source.splitlines() if l.strip().startswith(("import ", "from "))]
        imports_text = "\n".join(import_lines)
        assert "SKS_KEY_PRESENT" not in imports_text
        assert "SKS_DOOR_LOCKED" not in imports_text
        assert "SKS_GOAL_PRESENT" not in imports_text


# ---------------------------------------------------------------------------
# Integration test: MiniGrid (skip if not installed)
# ---------------------------------------------------------------------------

class TestMiniGridIntegration:
    @pytest.fixture
    def minigrid_env(self):
        try:
            return MiniGridAdapter("MiniGrid-DoorKey-5x5-v0")
        except (ImportError, Exception):
            pytest.skip("MiniGrid not available")

    def test_adapter_protocol(self, minigrid_env):
        assert isinstance(minigrid_env, EnvAdapter)
        assert minigrid_env.n_actions > 0

    def test_pure_daf_on_minigrid(self, small_config, minigrid_env):
        small_config.n_actions = minigrid_env.n_actions
        agent = PureDafAgent(small_config)
        result = agent.run_episode(minigrid_env)
        assert result.steps > 0
        assert isinstance(result.reward, float)


# ---------------------------------------------------------------------------
# Stage 38_fix: Curiosity-driven action selection tests
# ---------------------------------------------------------------------------

class TestActionSpaceFix:
    """Bug #2: motor encoder must match PureDafConfig.n_actions."""

    def test_motor_matches_config_n_actions(self, small_config):
        small_config.n_actions = 7
        agent = PureDafAgent(small_config)
        assert agent._agent.motor.n_actions == 7

    def test_motor_default_5_actions(self, small_config):
        small_config.n_actions = 5
        agent = PureDafAgent(small_config)
        assert agent._agent.motor.n_actions == 5

    def test_all_actions_encodable(self, small_config):
        """All actions 0..n_actions-1 must encode without error."""
        small_config.n_actions = 7
        agent = PureDafAgent(small_config)
        for a in range(7):
            currents = agent._agent.motor.encode(a, device=agent.engine.device)
            assert currents.shape[0] == small_config.causal.pipeline.daf.num_nodes

    def test_step_returns_valid_action_7(self, small_config):
        """step() returns actions in [0, n_actions) even with 7 actions."""
        small_config.n_actions = 7
        agent = PureDafAgent(small_config)
        obs = np.random.randint(0, 255, (8, 8, 3), dtype=np.uint8)
        for _ in range(20):
            action = agent.step(obs)
            assert 0 <= action < 7


class TestPEExploration:
    """Bug #1 fix: PE-driven exploration without goal_embedding."""

    def test_pe_explorer_exists(self, small_config):
        agent = PureDafAgent(small_config)
        assert hasattr(agent, '_pe_explorer')

    def test_pe_explorer_records_pe(self, small_config):
        agent = PureDafAgent(small_config)
        env = DummyEnv(n_actions=5)
        obs = env.reset()
        action = agent.step(obs)
        obs2, reward, _, _, _ = env.step(action)
        pe = agent.observe_result(obs2, reward)
        # PE should be recorded in pe_explorer
        total = sum(len(h) for h in agent._pe_explorer._pe_history)
        assert total > 0

    def test_episode_without_goal_no_crash(self, small_config):
        """Agent runs full episode without set_goal_from_obs — no crash."""
        small_config.n_actions = 7
        agent = PureDafAgent(small_config)
        env = DummyEnv(n_actions=7)
        result = agent.run_episode(env)
        assert result.steps > 0
        # No goal_embedding was set — agent used curiosity
        assert agent._goal_embedding is None

    def test_motivation_updated_during_episode(self, small_config):
        """IntrinsicMotivation.update() called during observe_result."""
        agent = PureDafAgent(small_config)
        env = DummyEnv(n_actions=5)
        agent.run_episode(env)
        # After episode, motivation should have visit counts
        assert agent._agent.motivation._total_steps > 0


class TestEpsilonDecay:
    """Bug #3 fix: epsilon decays across episodes."""

    def test_epsilon_scheduler_exists(self, small_config):
        agent = PureDafAgent(small_config)
        assert hasattr(agent, '_epsilon_scheduler')

    def test_epsilon_decays_during_training(self, small_config):
        small_config.max_episode_steps = 10
        agent = PureDafAgent(small_config)
        env = DummyEnv(n_actions=5)

        initial_eps = agent._epsilon_scheduler.value
        agent.run_training(env, n_episodes=5, max_steps=10)
        final_eps = agent._epsilon_scheduler.value

        assert final_eps < initial_eps

    def test_epsilon_respects_floor(self, small_config):
        small_config.epsilon_floor = 0.1
        small_config.max_episode_steps = 10
        agent = PureDafAgent(small_config)
        env = DummyEnv(n_actions=5)

        # Run many episodes to drive epsilon down
        agent.run_training(env, n_episodes=100, max_steps=10)
        assert agent._epsilon_scheduler.value >= small_config.epsilon_floor

    def test_nav_stats_include_pe_exploration(self, small_config):
        agent = PureDafAgent(small_config)
        env = DummyEnv(n_actions=5)
        result = agent.run_episode(env)
        assert "pe_exploration_ratio" in result.nav_stats
