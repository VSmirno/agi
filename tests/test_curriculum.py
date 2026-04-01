"""Tests for Stage 39: Curriculum Learning + Adaptive Exploration."""

from __future__ import annotations

import numpy as np
import pytest

from snks.agent.curriculum import (
    CurriculumScheduler,
    CurriculumStage,
    EpsilonScheduler,
    PredictionErrorExplorer,
    CurriculumTrainer,
)
from snks.agent.pure_daf_agent import PureDafAgent, PureDafConfig


# ── CurriculumScheduler ────────────────────────────────────────────────

class TestCurriculumScheduler:
    def test_init_starts_at_stage_0(self):
        stages = [
            CurriculumStage("env_a", gate_threshold=0.5, min_episodes=3),
            CurriculumStage("env_b", gate_threshold=0.3, min_episodes=5),
        ]
        sched = CurriculumScheduler(stages)
        assert sched.current_stage_idx == 0
        assert sched.current_stage.env_name == "env_a"

    def test_no_promote_below_gate(self):
        stages = [
            CurriculumStage("env_a", gate_threshold=0.5, min_episodes=2),
            CurriculumStage("env_b", gate_threshold=0.3, min_episodes=2),
        ]
        sched = CurriculumScheduler(stages)
        sched.record_episode(success=False)
        sched.record_episode(success=False)
        assert sched.current_stage_idx == 0

    def test_promote_when_gate_met(self):
        stages = [
            CurriculumStage("env_a", gate_threshold=0.5, min_episodes=2),
            CurriculumStage("env_b", gate_threshold=0.3, min_episodes=2),
        ]
        sched = CurriculumScheduler(stages)
        sched.record_episode(success=True)
        sched.record_episode(success=True)
        # 2/2 = 1.0 >= 0.5, min_episodes met → promote
        assert sched.current_stage_idx == 1
        assert sched.current_stage.env_name == "env_b"

    def test_no_promote_past_last_stage(self):
        stages = [
            CurriculumStage("env_a", gate_threshold=0.5, min_episodes=1),
        ]
        sched = CurriculumScheduler(stages)
        sched.record_episode(success=True)
        assert sched.current_stage_idx == 0  # stay at last
        assert sched.is_complete

    def test_success_rate_rolling_window(self):
        stages = [
            CurriculumStage("env_a", gate_threshold=0.6, min_episodes=3),
            CurriculumStage("env_b", gate_threshold=0.3, min_episodes=1),
        ]
        sched = CurriculumScheduler(stages)
        sched.record_episode(success=False)
        sched.record_episode(success=True)
        sched.record_episode(success=True)
        # 2/3 = 0.67 >= 0.6, promoted
        assert sched.current_stage_idx == 1


# ── EpsilonScheduler ───────────────────────────────────────────────────

class TestEpsilonScheduler:
    def test_initial_value(self):
        eps = EpsilonScheduler(initial=0.7, decay=0.95, floor=0.1)
        assert eps.value == pytest.approx(0.7)

    def test_decay_per_step(self):
        eps = EpsilonScheduler(initial=1.0, decay=0.5, floor=0.0)
        eps.step()
        assert eps.value == pytest.approx(0.5)
        eps.step()
        assert eps.value == pytest.approx(0.25)

    def test_floor_respected(self):
        eps = EpsilonScheduler(initial=0.5, decay=0.1, floor=0.2)
        for _ in range(100):
            eps.step()
        assert eps.value == pytest.approx(0.2)

    def test_no_negative(self):
        eps = EpsilonScheduler(initial=0.01, decay=0.001, floor=0.0)
        for _ in range(100):
            eps.step()
        assert eps.value >= 0.0


# ── PredictionErrorExplorer ────────────────────────────────────────────

class TestPredictionErrorExplorer:
    def test_init_uniform_bonus(self):
        pe = PredictionErrorExplorer(n_actions=4, window_size=5)
        bonuses = pe.action_bonuses()
        assert len(bonuses) == 4
        # All uniform initially
        assert all(abs(b - 0.25) < 0.01 for b in bonuses)

    def test_record_shifts_bonus(self):
        pe = PredictionErrorExplorer(n_actions=3, window_size=3)
        # Action 0 has high PE
        pe.record(action=0, prediction_error=1.0)
        pe.record(action=1, prediction_error=0.1)
        pe.record(action=2, prediction_error=0.1)
        bonuses = pe.action_bonuses()
        # Action 0 should have highest bonus
        assert bonuses[0] > bonuses[1]
        assert bonuses[0] > bonuses[2]

    def test_select_with_bonus_returns_valid_action(self):
        pe = PredictionErrorExplorer(n_actions=5, window_size=3)
        for _ in range(10):
            action = pe.select_with_bonus()
            assert 0 <= action < 5


# ── CurriculumTrainer ─────────────────────────────────────────────────

class TestCurriculumTrainer:
    def test_trainer_creates_with_defaults(self):
        trainer = CurriculumTrainer()
        assert trainer.scheduler is not None
        assert trainer.epsilon is not None

    def test_trainer_runs_one_episode_on_dummy_env(self):
        """Smoke test: run single episode on array env."""
        from snks.env.adapter import ArrayEnvAdapter

        class SimpleEnv:
            def reset(self, seed=None):
                self._step = 0
                return np.zeros(16)
            def step(self, action):
                self._step += 1
                done = self._step >= 5
                reward = 1.0 if done else 0.0
                return np.full(16, self._step / 10.0), reward, done, False, {}

        env = ArrayEnvAdapter(SimpleEnv(), n_actions=3, name="simple")
        trainer = CurriculumTrainer(
            stages=[CurriculumStage("simple", gate_threshold=0.5, min_episodes=1)],
            epsilon_initial=0.5,
        )
        result = trainer.train_episode(env)
        assert result is not None
        assert result.steps > 0

    def test_epsilon_decays_after_episode(self):
        trainer = CurriculumTrainer(
            stages=[CurriculumStage("test", gate_threshold=0.9, min_episodes=1)],
            epsilon_initial=0.8,
            epsilon_decay=0.5,
        )
        initial = trainer.epsilon.value
        # Simulate episode completion
        trainer.epsilon.step()
        assert trainer.epsilon.value < initial
