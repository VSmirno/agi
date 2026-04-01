"""Curriculum Learning + Adaptive Exploration for Pure DAF Agent (Stage 39).

CurriculumScheduler: manages progressive env difficulty
EpsilonScheduler: decaying exploration rate
PredictionErrorExplorer: PE-based action bias
CurriculumTrainer: orchestrates curriculum training loop
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from snks.agent.pure_daf_agent import PureDafAgent, PureDafConfig, PureDafEpisodeResult
from snks.env.adapter import EnvAdapter


@dataclass
class CurriculumStage:
    """One stage in a curriculum sequence."""
    env_name: str
    gate_threshold: float  # success rate to advance
    min_episodes: int  # minimum episodes before promotion check


class CurriculumScheduler:
    """Manages progressive difficulty through env stages.

    Promotes to next stage when success rate >= gate_threshold
    over min_episodes consecutive episodes.
    """

    def __init__(self, stages: list[CurriculumStage]) -> None:
        self._stages = stages
        self._current_idx = 0
        self._episode_results: list[bool] = []

    @property
    def current_stage_idx(self) -> int:
        return self._current_idx

    @property
    def current_stage(self) -> CurriculumStage:
        return self._stages[self._current_idx]

    @property
    def is_complete(self) -> bool:
        """True if at last stage and gate is met."""
        if self._current_idx < len(self._stages) - 1:
            return False
        stage = self.current_stage
        if len(self._episode_results) < stage.min_episodes:
            return False
        recent = self._episode_results[-stage.min_episodes:]
        rate = sum(recent) / len(recent)
        return rate >= stage.gate_threshold

    def record_episode(self, success: bool) -> None:
        """Record episode result and check for promotion."""
        self._episode_results.append(success)
        self._try_promote()

    def _try_promote(self) -> None:
        """Promote to next stage if gate threshold met."""
        if self._current_idx >= len(self._stages) - 1:
            return
        stage = self.current_stage
        if len(self._episode_results) < stage.min_episodes:
            return
        recent = self._episode_results[-stage.min_episodes:]
        rate = sum(recent) / len(recent)
        if rate >= stage.gate_threshold:
            self._current_idx += 1
            self._episode_results.clear()

    @property
    def stats(self) -> dict:
        total = len(self._episode_results)
        successes = sum(self._episode_results) if self._episode_results else 0
        return {
            "stage_idx": self._current_idx,
            "stage_name": self.current_stage.env_name,
            "episodes_in_stage": total,
            "success_rate": successes / max(total, 1),
            "is_complete": self.is_complete,
        }


class EpsilonScheduler:
    """Exponentially decaying epsilon for exploration.

    epsilon(t) = max(floor, initial * decay^t)
    """

    def __init__(
        self,
        initial: float = 0.7,
        decay: float = 0.95,
        floor: float = 0.1,
    ) -> None:
        self._initial = initial
        self._decay = decay
        self._floor = floor
        self._value = initial
        self._step_count = 0

    @property
    def value(self) -> float:
        return self._value

    def step(self) -> float:
        """Decay epsilon by one step. Returns new value."""
        self._step_count += 1
        self._value = max(self._floor, self._initial * (self._decay ** self._step_count))
        return self._value

    def reset(self) -> None:
        self._value = self._initial
        self._step_count = 0


class PredictionErrorExplorer:
    """Biases exploration toward actions with high prediction error.

    Maintains rolling average of PE per action. Softmax over PE means
    gives probability distribution → actions with higher PE get more
    exploration budget.
    """

    def __init__(self, n_actions: int, window_size: int = 10, temperature: float = 2.0) -> None:
        self._n_actions = n_actions
        self._pe_history: list[deque[float]] = [
            deque(maxlen=window_size) for _ in range(n_actions)
        ]
        self._temperature = temperature

    def record(self, action: int, prediction_error: float) -> None:
        """Record PE for an action."""
        if 0 <= action < self._n_actions:
            self._pe_history[action].append(prediction_error)

    def action_bonuses(self) -> list[float]:
        """Softmax probability distribution over actions based on PE.

        Higher PE → higher probability → more exploration.
        """
        means = []
        for h in self._pe_history:
            if h:
                means.append(sum(h) / len(h))
            else:
                means.append(1.0)  # uniform prior for unexplored

        # Softmax with temperature
        max_m = max(means)
        exps = [np.exp((m - max_m) * self._temperature) for m in means]
        total = sum(exps)
        if total < 1e-12:
            return [1.0 / self._n_actions] * self._n_actions
        return [e / total for e in exps]

    def select_with_bonus(self) -> int:
        """Select action proportional to PE bonuses."""
        probs = self.action_bonuses()
        return int(np.random.choice(self._n_actions, p=probs))


class CurriculumTrainer:
    """Orchestrates curriculum training with decaying epsilon and PE exploration.

    Wraps PureDafAgent and adds:
    1. CurriculumScheduler: progressive env difficulty
    2. EpsilonScheduler: decaying exploration
    3. PredictionErrorExplorer: curiosity-biased action selection
    """

    def __init__(
        self,
        config: PureDafConfig | None = None,
        stages: list[CurriculumStage] | None = None,
        epsilon_initial: float = 0.7,
        epsilon_decay: float = 0.95,
        epsilon_floor: float = 0.1,
        pe_window: int = 10,
        pe_temperature: float = 2.0,
    ) -> None:
        if config is None:
            config = PureDafConfig()

        self.agent = PureDafAgent(config)

        # Default curriculum: Empty-5x5 → Empty-8x8 → DoorKey-5x5
        if stages is None:
            stages = [
                CurriculumStage("MiniGrid-Empty-5x5-v0", gate_threshold=0.4, min_episodes=5),
                CurriculumStage("MiniGrid-Empty-8x8-v0", gate_threshold=0.4, min_episodes=5),
                CurriculumStage("MiniGrid-DoorKey-5x5-v0", gate_threshold=0.2, min_episodes=5),
            ]

        self.scheduler = CurriculumScheduler(stages)
        self.epsilon = EpsilonScheduler(epsilon_initial, epsilon_decay, epsilon_floor)
        self.pe_explorer = PredictionErrorExplorer(
            n_actions=config.n_actions, window_size=pe_window, temperature=pe_temperature,
        )
        self._episode_count = 0
        self._all_results: list[PureDafEpisodeResult] = []

    def train_episode(self, env: EnvAdapter, max_steps: int | None = None) -> PureDafEpisodeResult:
        """Run one training episode with curriculum enhancements.

        Overrides agent epsilon with scheduler value.
        Records PE per action for curiosity bias.
        """
        # Update navigator epsilon from scheduler
        self.agent._navigator._epsilon = self.epsilon.value

        # Run episode with PE tracking
        result = self._run_episode_with_pe(env, max_steps)

        # Update schedulers
        self.scheduler.record_episode(result.success)
        self.epsilon.step()
        self._episode_count += 1
        self._all_results.append(result)

        return result

    def _run_episode_with_pe(
        self, env: EnvAdapter, max_steps: int | None = None,
    ) -> PureDafEpisodeResult:
        """Run episode, recording PE per action for curiosity explorer."""
        if max_steps is None:
            max_steps = self.agent.config.max_episode_steps

        self.agent._episode_rewards.clear()
        self.agent._episode_pes.clear()
        obs = env.reset()

        total_reward = 0.0
        steps = 0
        unique_states: set[int] = set()

        for _ in range(max_steps):
            # Use PE-biased action selection when exploring
            if random.random() < self.epsilon.value:
                action = self.pe_explorer.select_with_bonus()
                action = min(action, self.agent._effective_n_actions - 1)
                # Still do causal learning
                self.agent._causal.before_action(
                    action, self.agent._current_sks,
                    self.agent._current_embedding,
                )
                # Inject motor currents
                motor_currents = self.agent._agent.motor.encode(
                    action, device=self.agent.engine.device,
                )
                self.agent._agent.pipeline.inject_motor_currents(motor_currents)
            else:
                action = self.agent.step(obs)

            obs, reward, terminated, truncated, info = env.step(action)
            pe = self.agent.observe_result(obs, reward)

            # Track PE per action for curiosity
            self.pe_explorer.record(action, pe)

            # Track unique states
            state_hash = hash(obs.tobytes())
            unique_states.add(state_hash)

            total_reward += reward
            steps += 1

            if terminated or truncated:
                break

        result = PureDafEpisodeResult(
            success=total_reward > 0,
            reward=total_reward,
            steps=steps,
            sks_count=len(self.agent._current_sks),
            mean_pe=sum(self.agent._episode_pes) / max(len(self.agent._episode_pes), 1),
            causal_stats=self.agent._causal.stats,
            nav_stats={
                **self.agent._navigator.stats,
                "unique_states": len(unique_states),
                "epsilon": self.epsilon.value,
            },
        )
        return result

    def train(
        self,
        env_factory: Callable[[str], EnvAdapter],
        total_episodes: int = 100,
        max_steps: int | None = None,
    ) -> list[PureDafEpisodeResult]:
        """Full curriculum training loop.

        Args:
            env_factory: callable(env_name) → EnvAdapter
            total_episodes: total episodes across all curriculum stages
            max_steps: max steps per episode

        Returns:
            list of all episode results
        """
        results: list[PureDafEpisodeResult] = []

        for ep in range(total_episodes):
            stage = self.scheduler.current_stage
            env = env_factory(stage.env_name)
            result = self.train_episode(env, max_steps)
            results.append(result)

            # Log progress every 10 episodes
            if (ep + 1) % 10 == 0:
                recent = results[-10:]
                rate = sum(1 for r in recent if r.success) / len(recent)
                print(
                    f"  Ep {ep+1}/{total_episodes} "
                    f"stage={stage.env_name} "
                    f"rate={rate:.2f} "
                    f"eps={self.epsilon.value:.3f}"
                )

        return results

    @property
    def stats(self) -> dict:
        return {
            "episode_count": self._episode_count,
            "curriculum": self.scheduler.stats,
            "epsilon": self.epsilon.value,
        }
