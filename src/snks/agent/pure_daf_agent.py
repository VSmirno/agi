"""PureDafAgent: agent that uses ONLY the DAF pipeline for perception, planning, and learning.

NO scaffolding:
- No GridPerception (no direct grid reading)
- No GridNavigator (no BFS)
- No BlockingAnalyzer (no grid scanning)
- No hardcoded SKS IDs (50-58)
- No dict-based CausalWorldModel

ONLY DAF:
- VisualEncoder → Gabor → SDR → oscillator currents
- DafEngine → 50K FHN oscillators → STDP learning
- Coherence → SKS clusters (learned, not hardcoded)
- HAC embeddings → state similarity
- Reward-modulated STDP → causal learning
- Mental simulation → action selection

This is the core SNKS paradigm: computation = evolution of a dynamical system.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from snks.agent.agent import CausalAgent, _perceptual_hash
from snks.agent.attractor_navigator import AttractorNavigator
from snks.agent.daf_causal_model import DafCausalModel
from snks.agent.motor import MotorEncoder
from snks.daf.types import CausalAgentConfig
from snks.env.adapter import EnvAdapter
from snks.env.obs_adapter import ObsAdapter


@dataclass
class PureDafConfig:
    """Configuration for PureDafAgent."""
    causal: CausalAgentConfig = field(default_factory=CausalAgentConfig)
    # Navigation
    exploration_epsilon: float = 0.3
    min_similarity: float = -0.5
    n_sim_steps: int = 10
    # Causal learning
    reward_scale: float = 2.0
    trace_length: int = 5
    negative_scale: float = 0.5
    # Episode
    max_episode_steps: int = 200
    n_actions: int = 5
    # Stage 40: Hebbian encoder
    use_hebbian: bool = False


@dataclass
class PureDafEpisodeResult:
    """Result of one PureDafAgent episode."""
    success: bool = False
    reward: float = 0.0
    steps: int = 0
    sks_count: int = 0
    mean_pe: float = 0.0
    causal_stats: dict = field(default_factory=dict)
    nav_stats: dict = field(default_factory=dict)


class PureDafAgent:
    """Agent using ONLY DAF pipeline — no environment-specific scaffolding.

    Architecture:
        PureDafAgent
        ├── CausalAgent (Pipeline → DafEngine → oscillators + STDP + SKS)
        ├── DafCausalModel (reward-modulated STDP)
        ├── AttractorNavigator (mental simulation → action selection)
        └── ObsAdapter (image preprocessing)

    Episode cycle:
        1. obs → ObsAdapter → grayscale image
        2. image → Pipeline.perception_cycle() → SKS clusters + embeddings
        3. AttractorNavigator.select_action(current_embed, goal_embed) → action
        4. DafCausalModel.before_action(action, sks)
        5. env.step(action) → (obs', reward, done)
        6. DafCausalModel.after_action(reward)
        7. Loop until done
    """

    def __init__(self, config: PureDafConfig | None = None) -> None:
        if config is None:
            config = PureDafConfig()
        self.config = config

        # Stage 40: enable Hebbian encoder if requested
        if config.use_hebbian:
            config.causal.pipeline.encoder.hebbian = True

        # Core DAF agent (owns Pipeline, DafEngine, STDP)
        self._agent = CausalAgent(config.causal)
        self._obs_adapter = ObsAdapter(target_size=config.causal.pipeline.encoder.image_size)

        # Reward-modulated causal learning
        self._causal = DafCausalModel(
            engine=self._agent.pipeline.engine,
            reward_scale=config.reward_scale,
            trace_length=config.trace_length,
            negative_scale=config.negative_scale,
        )

        # Attractor-based navigation
        self._navigator = AttractorNavigator(
            daf_causal_model=self._causal,
            motor_encoder=self._agent.motor,
            n_sim_steps=config.n_sim_steps,
            min_similarity=config.min_similarity,
            exploration_epsilon=config.exploration_epsilon,
        )

        # State tracking
        self._goal_embedding: torch.Tensor | None = None
        self._current_embedding: torch.Tensor | None = None
        self._current_sks: set[int] = set()
        self._episode_rewards: list[float] = []
        self._episode_pes: list[float] = []

    @property
    def pipeline(self):
        """Access underlying Pipeline for inspection."""
        return self._agent.pipeline

    @property
    def engine(self):
        """Access underlying DafEngine."""
        return self._agent.pipeline.engine

    def set_goal_from_obs(self, goal_obs: np.ndarray) -> None:
        """Set goal state from observation image.

        Encodes goal observation through DAF to get goal embedding.
        """
        image = self._obs_adapter.convert(goal_obs)
        result = self._agent.pipeline.perception_cycle(image)
        if result.winner_embedding is not None:
            self._goal_embedding = result.winner_embedding.clone()
        else:
            # Use firing rate pattern as fallback embedding
            self._goal_embedding = self._causal._extract_embedding(
                self.engine.step(10)
            )

    def step(self, obs: np.ndarray) -> int:
        """One agent step: perceive → decide → prepare to learn.

        Args:
            obs: RGB observation (H, W, 3) uint8

        Returns:
            action integer
        """
        # 1. Perception via DAF pipeline
        image = self._obs_adapter.convert(obs)
        result = self._agent.pipeline.perception_cycle(image)
        self._current_sks = set(result.sks_clusters.keys())
        self._current_sks |= _perceptual_hash(image)

        # Get current embedding
        if result.winner_embedding is not None:
            self._current_embedding = result.winner_embedding
        else:
            self._current_embedding = None

        # 2. Action selection via attractor navigator
        if self._current_embedding is not None and self._goal_embedding is not None:
            action = self._navigator.select_action(
                self._current_embedding,
                self._goal_embedding,
                self.config.n_actions,
            )
        else:
            # Fallback: use intrinsic motivation from CausalAgent
            action = self._agent.motivation.select_action(
                self._current_sks,
                self._agent.causal_model,
                self.config.n_actions,
            )

        # 3. Prepare causal learning
        self._causal.before_action(action, self._current_sks, self._current_embedding)

        # 4. Inject motor currents
        motor_currents = self._agent.motor.encode(action, device=self.engine.device)
        self._agent.pipeline.inject_motor_currents(motor_currents)

        return action

    def observe_result(self, obs: np.ndarray, reward: float) -> float:
        """Observe action result and learn.

        Args:
            obs: post-action observation (H, W, 3) uint8
            reward: environment reward signal

        Returns:
            prediction_error (float)
        """
        # 1. Perceive new state
        image = self._obs_adapter.convert(obs)
        result = self._agent.pipeline.perception_cycle(image)
        post_sks = set(result.sks_clusters.keys())
        post_sks |= _perceptual_hash(image)

        # 2. Reward-modulated STDP learning
        self._causal.after_action(reward)

        # 3. Compute prediction error (for monitoring)
        pe = len(post_sks.symmetric_difference(self._current_sks)) / max(
            len(post_sks | self._current_sks), 1
        )

        self._episode_rewards.append(reward)
        self._episode_pes.append(pe)

        # Update current state
        self._current_sks = post_sks
        if result.winner_embedding is not None:
            self._current_embedding = result.winner_embedding

        return pe

    def run_episode(self, env: EnvAdapter, max_steps: int | None = None) -> PureDafEpisodeResult:
        """Run one complete episode.

        Args:
            env: environment adapter (no grid access!)
            max_steps: override config max_episode_steps

        Returns:
            PureDafEpisodeResult with metrics
        """
        if max_steps is None:
            max_steps = self.config.max_episode_steps

        # Reset
        self._episode_rewards.clear()
        self._episode_pes.clear()
        obs = env.reset()

        total_reward = 0.0
        steps = 0

        for _ in range(max_steps):
            action = self.step(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            pe = self.observe_result(obs, reward)
            total_reward += reward
            steps += 1

            if terminated or truncated:
                break

        return PureDafEpisodeResult(
            success=total_reward > 0,
            reward=total_reward,
            steps=steps,
            sks_count=len(self._current_sks),
            mean_pe=sum(self._episode_pes) / max(len(self._episode_pes), 1),
            causal_stats=self._causal.stats,
            nav_stats=self._navigator.stats,
        )

    def run_training(
        self,
        env: EnvAdapter,
        n_episodes: int = 100,
        max_steps: int | None = None,
    ) -> list[PureDafEpisodeResult]:
        """Run multiple episodes with learning.

        Args:
            env: environment adapter
            n_episodes: number of episodes
            max_steps: max steps per episode

        Returns:
            list of episode results
        """
        results: list[PureDafEpisodeResult] = []
        for ep in range(n_episodes):
            result = self.run_episode(env, max_steps)
            results.append(result)
        return results
