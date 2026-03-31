"""AutonomousAgent: curriculum-driven learning on large grids (Stage 36).

Combines GoalAgent (backward chaining + causal learning) with
CurriculumManager (progressive difficulty) and knowledge transfer.

Key insight: GoalAgent works on 5x5 because causal model learns fast
on small state space. AutonomousAgent bootstraps from small grids and
transfers causal knowledge to larger ones.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import gymnasium as gym

from snks.agent.causal_model import CausalWorldModel
from snks.daf.types import CausalAgentConfig
from snks.language.curriculum_manager import CurriculumManager, LevelStats
from snks.language.goal_agent import GoalAgent, EpisodeResult
from snks.language.grounding_map import GroundingMap


@dataclass
class CurriculumResult:
    """Result of a full curriculum run."""
    levels_completed: int = 0
    total_episodes: int = 0
    total_steps: int = 0
    elapsed_seconds: float = 0.0
    level_stats: dict[int, dict] = field(default_factory=dict)
    final_grid_size: int = 5
    causal_links: int = 0

    @property
    def final_success_rate(self) -> float:
        """Success rate on the largest grid reached."""
        if self.final_grid_size in self.level_stats:
            s = self.level_stats[self.final_grid_size]
            return s.get("success_rate", 0.0)
        return 0.0


DOORKEY_ENV_MAP = {
    5: "MiniGrid-DoorKey-5x5-v0",
    6: "MiniGrid-DoorKey-6x6-v0",
    8: "MiniGrid-DoorKey-8x8-v0",
    16: "MiniGrid-DoorKey-16x16-v0",
}

# Multi-room environments (Stage 37)
MULTIROOM_ENV_MAP = {
    "N2": "MiniGrid-MultiRoom-N2-S4-v0",
    "N4": "MiniGrid-MultiRoom-N4-S5-v0",
    "N6": "MiniGrid-MultiRoom-N6-v0",
    "unlock": "MiniGrid-UnlockPickup-v0",
    "blocked": "MiniGrid-BlockedUnlockPickup-v0",
}


def _env_name(grid_size: int, env_type: str = "doorkey") -> str:
    """Get MiniGrid env name for a grid size or env type."""
    if env_type != "doorkey" and env_type in MULTIROOM_ENV_MAP:
        return MULTIROOM_ENV_MAP[env_type]
    if grid_size in DOORKEY_ENV_MAP:
        return DOORKEY_ENV_MAP[grid_size]
    if grid_size <= 5:
        return DOORKEY_ENV_MAP[5]
    elif grid_size <= 6:
        return DOORKEY_ENV_MAP[6]
    elif grid_size <= 8:
        return DOORKEY_ENV_MAP[8]
    else:
        return DOORKEY_ENV_MAP[16]


class AutonomousAgent:
    """Curriculum-driven autonomous learning agent.

    Architecture:
        AutonomousAgent
        ├── CurriculumManager — controls difficulty progression
        ├── CausalWorldModel — shared across all grid sizes (transfer)
        └── GoalAgent — episode execution (created per episode)

    The key mechanism is knowledge transfer: causal links learned on
    5x5 (key→pickup, door→toggle) apply directly to 8x8 and 12x12
    because GridPerception produces the same SKS predicates regardless
    of grid size.
    """

    def __init__(
        self,
        levels: list[int] | None = None,
        advance_threshold: float = 0.5,
        causal_model: CausalWorldModel | None = None,
    ) -> None:
        self.curriculum = CurriculumManager(
            levels=levels or [5, 6, 8, 16],
            advance_threshold=advance_threshold,
        )
        if causal_model is None:
            config = CausalAgentConfig(causal_min_observations=1)
            causal_model = CausalWorldModel(config)
        self._causal_model = causal_model
        self._episode_log: list[dict] = []

    @property
    def causal_model(self) -> CausalWorldModel:
        return self._causal_model

    def run_episode(self, seed: int | None = None) -> EpisodeResult:
        """Run one episode at current curriculum level."""
        grid_size = self.curriculum.current_grid_size
        max_steps = self.curriculum.max_steps_for_level()
        env_name = _env_name(grid_size)

        env = gym.make(env_name)
        obs, _ = env.reset(seed=seed)

        gmap = GroundingMap()
        agent = GoalAgent(env, grounding_map=gmap, causal_model=self._causal_model)

        instruction = obs.get("mission", "go to the goal") if isinstance(obs, dict) else "go to the goal"
        result = agent.run_episode(instruction, max_steps=max_steps)

        env.close()

        self.curriculum.record_episode(result.success, result.steps_taken)
        self._episode_log.append({
            "grid_size": grid_size,
            "success": result.success,
            "steps": result.steps_taken,
            "explored": result.explored,
            "subgoals": result.subgoals_identified,
            "causal_links": self._causal_model.n_links,
        })

        return result

    def run_curriculum(
        self,
        total_episodes: int = 400,
        callback: callable | None = None,
    ) -> CurriculumResult:
        """Run full curriculum from smallest to largest grid.

        Args:
            total_episodes: Maximum total episodes across all levels.
            callback: Optional fn(episode_idx, grid_size, result) called each episode.

        Returns:
            CurriculumResult with per-level statistics.
        """
        t0 = time.perf_counter()
        ep_idx = 0

        while ep_idx < total_episodes:
            grid_size = self.curriculum.current_grid_size
            budget = self.curriculum.episodes_budget_for_level()

            # Run episodes at this level
            level_eps = 0
            while level_eps < budget and ep_idx < total_episodes:
                result = self.run_episode(seed=ep_idx)
                if callback:
                    callback(ep_idx, grid_size, result)
                ep_idx += 1
                level_eps += 1

                # Check advancement mid-level
                if self.curriculum.should_advance():
                    break

            # Try to advance
            if self.curriculum.should_advance():
                self.curriculum.advance()
            elif self.curriculum.is_final_level:
                # On final level, run remaining budget
                remaining = budget - level_eps
                for _ in range(remaining):
                    if ep_idx >= total_episodes:
                        break
                    result = self.run_episode(seed=ep_idx)
                    if callback:
                        callback(ep_idx, grid_size, result)
                    ep_idx += 1
                break  # Done with final level
            else:
                # Couldn't advance — spend more episodes
                extra = min(50, total_episodes - ep_idx)
                for _ in range(extra):
                    if ep_idx >= total_episodes:
                        break
                    result = self.run_episode(seed=ep_idx)
                    if callback:
                        callback(ep_idx, grid_size, result)
                    ep_idx += 1
                if self.curriculum.should_advance():
                    self.curriculum.advance()
                else:
                    break  # Stuck — stop

        elapsed = time.perf_counter() - t0

        # Compile results
        cr = CurriculumResult(
            levels_completed=self.curriculum.current_level_idx + 1,
            total_episodes=ep_idx,
            total_steps=sum(e["steps"] for e in self._episode_log),
            elapsed_seconds=round(elapsed, 1),
            final_grid_size=self.curriculum.current_grid_size,
            causal_links=self._causal_model.n_links,
        )

        for size in self.curriculum.levels:
            stats = self.curriculum.get_stats(size)
            cr.level_stats[size] = {
                "episodes": stats.episodes,
                "successes": stats.successes,
                "success_rate": round(stats.success_rate, 3),
                "total_steps": stats.total_steps,
            }

        return cr
