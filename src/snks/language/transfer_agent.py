"""TransferAgent: GoalAgent wrapper with transfer metrics (Stage 26)."""

from __future__ import annotations

from dataclasses import dataclass, field

from snks.agent.causal_model import CausalWorldModel
from snks.daf.types import CausalAgentConfig
from snks.language.goal_agent import GoalAgent, EpisodeResult
from snks.language.grounding_map import GroundingMap


@dataclass
class TransferResult:
    """Result of one TransferAgent episode."""

    success: bool = False
    steps_taken: int = 0
    explored: bool = False
    links_reused: int = 0
    links_new: int = 0


@dataclass
class TransferStats:
    """Aggregate stats across episodes."""

    episodes: int = 0
    successes: int = 0
    total_steps: int = 0
    total_links_reused: int = 0
    total_links_new: int = 0
    exploration_episodes: int = 0

    @property
    def success_rate(self) -> float:
        return self.successes / max(self.episodes, 1)

    @property
    def mean_steps(self) -> float:
        return self.total_steps / max(self.episodes, 1)


class TransferAgent:
    """GoalAgent wrapper that tracks transfer learning metrics."""

    def __init__(
        self,
        causal_model: CausalWorldModel | None = None,
    ) -> None:
        if causal_model is None:
            config = CausalAgentConfig(causal_min_observations=1)
            causal_model = CausalWorldModel(config)
        self._causal_model = causal_model
        self._pre_loaded_links = causal_model.n_links
        self._stats = TransferStats()

    @property
    def causal_model(self) -> CausalWorldModel:
        return self._causal_model

    @property
    def pre_loaded_links(self) -> int:
        return self._pre_loaded_links

    def run_episode(
        self,
        env,
        instruction: str,
        max_steps: int = 300,
    ) -> TransferResult:
        """Run one episode and return transfer-aware result."""
        links_before = self._causal_model.n_links

        # New GroundingMap per episode (environment may differ)
        gmap = GroundingMap()
        agent = GoalAgent(
            env, grounding_map=gmap, causal_model=self._causal_model,
        )

        episode_result: EpisodeResult = agent.run_episode(instruction, max_steps)

        links_after = self._causal_model.n_links
        links_new = max(0, links_after - links_before)

        # Estimate reused links: if agent didn't explore and succeeded,
        # it used pre-loaded knowledge
        links_reused = 0
        if not episode_result.explored and episode_result.subgoals_identified:
            links_reused = len(episode_result.subgoals_identified)

        result = TransferResult(
            success=episode_result.success,
            steps_taken=episode_result.steps_taken,
            explored=episode_result.explored,
            links_reused=links_reused,
            links_new=links_new,
        )

        self._record_result(result)
        return result

    def _record_result(self, result: TransferResult) -> None:
        self._stats.episodes += 1
        if result.success:
            self._stats.successes += 1
        self._stats.total_steps += result.steps_taken
        self._stats.total_links_reused += result.links_reused
        self._stats.total_links_new += result.links_new
        if result.explored:
            self._stats.exploration_episodes += 1

    def get_stats(self) -> TransferStats:
        return self._stats
