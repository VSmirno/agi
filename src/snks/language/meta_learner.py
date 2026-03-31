"""MetaLearner: adaptive strategy selection for SNKS agents (Stage 32).

Observes task characteristics (demos, skills, coverage, prediction error)
and selects the optimal learning strategy. Adapts hyperparameters after
each episode based on outcome.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TaskProfile:
    """Characterizes the current task/agent state for strategy selection."""

    has_demos: bool = False
    n_demos: int = 0
    known_skills: int = 0
    causal_links: int = 0
    state_coverage: float = 0.0
    mean_prediction_error: float = 1.0
    episodes_completed: int = 0
    last_success: bool = False


@dataclass
class StrategyConfig:
    """Output of MetaLearner: which strategy and params to use."""

    strategy: str  # "curiosity" | "skill" | "few_shot" | "explore"
    curiosity_epsilon: float = 0.2
    analogy_threshold: float = 0.7
    exploration_budget: int = 60
    use_analogy: bool = True
    reason: str = ""


@dataclass
class EpisodeResult:
    """Outcome of a single episode for adaptation."""

    success: bool
    steps: int
    skills_used: int = 0
    new_states_discovered: int = 0
    prediction_error: float = 0.5


class MetaLearner:
    """Selects and adapts learning strategies based on task characteristics.

    Uses rule-based strategy selection with adaptive threshold adjustment.
    No backpropagation — purely metric-driven decisions.
    """

    def __init__(self, adaptation_rate: float = 0.1) -> None:
        self._adaptation_rate = adaptation_rate
        self._epsilon = 0.2
        self._analogy_threshold = 0.7
        self._exploration_budget = 60
        self._history: list[tuple[StrategyConfig, EpisodeResult]] = []

    def select_strategy(self, profile: TaskProfile) -> StrategyConfig:
        """Select optimal strategy based on task characteristics."""
        # Rule 1: If demos available and no skills yet → bootstrap via few-shot
        if profile.has_demos and profile.known_skills == 0:
            return StrategyConfig(
                strategy="few_shot",
                curiosity_epsilon=self._epsilon,
                analogy_threshold=self._analogy_threshold,
                exploration_budget=self._exploration_budget,
                use_analogy=False,
                reason="demos available, no skills — bootstrap via few-shot",
            )

        # Rule 2: If good knowledge base → exploit skills
        if profile.known_skills >= 2 and profile.causal_links >= 5:
            return StrategyConfig(
                strategy="skill",
                curiosity_epsilon=max(0.05, self._epsilon - 0.05),
                analogy_threshold=self._analogy_threshold,
                exploration_budget=self._exploration_budget,
                use_analogy=True,
                reason=f"skills={profile.known_skills}, links={profile.causal_links} — exploit",
            )

        # Rule 3: If low coverage → explore with curiosity
        if profile.state_coverage < 0.3:
            return StrategyConfig(
                strategy="curiosity",
                curiosity_epsilon=max(self._epsilon, 0.3),
                analogy_threshold=self._analogy_threshold,
                exploration_budget=self._exploration_budget,
                use_analogy=False,
                reason=f"coverage={profile.state_coverage:.2f} < 0.3 — explore",
            )

        # Rule 4: Moderate knowledge, moderate coverage → balanced exploration
        if profile.mean_prediction_error > 0.5:
            return StrategyConfig(
                strategy="curiosity",
                curiosity_epsilon=self._epsilon,
                analogy_threshold=self._analogy_threshold,
                exploration_budget=self._exploration_budget,
                use_analogy=profile.known_skills > 0,
                reason=f"pred_error={profile.mean_prediction_error:.2f} > 0.5 — more exploration",
            )

        # Default: balanced explore
        return StrategyConfig(
            strategy="explore",
            curiosity_epsilon=self._epsilon,
            analogy_threshold=self._analogy_threshold,
            exploration_budget=self._exploration_budget,
            use_analogy=profile.known_skills > 0,
            reason="default fallback — balanced exploration",
        )

    def adapt(self, profile: TaskProfile, result: EpisodeResult) -> StrategyConfig:
        """Adapt strategy after observing episode outcome."""
        prev_config = self.select_strategy(profile)
        self._history.append((prev_config, result))

        # Adaptation rules
        if not result.success and prev_config.strategy == "skill":
            # Skills didn't work → increase exploration
            self._epsilon = min(0.5, self._epsilon + self._adaptation_rate)
            self._analogy_threshold = max(0.4, self._analogy_threshold - 0.05)

        if result.success and result.steps < self._exploration_budget // 2:
            # Quick success → reduce exploration (exploit more)
            self._epsilon = max(0.05, self._epsilon - self._adaptation_rate)

        if result.new_states_discovered == 0 and profile.state_coverage < 0.5:
            # Stagnation → switch to curiosity
            self._epsilon = min(0.5, self._epsilon + self._adaptation_rate * 2)

        if result.success and result.skills_used > 0:
            # Skills working well → tighten analogy threshold
            self._analogy_threshold = min(0.8, self._analogy_threshold + 0.02)

        # Re-select with adapted parameters
        return self.select_strategy(profile)

    @property
    def current_epsilon(self) -> float:
        return self._epsilon

    @property
    def current_analogy_threshold(self) -> float:
        return self._analogy_threshold

    @property
    def history(self) -> list[tuple[StrategyConfig, EpisodeResult]]:
        return list(self._history)

    def reset(self) -> None:
        """Reset adaptive state."""
        self._epsilon = 0.2
        self._analogy_threshold = 0.7
        self._exploration_budget = 60
        self._history.clear()
