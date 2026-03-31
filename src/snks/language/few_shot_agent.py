"""FewShotAgent: CuriosityAgent that learns from demonstrations (Stage 30).

Before acting, observes 1-N demonstrations and bootstraps its causal model
and skill library. Then uses the standard skill → analogy → backward chaining
→ curiosity pipeline to solve the task.
"""

from __future__ import annotations

from snks.language.curiosity_agent import CuriosityAgent
from snks.language.demonstration import Demonstration
from snks.language.few_shot_learner import FewShotLearner


class FewShotAgent(CuriosityAgent):
    """CuriosityAgent extended with few-shot learning from demonstrations.

    Usage:
        agent = FewShotAgent(env)
        agent.learn_from_demos(demonstrations)
        result = agent.run_episode(instruction)
    """

    def __init__(self, env, **kwargs) -> None:
        super().__init__(env, **kwargs)
        self._few_shot_learner = FewShotLearner(min_observations=1)
        self._n_demos_learned: int = 0

    @property
    def n_demos_learned(self) -> int:
        return self._n_demos_learned

    def learn_from_demos(self, demos: list[Demonstration]) -> int:
        """Learn from demonstrations, updating causal model and skill library.

        Args:
            demos: List of recorded demonstrations.

        Returns:
            Number of new skills extracted.
        """
        old_skills = len(self._library.skills)

        model, library = self._few_shot_learner.learn_from_demonstrations(
            demos,
            existing_model=self._causal_model,
            existing_library=self._library,
        )
        self._causal_model = model
        self._library = library
        self._n_demos_learned += len(demos)

        return len(self._library.skills) - old_skills
