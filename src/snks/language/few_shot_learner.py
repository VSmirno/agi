"""FewShotLearner: extract causal model + skills from demonstrations (Stage 30).

Takes 1-N recorded demonstrations and produces a CausalWorldModel and
SkillLibrary sufficient for the agent to solve similar tasks.
"""

from __future__ import annotations

from snks.agent.causal_model import CausalWorldModel
from snks.daf.types import CausalAgentConfig
from snks.language.demonstration import Demonstration
from snks.language.skill_library import SkillLibrary


class FewShotLearner:
    """Learns causal model and skills from observed demonstrations."""

    def __init__(self, min_observations: int = 1) -> None:
        self._min_obs = min_observations

    def learn_from_demonstrations(
        self,
        demos: list[Demonstration],
        existing_model: CausalWorldModel | None = None,
        existing_library: SkillLibrary | None = None,
    ) -> tuple[CausalWorldModel, SkillLibrary]:
        """Extract causal knowledge and skills from demonstrations.

        Args:
            demos: List of recorded demonstrations.
            existing_model: Optional existing model to merge into.
            existing_library: Optional existing library to extend.

        Returns:
            (CausalWorldModel, SkillLibrary) ready for agent use.
        """
        if existing_model is not None:
            model = existing_model
        else:
            config = CausalAgentConfig(causal_min_observations=self._min_obs)
            model = CausalWorldModel(config)

        # Feed all demo steps into causal model.
        for demo in demos:
            if not demo.success:
                continue  # Only learn from successful demonstrations
            for step in demo.steps:
                model.observe_transition(
                    set(step.sks_before),
                    step.action,
                    set(step.sks_after),
                )

        # Extract skills from the learned causal model.
        library = existing_library or SkillLibrary()
        library.extract_from_causal_model(model)
        library.compose_skills()

        return model, library
