"""Stage 84: StimuliLayer — Category 4 from IDEOLOGY v2.

Stimuli are the scoring policy for the MPC planner. They live outside
the mechanism layer (Category 2) and are passed into score_trajectory
as a configurable object.

Usage:
    layer = StimuliLayer([SurvivalAversion(), HomeostasisStimulus()])
    score = layer.evaluate(trajectory)

Stage 85 adds CuriosityStimulus here — zero changes to vector_sim.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from snks.agent.death_hypothesis import DeathHypothesis
    from snks.agent.vector_sim import VectorTrajectory
    from snks.agent.vector_world_model import VectorWorldModel


@dataclass
class Stimulus:
    """Base class for stimuli. evaluate(trajectory) -> float score contribution."""

    def evaluate(self, trajectory: "VectorTrajectory") -> float:
        raise NotImplementedError


@dataclass
class SurvivalAversion(Stimulus):
    """Large penalty if trajectory terminated (agent died).

    Dominates all other stimuli — death is the worst outcome.
    """

    weight: float = 1000.0

    def evaluate(self, trajectory: "VectorTrajectory") -> float:
        return -self.weight if trajectory.terminated else 0.0


@dataclass
class HomeostasisStimulus(Stimulus):
    """Penalise vital deficits below per-vital thresholds.

    Returns -weight * sum(max(0, threshold[v] - body[v]) for v in vital_vars).
    Zero when all vitals are above their thresholds.

    thresholds: per-vital floor. Defaults to {} (no thresholds active).
    Backwards-compatible: callers that don't pass thresholds get zero penalty
    for any vital level, which is equivalent to the old behaviour when vitals
    are all positive.
    """

    vital_vars: list[str] = field(
        default_factory=lambda: ["health", "food", "drink", "energy"]
    )
    weight: float = 1.0
    thresholds: dict = field(default_factory=dict)

    def evaluate(self, trajectory: "VectorTrajectory") -> float:
        final = trajectory.final_state
        if not final:
            return 0.0
        deficit = sum(
            max(0.0, self.thresholds.get(v, 0.0) - final.body.get(v, 0.0))
            for v in self.vital_vars
        )
        return -self.weight * deficit


@dataclass
class VitalDeltaStimulus(Stimulus):
    """Penalize negative trajectory deltas on selected vital variables.

    Unlike HomeostasisStimulus, this reacts to short-horizon losses even when
    the final value stays above any threshold. This is the right layer for
    projected threat aversion: if a plan causes health to drop from 9 to 6, it
    should score worse than a plan that preserves health, even if both survive.
    """

    vital_vars: list[str] = field(
        default_factory=lambda: ["health", "food", "drink", "energy"]
    )
    weight: float = 1.0

    def evaluate(self, trajectory: "VectorTrajectory") -> float:
        loss = 0.0
        for vital in self.vital_vars:
            delta = trajectory.vital_delta(vital)
            if delta < 0:
                loss += -delta
        return -self.weight * loss


@dataclass
class CuriosityStimulus(Stimulus):
    """Stage 87: Death-relevant curiosity weighting.

    Scores trajectory by prediction surprise weighted by death relevance:
        U_curiosity(s) = weight * avg_surprise(s) * death_relevance(s)

    death_relevance comes from an active DeathHypothesis — trajectories that
    expose vital states near the death-correlated threshold score higher.
    Without a hypothesis, relevance = 1.0 (pure surprise signal).
    """

    weight: float = 0.1
    hypothesis: "DeathHypothesis | None" = None

    def evaluate(self, trajectory: "VectorTrajectory") -> float:
        surprise = trajectory.avg_surprise()
        relevance = (
            self.hypothesis.death_relevance(trajectory)
            if self.hypothesis is not None
            else 1.0
        )
        return self.weight * surprise * relevance


def resolve_outcome_pair(
    plan_steps: list,
    near_concept: str | None,
) -> tuple[str, str] | None:
    """Resolve the (concept, action) pair the outcome role is keyed on.

    Shared by `OutcomeStimulus` (read side) and the lifecycle recorder
    (write side) so the planner and the learner agree on the address.
    Returns None when no pair can be resolved (e.g. motion plan without
    a known facing concept).
    """
    if not plan_steps:
        if near_concept is None:
            return None
        return (str(near_concept), "noop")
    first = plan_steps[0]
    action = first.action
    target = first.target
    if action in ("do", "make", "place"):
        return (str(target), action)
    if action == "sleep":
        return ("self", "sleep")
    if action.startswith("move_"):
        if near_concept is None:
            return None
        return (str(near_concept), action)
    return None


@dataclass
class OutcomeStimulus(Stimulus):
    """Cross-episode advisory signal from the world model's outcome role.

    For each candidate trajectory, resolve the `(concept, action)` pair the
    plan represents and call `model.predict_outcome(concept, action)`. When
    a confident match is returned, the decoded outcome contributes a
    composite signal:

        weight * confidence * (
            survived_bonus if survived else -died_penalty
            - damage_unit_penalty * damage_h
            - death_cause_penalty if died_to is not None
        )

    Concept resolution per plan shape (matches the spec §3):
      - `do` plan       : (plan.steps[0].target, "do")
      - `make`/`place`  : (plan.steps[0].target, plan.steps[0].action)
      - `sleep`         : ("self", "sleep")
      - motion plan     : (near_concept_provider(), plan.steps[0].action)
      - empty/baseline  : (near_concept_provider(), "noop")

    `near_concept_provider` is a callable set per planning step by
    `run_vector_mpc_episode`; it returns the current `vf.near_concept`
    string so motion / baseline queries are conditioned on what the agent
    is currently facing.
    """

    model: "VectorWorldModel | None" = None
    weight: float = 1.0
    survived_bonus: float = 1.0
    died_penalty: float = 3.0
    damage_unit_penalty: float = 0.25
    death_cause_penalty: float = 5.0
    near_concept_provider: "Callable[[], str | None] | None" = None

    def _resolve_pair(
        self, trajectory: "VectorTrajectory",
    ) -> tuple[str, str] | None:
        near = self.near_concept_provider() if self.near_concept_provider else None
        return resolve_outcome_pair(trajectory.plan.steps, near)

    def evaluate(self, trajectory: "VectorTrajectory") -> float:
        if self.model is None:
            return 0.0
        pair = self._resolve_pair(trajectory)
        if pair is None:
            return 0.0
        decoded, confidence = self.model.predict_outcome(pair[0], pair[1])
        if decoded is None:
            return 0.0
        # Only NEGATIVE recall contributes. Survived outcomes are the default
        # expectation — boosting them would systematically pull the planner
        # away from candidate plans that the agent has not yet tried
        # (crafting plans have no recall → 0 boost → relatively penalised
        # vs known-safe motion/do plans). The stimulus is a "death
        # warning", not a value function. This matches the user's framing:
        # "если в этом контексте я уже умирал → не делай так".
        if decoded.get("survived_h", True):
            return 0.0
        signal = -self.died_penalty - self.damage_unit_penalty * float(decoded.get("damage_h", 0))
        if decoded.get("died_to") not in (None, "none"):
            signal -= self.death_cause_penalty
        return self.weight * confidence * signal


@dataclass
class StimuliLayer:
    """Aggregates multiple stimuli into a single score.

    evaluate(trajectory) = sum of all stimulus.evaluate(trajectory).
    Passed to score_trajectory() in vector_sim.py.
    """

    stimuli: list[Stimulus] = field(default_factory=list)

    def evaluate(self, trajectory: "VectorTrajectory") -> float:
        return sum(s.evaluate(trajectory) for s in self.stimuli)
