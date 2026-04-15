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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from snks.agent.vector_sim import VectorTrajectory


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
class CuriosityStimulus(Stimulus):
    """Defined for Stage 87 (death-relevant curiosity weighting). Unused in Stage 85.

    Scores trajectory by average prediction surprise — low confidence predictions
    indicate unexplored state space, which is intrinsically rewarding.
    Not wired into StimuliLayer default; curiosity is handled via Goal("explore").progress().
    """
    weight: float = 0.1

    def evaluate(self, trajectory: "VectorTrajectory") -> float:
        return self.weight * trajectory.avg_surprise()


@dataclass
class StimuliLayer:
    """Aggregates multiple stimuli into a single score.

    evaluate(trajectory) = sum of all stimulus.evaluate(trajectory).
    Passed to score_trajectory() in vector_sim.py.
    """

    stimuli: list[Stimulus] = field(default_factory=list)

    def evaluate(self, trajectory: "VectorTrajectory") -> float:
        return sum(s.evaluate(trajectory) for s in self.stimuli)
