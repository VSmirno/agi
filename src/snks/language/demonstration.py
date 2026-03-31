"""Demonstration recording for few-shot learning (Stage 30).

Records agent trajectories as sequences of (sks_before, action, sks_after)
transitions for later replay into CausalWorldModel.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DemoStep:
    """Single observed transition."""

    sks_before: frozenset[int]
    action: int
    sks_after: frozenset[int]


@dataclass
class Demonstration:
    """Complete recorded trajectory."""

    steps: list[DemoStep] = field(default_factory=list)
    goal_instruction: str = ""
    success: bool = False

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    def unique_actions(self) -> set[int]:
        return {s.action for s in self.steps}

    def unique_sks(self) -> set[int]:
        """All SKS IDs observed across all steps."""
        result: set[int] = set()
        for s in self.steps:
            result.update(s.sks_before)
            result.update(s.sks_after)
        return result


class DemonstrationRecorder:
    """Records an agent's episode as a Demonstration.

    Wraps env.step() to capture sks transitions via GridPerception.
    Usage:
        recorder = DemonstrationRecorder(env, perception)
        recorder.start(instruction)
        # ... agent runs episode using env ...
        demo = recorder.stop(success=True)
    """

    def __init__(self, env, perception) -> None:
        self._env = env
        self._perception = perception
        self._recording = False
        self._steps: list[DemoStep] = []
        self._instruction: str = ""
        self._orig_step = None
        self._last_sks: frozenset[int] | None = None

    def start(self, instruction: str = "") -> None:
        """Begin recording. Wraps env.step()."""
        self._instruction = instruction
        self._steps = []
        self._recording = True
        self._orig_step = self._env.step

        # Snapshot initial state.
        self._last_sks = self._perceive()

        def _recording_step(action):
            sks_before = self._last_sks
            result = self._orig_step(action)
            sks_after = self._perceive()
            self._steps.append(DemoStep(
                sks_before=sks_before,
                action=action,
                sks_after=sks_after,
            ))
            self._last_sks = sks_after
            return result

        self._env.step = _recording_step

    def stop(self, success: bool = False) -> Demonstration:
        """Stop recording and return the Demonstration."""
        if self._orig_step is not None:
            self._env.step = self._orig_step
            self._orig_step = None
        self._recording = False
        return Demonstration(
            steps=list(self._steps),
            goal_instruction=self._instruction,
            success=success,
        )

    def _perceive(self) -> frozenset[int]:
        """Get current SKS set from perception."""
        uw = self._env.unwrapped
        carrying = getattr(uw, "carrying", None)
        sks = self._perception.perceive(uw.grid, uw.agent_pos, uw.agent_dir, carrying)
        return frozenset(sks)
