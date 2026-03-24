"""CausalAgent: top-level agent integrating Pipeline with causal learning."""

from __future__ import annotations

import numpy as np
import torch

from snks.agent.causal_model import CausalWorldModel
from snks.agent.motor import MotorEncoder
from snks.agent.motivation import IntrinsicMotivation
from snks.agent.simulation import MentalSimulator
from snks.daf.types import CausalAgentConfig
from snks.env.obs_adapter import ObsAdapter
from snks.pipeline.runner import Pipeline


def _perceptual_hash(image: torch.Tensor, n_bins: int = 8) -> set[int]:
    """Compute rotation-invariant perceptual hash from image.

    Divides image into n_bins×n_bins grid, collects mean intensity per cell,
    then sorts intensities so that rotations of the same scene produce the
    same hash.  Each sorted intensity is quantised into 8 brightness levels
    and mapped to a pseudo-SKS ID in [10000, 10000 + n_bins*n_bins*8).

    Returns set of pseudo-SKS IDs (no collision with real SKS cluster IDs).
    """
    h, w = image.shape[-2:]
    cell_h, cell_w = h // n_bins, w // n_bins

    intensities: list[float] = []
    for i in range(n_bins):
        for j in range(n_bins):
            cell = image[..., i * cell_h:(i + 1) * cell_h, j * cell_w:(j + 1) * cell_w]
            intensities.append(cell.mean().item())

    # Sort → rotation-invariant
    intensities.sort()

    # Quantise into bins and create pseudo-SKS IDs
    ids: set[int] = set()
    offset = 10000
    for idx, val in enumerate(intensities):
        bin_id = min(int(val * 8), 7)  # 8 brightness levels, clamp to [0,7]
        ids.add(offset + idx * 8 + bin_id)
    return ids


class CausalAgent:
    """Top-level agent: perceive → decide → act → learn.

    Integrates Pipeline (Stage 3) with causal learning (Stage 6).
    """

    N_ACTIONS = 5  # turn_left, turn_right, forward, interact, noop

    def __init__(self, config: CausalAgentConfig):
        self.config = config
        self.pipeline = Pipeline(config.pipeline)
        self.obs_adapter = ObsAdapter(target_size=config.pipeline.encoder.image_size)
        self.motor = MotorEncoder(
            n_actions=self.N_ACTIONS,
            num_nodes=config.pipeline.daf.num_nodes,
            sdr_size=config.motor_sdr_size,
            current_strength=config.motor_current_strength,
        )
        self.causal_model = CausalWorldModel(config)
        self.simulator = MentalSimulator(self.causal_model)
        self.motivation = IntrinsicMotivation(config)

        self._pre_sks: set[int] | None = None
        self._last_action: int | None = None
        self._step_count: int = 0

    def step(self, obs: np.ndarray) -> int:
        """Full agent cycle:

        1. obs → grayscale 64×64 (ObsAdapter)
        2. image → Pipeline.perception_cycle() → CycleResult (SKS)
        3. IntrinsicMotivation.select_action() → action
        4. MotorEncoder.encode(action) → inject motor currents into DAF
        5. Return action for environment
        """
        # 1. Convert observation
        image = self.obs_adapter.convert(obs)

        # 2. Perception cycle + perceptual hash for robust context
        result = self.pipeline.perception_cycle(image)
        current_sks = set(result.sks_clusters.keys())
        current_sks |= _perceptual_hash(image)

        # 3. Select action via intrinsic motivation
        action = self.motivation.select_action(
            current_sks, self.causal_model, self.N_ACTIONS
        )

        # 4. Inject motor currents for the chosen action
        motor_currents = self.motor.encode(action, device=self.pipeline.engine.device)
        self.pipeline.inject_motor_currents(motor_currents)

        # Save state for observe_result
        self._pre_sks = current_sks
        self._last_action = action
        self._step_count += 1

        return action

    def observe_result(self, obs: np.ndarray) -> float:
        """After env.step(action), observe the consequence:

        1. obs → perception_cycle → new SKS
        2. CausalWorldModel.observe_transition(pre_sks, action, post_sks)
        3. Compute prediction error
        4. IntrinsicMotivation.update()

        Returns:
            prediction_error (float)
        """
        if self._pre_sks is None or self._last_action is None:
            return 0.0

        # 1. Perceive new state + perceptual hash
        image = self.obs_adapter.convert(obs)
        result = self.pipeline.perception_cycle(image)
        post_sks = set(result.sks_clusters.keys())
        post_sks |= _perceptual_hash(image)

        # 2. Record causal transition
        self.causal_model.observe_transition(
            self._pre_sks, self._last_action, post_sks
        )

        # 3. Compute prediction error
        predicted_effect, confidence = self.causal_model.predict_effect(
            self._pre_sks, self._last_action
        )
        if predicted_effect:
            actual_effect = post_sks.symmetric_difference(self._pre_sks)
            # Symmetric difference between predicted and actual as error
            diff = predicted_effect.symmetric_difference(actual_effect)
            prediction_error = len(diff) / max(len(predicted_effect | actual_effect), 1)
        else:
            prediction_error = 1.0  # no prediction = max error

        # 4. Update motivation
        self.motivation.update(self._pre_sks, self._last_action, prediction_error)

        return prediction_error

    def plan_to_goal(self, goal_obs: np.ndarray) -> list[int] | None:
        """Plan action sequence to reach a goal state using mental simulation."""
        image = self.obs_adapter.convert(goal_obs)
        result = self.pipeline.perception_cycle(image)
        goal_sks = set(result.sks_clusters.keys())

        current_sks = self._pre_sks or set()

        return self.simulator.find_plan(
            current_sks,
            goal_sks,
            max_depth=self.config.simulation_max_depth,
            n_actions=self.N_ACTIONS,
            min_confidence=self.config.simulation_min_confidence,
        )
