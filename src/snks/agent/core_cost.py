"""Goal scoring, separate from predictions and environment rewards."""

import torch
from torch import Tensor

from snks.agent.core_world_model import Prediction


class GoalCost:
    def __init__(self, goal_z: Tensor | None, ranges: dict[int, tuple[float, float]],
                 image_weight: float = 1.0, sensor_weight: float = 1.0,
                 uncertainty_weight: float = 0.0, termination_weight: float = 0.0):
        self.goal_z = None if goal_z is None else goal_z.detach().clone()
        self.ranges = dict(ranges)
        self.image_weight = image_weight
        self.sensor_weight = sensor_weight
        self.uncertainty_weight = uncertainty_weight
        # Ending can mean success; penalizing it is an explicit profile choice.
        self.termination_weight = termination_weight

    def __call__(self, prediction: Prediction) -> Tensor:
        state = prediction.next_state
        score = torch.zeros(len(state.z), device=state.z.device)
        if self.goal_z is not None:
            score = score + self.image_weight * (state.z - self.goal_z).square().mean(-1)
        for index, (lower, upper) in self.ranges.items():
            if lower > upper or not state.sensor_mask[:, index].all():
                raise ValueError("goal requires a present sensor and ordered interval")
            value = state.sensors[:, index]
            violation = torch.relu(lower - value) + torch.relu(value - upper)
            score = score + self.sensor_weight * violation.square()
        return (score + self.uncertainty_weight * prediction.uncertainty
                + self.termination_weight * prediction.terminated_prob)
