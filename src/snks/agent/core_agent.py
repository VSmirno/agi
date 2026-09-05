"""A learned agent gets observations, never an environment or evaluator."""

import random

import torch

from snks.agent.core_cost import GoalCost
from snks.agent.core_planner import beam_plan
from snks.agent.core_world_model import CoreWorldModel, LatentState
from snks.env.core_types import GoalSpec, Observation, Transition
from snks.pipeline.core_config import CoreConfig


class CoreAgent:
    def __init__(self, model: CoreWorldModel, config: CoreConfig):
        self.model, self.config = model, config
        self.rng = random.Random(config.seed)
        self.last_trace: list[dict] = []
        self.last_model_calls = 0

    @torch.no_grad()
    def start(self, obs: Observation, goal: GoalSpec) -> None:
        self.state = self.model.initial(obs)
        goal_z = None if goal.image is None else self.model.initial(goal.image).z
        self.cost = GoalCost(goal_z, goal.ranges)
        self.last_trace = []

    @torch.no_grad()
    def act(self, exploration_fraction: float = 0.0) -> int:
        n_actions = self.model.schemas[self.state.schema][0]
        if self.rng.random() < exploration_fraction:
            self.last_model_calls = 0
            action = self.rng.randrange(n_actions)
            self.last_trace = [{"exploration": True, "action": action}]
            return action
        result = beam_plan(self.model, self.state, self.cost, n_actions,
                           self.config.planner_horizon, self.config.beam_width,
                           self.config.max_model_calls)
        self.last_trace, self.last_model_calls = result.trace, result.model_calls
        return result.actions[0]

    @torch.no_grad()
    def observe(self, transition: Transition) -> None:
        action = torch.tensor([transition.action], device=self.state.z.device)
        predicted = self.model.step(self.state, action)
        actual = self.model.initial(transition.after)
        self.state = LatentState(actual.z, actual.sensors, actual.sensor_mask,
                                 predicted.next_state.hidden.detach(), actual.schema)
