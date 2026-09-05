"""Small end-to-end safeguards for the learning-core experiment."""

import numpy as np
import pytest


def test_observation_does_not_alias_environment_arrays():
    from snks.env.core_types import Observation

    pixels = np.zeros((3, 64, 64), dtype=np.uint8)
    obs = Observation(pixels, np.array([1.0]), np.array([True]), "toy", 0)
    pixels[:] = 255
    assert obs.rgb.max() == 0
    assert not hasattr(obs, "info")


def test_invalid_experiment_budget_is_rejected():
    from snks.pipeline.core_config import CoreConfig

    with pytest.raises(ValueError):
        CoreConfig(max_model_calls=0)


def test_planner_uses_dynamics_and_respects_candidate_budget():
    import torch
    from snks.agent.core_cost import GoalCost
    from snks.agent.core_planner import beam_plan
    from snks.agent.core_world_model import LatentState, Prediction

    class IncrementDynamics:
        def step(self, state, actions):
            sensors = state.sensors + actions[:, None]
            next_state = LatentState(state.z, sensors, state.sensor_mask,
                                     state.hidden, state.schema)
            zeros = torch.zeros(len(actions))
            return Prediction(next_state, zeros, zeros, state.z[None])

    root = LatentState(torch.zeros(1, 2), torch.zeros(1, 1),
                       torch.ones(1, 1, dtype=torch.bool), torch.zeros(1, 2), "toy")
    result = beam_plan(IncrementDynamics(), root, GoalCost(None, {0: (2., 2.)}),
                       n_actions=2, horizon=2, beam_width=2, max_calls=6)
    assert result.actions == (1, 1)
    assert result.model_calls == 6
    assert root.sensors.item() == 0


def test_evaluation_leaves_model_and_replay_unchanged():
    import torch
    from types import SimpleNamespace
    from snks.agent.core_agent import CoreAgent
    from snks.agent.core_world_model import CoreWorldModel
    from snks.encoder.core_encoder import CoreEncoder
    from snks.env.core_types import ActionSpec, GoalSpec, Mode, Observation, Transition
    from snks.learning.core_replay import SequenceReplay
    from snks.pipeline.core_config import CoreConfig
    from snks.pipeline.core_runner import run_episode

    class TinyEnv:
        actions = ActionSpec("toy", ("hold", "increment"))
        reset_transitions = 0

        def reset(self, seed):
            self.obs = Observation(np.zeros((3, 64, 64), np.uint8),
                                   np.array([0.]), np.array([True]), "toy", 0)
            return self.obs

        def step(self, action):
            before = self.obs
            self.obs = Observation(before.rgb, before.sensors + action,
                                   before.sensor_mask, "toy", before.step + 1)
            return Transition(before, action, self.obs, self.obs.step == 3, False)

        def diagnostic_snapshot(self):
            return {"success": bool(self.obs.sensors[0] >= 1)}

    torch.manual_seed(0)
    config = CoreConfig(planner_horizon=1, max_model_calls=2)
    model = CoreWorldModel(CoreEncoder(64), {"toy": (2, 1)}, 32, 3)
    replay = SequenceReplay(8, 0)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    case = SimpleNamespace(uid="toy-0", family="toy", ruleset="toy", seed=0,
                           split="validation", goal=GoalSpec(None, {0: (1., 3.)}),
                           max_steps=3)
    result = run_episode(TinyEnv(), CoreAgent(model, config), case,
                         Mode.EVALUATE, replay, None)
    assert result.steps == 3
    assert not result.agent_failed
    assert all(torch.equal(value, before[key]) for key, value in model.state_dict().items())
    assert replay.manifest()["episode_count"] == 0
