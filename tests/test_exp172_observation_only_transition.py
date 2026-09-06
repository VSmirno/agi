"""Focused model contract for the exp172 observation-only transition."""

from __future__ import annotations

import importlib
import importlib.util

import torch

from snks.agent.core_world_model import CoreWorldModel, LatentState
from snks.encoder.core_encoder import CoreEncoder


def _experiment():
    name = "experiments.exp172_observation_only_transition"
    assert importlib.util.find_spec(name) is not None, "exp172 experiment is missing"
    return importlib.import_module(name)


def _state(batch: int) -> LatentState:
    return LatentState(
        z=torch.arange(batch * 4, dtype=torch.float32).reshape(batch, 4) / 10,
        sensors=torch.ones(batch, 2),
        sensor_mask=torch.ones(batch, 2, dtype=torch.bool),
        hidden=torch.arange(batch * 3, dtype=torch.float32).reshape(batch, 3) / 10,
        schema="grid-v1",
    )


def test_observation_only_model_supports_arbitrary_batch_and_carried_rollout():
    """Dropping any external pose/batch context must keep native recurrent plumbing."""

    exp = _experiment()
    baseline = CoreWorldModel(CoreEncoder(4), {"grid-v1": (5, 2)}, 3, 2)
    vector = exp.ObservationVectorDelta(4, 3, 2)
    event = exp.ObservationEventHead(4, 3)
    model = exp._installed_model(baseline, vector, event)

    state = _state(3)
    actions = torch.tensor([0, 2, 4], dtype=torch.long)
    native = baseline.step(state, actions)
    predicted = model.step(state, actions)

    assert predicted.member_z.shape == (2, 3, 4)
    torch.testing.assert_close(predicted.next_state.hidden, native.next_state.hidden)
    torch.testing.assert_close(predicted.next_state.sensors, native.next_state.sensors)
    torch.testing.assert_close(predicted.terminated_prob, native.terminated_prob)

    first = model.step(_state(1), torch.tensor([1], dtype=torch.long))
    expected_second = CoreWorldModel.step(
        model, first.next_state, torch.tensor([3], dtype=torch.long)
    )
    rollout = model.rollout(
        _state(1), torch.tensor([[1, 3]], dtype=torch.long)
    )

    assert len(rollout) == 2
    torch.testing.assert_close(
        rollout[1].next_state.hidden, expected_second.next_state.hidden
    )
    torch.testing.assert_close(
        rollout[1].next_state.sensors, expected_second.next_state.sensors
    )
