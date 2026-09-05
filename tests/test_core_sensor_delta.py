"""Checks for profile-scoped residual sensor prediction."""

from pathlib import Path

import torch

from snks.agent.core_world_model import CoreWorldModel, LatentState
from snks.encoder.core_encoder import CoreEncoder
from snks.learning.core_trainer import CoreTrainer, SequenceBatch
from snks.pipeline.core_config import CoreConfig, load_core_config


def test_zero_delta_heads_predict_sensor_persistence_over_rollout():
    assert CoreConfig().predict_sensor_delta is False
    profile = load_core_config(
        Path(__file__).parents[1] / "configs" / "core_sensor_delta.yaml"
    )
    assert profile.predict_sensor_delta is True
    assert profile.normalize_sensor_condition is False
    model = CoreWorldModel(
        CoreEncoder(2), {"toy": (1, 1)}, 4, 2, predict_sensor_delta=True
    )
    with torch.no_grad():
        for head in model.sensor_heads["toy"]:
            head.weight.zero_()
            head.bias.zero_()
    state = LatentState(
        torch.zeros(1, 2), torch.tensor([[3.0]]), torch.ones(1, 1, dtype=torch.bool),
        torch.zeros(1, 4), "toy",
    )

    predictions = model.rollout(state, torch.zeros(1, 3, dtype=torch.long))

    assert all(torch.equal(prediction.next_state.sensors, state.sensors)
               for prediction in predictions)


def test_trainer_supervises_member_deltas_as_absolute_predictions():
    model = CoreWorldModel(
        CoreEncoder(2), {"toy": (1, 1)}, 4, 2, predict_sensor_delta=True
    )
    with torch.no_grad():
        for head in model.sensor_heads["toy"]:
            head.weight.zero_()
            head.bias.fill_(1.0)
    trainer = CoreTrainer(model, CoreConfig(train_horizon=1, sigreg_weight=0.0))
    batch = SequenceBatch(
        rgb=torch.zeros(1, 2, 3, 64, 64),
        sensors=torch.tensor([[[2.0], [3.0]]]),
        sensor_mask=torch.ones(1, 2, 1, dtype=torch.bool),
        actions=torch.zeros(1, 1, dtype=torch.long),
        terminated=torch.zeros(1, 1),
        valid=torch.ones(1, 1, dtype=torch.bool),
        schema="toy",
        burn_in=0,
    )

    trainer.compute_loss(batch).backward()

    assert all(head.bias.grad is not None
               and torch.equal(head.bias.grad, torch.zeros_like(head.bias))
               for head in model.sensor_heads["toy"])
