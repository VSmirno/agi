"""Checks for the profile-scoped sensor-condition normalization experiment."""

from dataclasses import asdict
from pathlib import Path

import torch

from snks.agent.core_world_model import CoreWorldModel, LatentState
from snks.encoder.core_encoder import CoreEncoder
from snks.pipeline.core_config import CoreConfig, load_core_config


def test_sensor_condition_normalization_is_profile_scoped():
    legacy_values = asdict(CoreConfig())
    legacy_values.pop("normalize_sensor_condition")
    assert CoreConfig(**legacy_values).normalize_sensor_condition is False
    profile = load_core_config(
        Path(__file__).parents[1] / "configs" / "core_condition_norm.yaml"
    )
    assert profile.normalize_sensor_condition is True
    assert CoreConfig(**asdict(profile)) == profile

    def condition(sensor: float, normalize: bool) -> torch.Tensor:
        torch.manual_seed(3)
        model = CoreWorldModel(
            CoreEncoder(2), {"toy": (1, 1)}, 4, 1,
            normalize_sensor_condition=normalize,
        )
        with torch.no_grad():
            projection = model.sensor_projections["toy"]
            projection.weight.zero_()
            projection.bias.zero_()
            projection.weight[:, 0] = torch.tensor([1.0, 2.0, -1.0, -2.0])
            model.action_embeddings["toy"].weight.zero_()
        state = LatentState(
            z=torch.zeros(1, 2),
            sensors=torch.tensor([[sensor]]),
            sensor_mask=torch.ones(1, 1, dtype=torch.bool),
            hidden=torch.zeros(1, 4),
            schema="toy",
        )
        recurrent_inputs = []
        hook = model.recurrent.register_forward_pre_hook(
            lambda _module, args: recurrent_inputs.append(args[0].detach())
        )
        model.step(state, torch.zeros(1, dtype=torch.long))
        hook.remove()
        return recurrent_inputs[0][:, 2:]

    assert torch.allclose(condition(20.0, True), condition(200.0, True))
    assert not torch.allclose(condition(20.0, False), condition(200.0, False))
