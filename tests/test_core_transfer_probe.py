"""Focused regressions for transfer-probe training and accounting."""

from __future__ import annotations

import math

import numpy as np
import pytest

from snks.agent.core_world_model import CoreWorldModel
from snks.encoder.core_encoder import CoreEncoder
from snks.env.core_types import Episode, Mode, Observation, Transition
from snks.learning.core_replay import SequenceReplay
from snks.learning.core_trainer import CoreTrainer
from snks.pipeline.core_config import CoreConfig
from snks.pipeline.core_experiment import (
    _lifetime_training_cost,
    _train_updates,
    _transfer_training_config,
)
from snks.pipeline.core_transfer import TransferCondition


def _one_step_episode() -> Episode:
    before = Observation(
        np.zeros((3, 64, 64), dtype=np.uint8),
        np.array([0.0], dtype=np.float32),
        np.array([True]),
        "grid-v1",
        0,
    )
    after = Observation(
        np.ones((3, 64, 64), dtype=np.uint8),
        np.array([1.0], dtype=np.float32),
        np.array([True]),
        "grid-v1",
        1,
    )
    transition = Transition(before, 1, after, terminated=True, truncated=False)
    return Episode("one-step", "adapt", "push_box", "push_2", (transition,))


def test_transfer_training_updates_from_one_step_episode_with_zero_burn_in() -> None:
    """Catches transfer profiles that discard a one-transition success as burn-in."""
    source_config = CoreConfig(
        z_dim=8,
        h_dim=8,
        ensemble_size=1,
        batch_size=1,
        burn_in=1,
        train_horizon=1,
        replay_capacity=2,
        recent_fraction=1.0,
        sigreg_weight=0.0,
    )
    config = _transfer_training_config(source_config)
    model = CoreWorldModel(CoreEncoder(config.z_dim), {"grid-v1": (2, 1)}, 8, 1)
    trainer = CoreTrainer(model, config)
    replay = SequenceReplay(config.replay_capacity, config.seed)
    replay.append(_one_step_episode(), Mode.ADAPT)

    metrics, schema_counts = _train_updates(
        model,
        trainer,
        replay,
        config,
        updates=1,
        mode=Mode.ADAPT,
        deadline=float("inf"),
        schema="grid-v1",
    )

    assert config.burn_in == 0
    assert len(metrics) == 1
    assert math.isfinite(metrics[0]["loss"])
    assert schema_counts == {"grid-v1": 1}


@pytest.mark.parametrize(
    ("condition", "expected_source"),
    (
        (TransferCondition.FRESH, 0),
        (TransferCondition.WEIGHTS, {"gradient_updates": 7}),
        (TransferCondition.WEIGHTS_REPLAY, {"gradient_updates": 7}),
    ),
)
def test_lifetime_cost_charges_source_training_only_to_transferred_conditions(
    condition: TransferCondition,
    expected_source: int | dict[str, int],
) -> None:
    """Catches charging inherited source-training cost to the FRESH condition."""
    cost = _lifetime_training_cost(
        condition,
        source_cost={"gradient_updates": 7},
        b_steps=11,
        updates=3,
    )

    assert cost == {
        "source": expected_source,
        "B_steps_including_reset": 11,
        "B_gradient_updates": 3,
    }
