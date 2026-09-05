"""Narrow transfer-condition construction for the learning-core comparison."""

from __future__ import annotations

import copy
from enum import Enum


class TransferCondition(Enum):
    FRESH = "fresh"
    WEIGHTS = "weights"
    WEIGHTS_REPLAY = "weights_replay"
    SOURCE_CONTROL = "source_control"


def prepare_transfer(source, replay, condition: TransferCondition, target_schema: str,
                     target_shape: tuple[int, int], seed: int, config):
    """Build one paired transfer branch with a new trainer and optimizer.

    Imports stay here so experiment setup does not depend on model import order.
    """
    import torch

    from snks.agent.core_world_model import CoreWorldModel
    from snks.encoder.core_encoder import CoreEncoder
    from snks.learning.core_replay import SequenceReplay
    from snks.learning.core_trainer import CoreTrainer

    if not isinstance(condition, TransferCondition):
        condition = TransferCondition(condition)
    schemas = dict(source.schemas)
    if target_schema in schemas and tuple(schemas[target_schema]) != tuple(target_shape):
        raise ValueError("target schema name already has a different shape")

    # Every branch creates its new target parameters under the same seed.  This
    # makes fresh-vs-weights comparisons differ by transfer, not initialization.
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        model = CoreWorldModel(CoreEncoder(source.encoder.z_dim), schemas,
                               source.h_dim, source.heads)
        if target_schema not in schemas:
            model.register_schema(target_schema, target_shape, seed)
        if condition is not TransferCondition.FRESH:
            model.load_state_dict(source.state_dict(), strict=False)
        model.to(config.device)

    trainer = CoreTrainer(model, config)
    branch_replay = copy.deepcopy(replay) if condition is TransferCondition.WEIGHTS_REPLAY else SequenceReplay(
        replay.capacity, seed
    )
    return model, trainer, branch_replay
