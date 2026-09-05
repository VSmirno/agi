"""Contract tests for the lean transferable-learning experience core."""

from __future__ import annotations

import numpy as np
import pytest

from snks.learning.core_replay import SequenceReplay
from snks.env.core_types import Episode, Mode, Observation, Transition
from snks.pipeline.core_metrics import normalized_auc, paired_cluster_interval
from snks.pipeline.core_transfer import TransferCondition, prepare_transfer


def _episode(
    uid: str,
    schema: str = "rgb-v1",
    *,
    steps: int = 3,
    complete: bool = True,
    sensor_changes: bool = True,
) -> Episode:
    """Build a hand-sized, ordered real episode for replay contracts."""
    observations = [
        Observation(
            rgb=np.full((3, 64, 64), step, dtype=np.uint8),
            sensors=np.array([step if sensor_changes else 0], dtype=np.float32),
            sensor_mask=np.array([True]),
            schema=schema,
            step=step,
        )
        for step in range(steps + 1)
    ]
    transitions = tuple(
        Transition(
            before=observations[step],
            action=step,
            after=observations[step + 1],
            terminated=complete and step == steps - 1,
            truncated=False,
        )
        for step in range(steps)
    )
    return Episode(uid=uid, split="train", family="toy", ruleset="r1", transitions=transitions)


def test_replay_rejects_unfinished_or_empty_and_evaluate_never_mutates() -> None:
    """Catches accidentally training on partial/evaluation experience."""
    replay = SequenceReplay(capacity=4, seed=3)

    with pytest.raises(ValueError):
        replay.append(_episode("partial", complete=False), Mode.TRAIN)
    with pytest.raises(ValueError):
        replay.append(_episode("empty", steps=0), Mode.ADAPT)

    with pytest.raises(PermissionError):
        replay.append(_episode("eval"), Mode.EVALUATE)
    assert replay.manifest()["episode_count"] == 0


def test_replay_deduplicates_and_samples_one_schema_in_ordered_windows() -> None:
    """Catches UID duplication, cross-schema batches, and shuffled transitions."""
    replay = SequenceReplay(capacity=4, seed=7)
    replay.append(_episode("a", "alpha"), Mode.TRAIN)
    replay.append(_episode("a", "alpha",), Mode.TRAIN)
    replay.append(_episode("b", "beta"), Mode.TRAIN)

    windows = replay.sample(batch_size=2, length=2, burn_in=0, recent_fraction=0.5)
    assert replay.manifest()["episode_count"] == 2
    assert len(windows) == 2
    assert {transition.before.schema for window in windows for transition in window.transitions} in (
        {"alpha"},
        {"beta"},
    )
    for window in windows:
        assert [item.before.step for item in window.transitions] == sorted(
            item.before.step for item in window.transitions
        )


def test_replay_can_mix_uniform_windows_with_observable_salient_events() -> None:
    replay = SequenceReplay(capacity=4, seed=7)
    replay.append(_episode("terminal", steps=6, sensor_changes=False), Mode.TRAIN)

    windows = replay.sample(
        batch_size=2,
        length=2,
        burn_in=0,
        recent_fraction=1.0,
        schema="rgb-v1",
        salient_fraction=0.5,
    )

    assert len(windows) == 2
    assert windows[0].transitions[-1].terminated
    with pytest.raises(ValueError, match="salient_fraction"):
        replay.sample(1, 2, 0, 1.0, "rgb-v1", salient_fraction=1.1)


def test_replay_snapshot_round_trip_preserves_manifest_and_seeded_sampling(tmp_path) -> None:
    """Catches lossy persistence of experience or sampling RNG state."""
    replay = SequenceReplay(capacity=4, seed=11)
    replay.append(_episode("a"), Mode.TRAIN)
    replay.append(Episode("b", "adapt", "toy", "r1", _episode("b").transitions), Mode.ADAPT)
    snapshot = tmp_path / "replay.npz"
    replay.save(snapshot)

    restored = SequenceReplay.load(snapshot)
    assert restored.manifest() == replay.manifest()
    assert [window.uid for window in restored.sample(2, 2, 0, 1.0, "rgb-v1")] == [
        window.uid for window in replay.sample(2, 2, 0, 1.0, "rgb-v1")
    ]


def test_metrics_integrate_normalized_area_and_refuse_small_seed_ci() -> None:
    """Catches endpoint averaging and statistically invalid bootstrap claims."""
    assert normalized_auc([0, 1, 2], [0.0, 1.0, 1.0]) == pytest.approx(0.75)
    with pytest.raises(ValueError, match="5"):
        paired_cluster_interval([1.0, 2.0], [0.0, 0.0], seed=13, n_boot=20)


def test_transfer_keeps_source_weights_and_pairs_new_schema_initialization() -> None:
    """Catches branches that confound transferred weights with target init/replay."""
    import torch

    from snks.agent.core_world_model import CoreWorldModel
    from snks.encoder.core_encoder import CoreEncoder
    from snks.pipeline.core_config import CoreConfig

    config = CoreConfig(z_dim=64, h_dim=32, ensemble_size=3)
    source = CoreWorldModel(CoreEncoder(64), {"source": (2, 1)}, 32, 3)
    replay = SequenceReplay(4, 1)
    weights, _, weights_replay = prepare_transfer(
        source, replay, TransferCondition.WEIGHTS, "target", (3, 1), 19, config
    )
    replay_branch, _, copied_replay = prepare_transfer(
        source, replay, TransferCondition.WEIGHTS_REPLAY, "target", (3, 1), 19, config
    )

    source_state = source.encoder.state_dict()
    transferred_state = weights.encoder.state_dict()
    matching_keys = source_state.keys() & transferred_state.keys()
    assert matching_keys
    assert all(torch.equal(source_state[key], transferred_state[key]) for key in matching_keys)
    assert torch.equal(weights.action_embeddings["target"].weight, replay_branch.action_embeddings["target"].weight)
    assert weights_replay.manifest()["episode_count"] == 0
    assert copied_replay.manifest()["episode_count"] == 0
