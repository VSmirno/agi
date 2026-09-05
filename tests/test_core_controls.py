"""The two invariants that make a dynamics comparison interpretable."""

import torch


def test_controls_freeze_the_same_representation():
    from snks.encoder.core_encoder import CoreEncoder
    from snks.agent.core_world_model import CoreWorldModel
    from snks.pipeline.core_controls import build_dynamics_controls

    source = CoreWorldModel(CoreEncoder(16), {"toy": (2, 1)}, 16, 2)
    controls = build_dynamics_controls(source)
    assert set(controls) == {"initial", "real_actions", "shuffled_actions"}
    for variant in controls.values():
        assert not any(p.requires_grad for p in variant.encoder.parameters())
        assert all(torch.equal(a, b) for a, b in
                   zip(source.encoder.parameters(), variant.encoder.parameters()))


def test_action_shuffle_preserves_targets_and_action_counts():
    from snks.learning.core_trainer import SequenceBatch
    from snks.pipeline.core_controls import shuffle_action_labels

    batch = SequenceBatch(torch.zeros(2, 4, 3, 64, 64), torch.zeros(2, 4, 1),
                          torch.ones(2, 4, 1, dtype=torch.bool),
                          torch.tensor([[0, 0, 1], [1, 1, 0]]),
                          torch.zeros(2, 3), torch.ones(2, 3, dtype=torch.bool), "toy", 0)
    shuffled = shuffle_action_labels(batch, seed=3)
    assert torch.equal(batch.rgb, shuffled.rgb)
    assert torch.equal(batch.actions.sort().values, torch.tensor([[0, 0, 1], [0, 1, 1]]))
    assert torch.equal(batch.actions.flatten().sort().values,
                       shuffled.actions.flatten().sort().values)
    assert not torch.equal(batch.actions, shuffled.actions)
