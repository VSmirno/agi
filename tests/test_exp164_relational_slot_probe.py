"""Focused contracts for the exp164 privileged relational-slot diagnostic."""

from __future__ import annotations

import importlib
import importlib.util
from types import SimpleNamespace

import numpy as np
import torch


def _experiment():
    name = "experiments.exp164_relational_slot_probe"
    assert importlib.util.find_spec(name) is not None, "exp164 experiment is missing"
    return importlib.import_module(name)


def test_relational_vector_is_normalized_and_translation_invariant():
    exp = _experiment()
    first = {"agent_pos": (1, 1), "box_pos": (2, 3), "goal_pos": (5, 4)}
    shifted = {"agent_pos": (2, 2), "box_pos": (3, 4), "goal_pos": (6, 5)}

    expected = torch.tensor([0.2, 0.4, 0.6, 0.2])
    torch.testing.assert_close(exp.relational_vector(first), expected)
    torch.testing.assert_close(exp.relational_vector(shifted), expected)


def test_sidecar_aligns_snapshot_before_step_without_after_leakage():
    exp = _experiment()
    observations = [
        SimpleNamespace(rgb=np.array([index], dtype=np.uint8)) for index in range(3)
    ]
    transitions = tuple(
        SimpleNamespace(before=observations[index], after=observations[index + 1], action=2)
        for index in range(2)
    )
    episode = SimpleNamespace(uid="episode-a", transitions=transitions)
    snapshots = [
        {"agent_pos": (1, 1), "box_pos": (2, 1), "goal_pos": (5, 1)},
        {"agent_pos": (2, 1), "box_pos": (3, 1), "goal_pos": (5, 1)},
        {"agent_pos": (3, 1), "box_pos": (4, 1), "goal_pos": (5, 1)},
    ]

    class FakeAdapter:
        def __init__(self):
            self.index = 0
            self.events = []

        def reset(self, seed):
            self.events.append(("reset", seed))
            return observations[0]

        def diagnostic_snapshot(self):
            self.events.append(("snapshot", self.index))
            return snapshots[self.index]

        def step(self, action):
            self.events.append(("step", self.index, action))
            transition = transitions[self.index]
            self.index += 1
            return transition

    adapter = FakeAdapter()
    sidecar = exp.aligned_episode_relations(episode, adapter, seed=123)

    assert list(sidecar) == [("episode-a", 0), ("episode-a", 1)]
    torch.testing.assert_close(sidecar[("episode-a", 0)], torch.tensor([0.2, 0, 0.6, 0]))
    torch.testing.assert_close(sidecar[("episode-a", 1)], torch.tensor([0.2, 0, 0.4, 0]))
    assert adapter.events == [
        ("reset", 123),
        ("snapshot", 0),
        ("step", 0, 2),
        ("snapshot", 1),
        ("step", 1, 2),
    ]


def test_probe_output_depends_on_relational_slot_at_fixed_state_and_action():
    exp = _experiment()
    probe = exp.RelationalSlotProbe(z_dim=1, h_dim=1, heads=1)
    with torch.no_grad():
        for parameter in probe.parameters():
            parameter.zero_()
        probe.by_action[2][0].weight[0, 2] = 1.0
        probe.by_action[2][2].weight[0, 0] = 1.0
    z = torch.zeros(1, 1)
    hidden = torch.zeros(1, 1)
    actions = torch.tensor([2], dtype=torch.long)

    without_relation = probe(z, hidden, torch.zeros(1, 4), actions)
    with_relation = probe(z, hidden, torch.tensor([[1.0, 0, 0, 0]]), actions)

    torch.testing.assert_close(without_relation, torch.tensor([[0.5]]))
    torch.testing.assert_close(with_relation, torch.tensor([[torch.sigmoid(torch.tensor(1.0))]]))


def test_defaults_lock_exact_corpus_and_progress_observability():
    exp = _experiment()
    args = exp.build_parser().parse_args(
        ["--config", "config.yaml", "--baseline-checkpoint", "base.pt", "--out", "run"]
    )

    assert args.episodes_per_layout == 512
    assert args.collection_steps == 64
    assert args.probe_updates == 400
    assert args.probe_batch_size == 256
    assert args.progress_interval == 30
    assert args.exp162_reference == exp.DEFAULT_EXP162_REFERENCE
