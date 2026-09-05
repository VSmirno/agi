"""Focused contracts for the exp165 privileged relational-pose diagnostic."""

from __future__ import annotations

import importlib
import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def _experiment():
    name = "experiments.exp165_relational_pose_probe"
    assert importlib.util.find_spec(name) is not None, "exp165 experiment is missing"
    return importlib.import_module(name)


def test_pose_vector_appends_valid_agent_direction_one_hot():
    exp = _experiment()
    snapshot = {"agent_pos": (1, 1), "box_pos": (2, 3), "goal_pos": (5, 4)}

    vector = exp.pose_vector(snapshot, agent_dir=2)

    torch.testing.assert_close(
        vector, torch.tensor([0.2, 0.4, 0.6, 0.2, 0, 0, 1, 0])
    )
    with pytest.raises(ValueError, match="agent_dir"):
        exp.pose_vector(snapshot, agent_dir=4)


def test_pose_sidecar_reads_direction_before_each_transition():
    exp = _experiment()
    observations = [
        SimpleNamespace(rgb=np.array([index], dtype=np.uint8)) for index in range(3)
    ]
    transitions = tuple(
        SimpleNamespace(before=observations[index], after=observations[index + 1], action=2)
        for index in range(2)
    )
    episode = SimpleNamespace(uid="episode-pose", transitions=transitions)

    class FakeAdapter:
        def __init__(self):
            self.index = 0
            self.world = SimpleNamespace(agent_dir=0)

        def reset(self, seed):
            self.index = 0
            self.world.agent_dir = 0
            return observations[0]

        def diagnostic_snapshot(self):
            return {
                "agent_pos": (1 + self.index, 1),
                "box_pos": (2 + self.index, 1),
                "goal_pos": (5, 1),
            }

        def step(self, action):
            transition = transitions[self.index]
            self.index += 1
            self.world.agent_dir = self.index
            return transition

    sidecar = exp.aligned_episode_pose(episode, FakeAdapter(), seed=123)

    torch.testing.assert_close(
        sidecar[("episode-pose", 0)][4:], torch.tensor([1, 0, 0, 0])
    )
    torch.testing.assert_close(
        sidecar[("episode-pose", 1)][4:], torch.tensor([0, 1, 0, 0])
    )
    assert ("episode-pose", 2) not in sidecar


def test_probe_dimension_and_output_depend_on_pose_at_fixed_relations():
    exp = _experiment()
    probe = exp.RelationalPoseProbe(z_dim=1, h_dim=1, heads=1)
    assert probe.by_action[0][0].in_features == 10
    with torch.no_grad():
        for parameter in probe.parameters():
            parameter.zero_()
        probe.by_action[2][0].weight[0, 6] = 1.0
        probe.by_action[2][2].weight[0, 0] = 1.0
    z = torch.zeros(1, 1)
    hidden = torch.zeros(1, 1)
    actions = torch.tensor([2], dtype=torch.long)
    pose0 = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 1]], dtype=torch.float32)
    pose1 = torch.tensor([[0, 0, 0, 0, 1, 0, 0, 0]], dtype=torch.float32)

    output0 = probe(z, hidden, pose0, actions)
    output1 = probe(z, hidden, pose1, actions)

    torch.testing.assert_close(output0, torch.tensor([[0.5]]))
    assert float(output1) > float(output0)


def test_defaults_lock_protocol_observability_and_exp164_reference():
    exp = _experiment()
    args = exp.build_parser().parse_args(
        ["--config", "config.yaml", "--baseline-checkpoint", "base.pt", "--out", "run"]
    )

    assert args.episodes_per_layout == 512
    assert args.collection_steps == 64
    assert args.probe_updates == 400
    assert args.probe_batch_size == 256
    assert args.progress_interval == 30
    assert args.exp164_reference == exp.DEFAULT_EXP164_REFERENCE
