"""Focused contracts for the exp161 teacher-forced amplitude input probe."""

from __future__ import annotations

import importlib
import importlib.util
from types import SimpleNamespace

import torch


def _experiment():
    name = "experiments.exp161_amplitude_input_probe"
    assert importlib.util.find_spec(name) is not None, "exp161 experiment is missing"
    return importlib.import_module(name)


def test_episode_split_is_complete_and_disjoint():
    exp = _experiment()
    episodes = {
        "layout": [SimpleNamespace(uid=f"episode-{index}") for index in range(8)]
    }

    train, heldout = exp.episode_disjoint_split(episodes)

    train_ids = {episode.uid for episode in train["layout"]}
    heldout_ids = {episode.uid for episode in heldout["layout"]}
    assert len(train_ids) == 6
    assert len(heldout_ids) == 2
    assert not train_ids & heldout_ids
    assert train_ids | heldout_ids == {episode.uid for episode in episodes["layout"]}


def test_hidden_arm_distinguishes_same_z_with_different_hidden_state():
    exp = _experiment()
    probe = exp.AmplitudeInputProbe(z_dim=2, h_dim=1, heads=1, use_hidden=True)
    with torch.no_grad():
        for parameter in probe.parameters():
            parameter.zero_()
        probe.by_action[3].weight[0, 2] = 2.0
    z = torch.zeros(2, 2)
    hidden = torch.tensor([[0.0], [1.0]])
    actions = torch.tensor([3, 3])

    amplitudes = probe(z, hidden, actions)

    torch.testing.assert_close(amplitudes[:, 0], torch.tensor([0.5]))
    torch.testing.assert_close(amplitudes[:, 1], torch.sigmoid(torch.tensor([2.0])))
