from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1] / "experiments"))

from experiments.exp145_physics_transfer import _fit_mixed_hindsight_policies


def _examples(offset: float):
    anchor = torch.tensor(
        [[offset, 0.0], [offset, 1.0], [offset, 2.0]],
        dtype=torch.float32,
    )
    goal = anchor + 0.5
    action = torch.tensor([0, 1, 2], dtype=torch.long)
    weight = torch.ones(3, dtype=torch.float32)
    return anchor, goal, action, weight, 3


def test_mixed_hindsight_training_balances_odd_batches_and_matches_controls():
    _, training = _fit_mixed_hindsight_policies(
        terminal_examples=_examples(0.0),
        local_examples=_examples(10.0),
        z_dim=2,
        updates=2,
        batch_size=5,
        seed=145,
    )

    batch_sources = training["batch_sources"]
    assert len(batch_sources) == 2
    assert all(
        batch["terminal"] + batch["local"] == 5
        and abs(batch["terminal"] - batch["local"]) == 1
        for batch in batch_sources
    )
    assert sum(batch["terminal"] for batch in batch_sources) == 5
    assert sum(batch["local"] for batch in batch_sources) == 5
    assert training["same_initialization_and_batches"] is True
