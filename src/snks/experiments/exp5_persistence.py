"""Experiment 5: DCAM Persistence.

Store episodes → save → reload → verify query accuracy preserved.
Gate: |accuracy_before - accuracy_after| <= 0.01
"""

from __future__ import annotations

import tempfile
import os
from dataclasses import dataclass

import torch

from snks.daf.types import DcamConfig
from snks.dcam.world_model import DcamWorldModel


@dataclass
class PersistenceResult:
    accuracy_before: float
    accuracy_after: float
    delta: float
    passed: bool


def run(device: str = "cpu") -> PersistenceResult:
    """Run Experiment 5."""
    config = DcamConfig(
        hac_dim=256,
        lsh_n_tables=8,
        lsh_n_bits=8,
        episodic_capacity=100,
    )
    model = DcamWorldModel(config, torch.device(device))

    # Generate and store episodes
    N = 50
    torch.manual_seed(42)
    vectors = [torch.randn(config.hac_dim) for _ in range(N)]
    vectors = [v / v.norm() for v in vectors]

    ids = []
    for i, v in enumerate(vectors):
        eid = model.store_episode({i: (0.0, 1.0)}, v, importance=0.5)
        ids.append(eid)

    def measure_accuracy(m: DcamWorldModel) -> float:
        hits = 0
        for i, v in enumerate(vectors):
            results = m.query_similar(v, top_k=1)
            if results and results[0][0] == ids[i]:
                hits += 1
        return hits / N

    acc_before = measure_accuracy(model)

    # Save and reload
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "dcam_checkpoint")
        model.save(path)

        model2 = DcamWorldModel(config, torch.device(device))
        model2.load(path)

        acc_after = measure_accuracy(model2)

    delta = abs(acc_before - acc_after)
    passed = delta <= 0.01

    result = PersistenceResult(
        accuracy_before=acc_before,
        accuracy_after=acc_after,
        delta=delta,
        passed=passed,
    )

    print("Experiment 5: DCAM Persistence")
    print(f"  Accuracy before save: {result.accuracy_before:.4f}")
    print(f"  Accuracy after load:  {result.accuracy_after:.4f}")
    print(f"  Delta: {result.delta:.6f}")
    print(f"  Gate (delta <= 1%): {'PASS' if result.passed else 'FAIL'}")

    return result


if __name__ == "__main__":
    run()
