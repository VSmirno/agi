"""Experiment 2: Continual Learning.

Phase A: train on classes 0-4 (500 presentations).
Phase B: train on classes 5-9 (500 presentations).
Test: present classes 0-4, check retention > 85%.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import numpy as np

from snks.daf.types import DafConfig, PipelineConfig, SKSConfig, PredictionConfig, EncoderConfig
from snks.data.stimuli import GratingGenerator
from snks.pipeline.runner import Pipeline


@dataclass
class ContinualResult:
    phase_a_nmi: float
    phase_b_nmi: float
    retention_nmi: float
    retention_pct: float  # retention_nmi / phase_a_nmi * 100


def run(device: str = "cpu", num_nodes: int = 10000) -> ContinualResult:
    """Run Experiment 2."""
    config = PipelineConfig(
        daf=DafConfig(
            num_nodes=num_nodes,
            avg_degree=30,
            oscillator_model="fhn",
            coupling_strength=0.05,
            dt=0.01,
            noise_sigma=0.005,
            fhn_I_base=0.0,
            device=device,
        ),
        encoder=EncoderConfig(sdr_current_strength=1.0),
        sks=SKSConfig(
            top_k=min(num_nodes // 2, 5000),
            dbscan_eps=0.3,
            dbscan_min_samples=5,
            min_cluster_size=5,
            coherence_mode="rate",
        ),
        prediction=PredictionConfig(),
        steps_per_cycle=200,
        device=device,
    )
    pipeline = Pipeline(config)
    gen = GratingGenerator(image_size=64, seed=42)

    # Phase A: classes 0-4
    imgs_a, labels_a = [], []
    for c in range(5):
        imgs, lbls = gen.generate(c, n_variations=100)
        imgs_a.append(imgs)
        labels_a.append(lbls)
    imgs_a = torch.cat(imgs_a)
    labels_a = torch.cat(labels_a)
    result_a = pipeline.train_on_dataset(imgs_a, labels_a, epochs=1)

    # Phase B: classes 5-9
    imgs_b, labels_b = [], []
    for c in range(5, 10):
        imgs, lbls = gen.generate(c, n_variations=100)
        imgs_b.append(imgs)
        labels_b.append(lbls)
    imgs_b = torch.cat(imgs_b)
    labels_b = torch.cat(labels_b)
    result_b = pipeline.train_on_dataset(imgs_b, labels_b, epochs=1)

    # Retention test: present classes 0-4 again
    result_retest = pipeline.train_on_dataset(imgs_a, labels_a, epochs=1)

    retention_pct = (result_retest.final_nmi / max(result_a.final_nmi, 1e-8)) * 100

    result = ContinualResult(
        phase_a_nmi=result_a.final_nmi,
        phase_b_nmi=result_b.final_nmi,
        retention_nmi=result_retest.final_nmi,
        retention_pct=retention_pct,
    )

    print(f"Experiment 2: Continual Learning")
    print(f"  Phase A NMI: {result.phase_a_nmi:.4f}")
    print(f"  Phase B NMI: {result.phase_b_nmi:.4f}")
    print(f"  Retention NMI: {result.retention_nmi:.4f}")
    print(f"  Retention: {result.retention_pct:.1f}%")
    print(f"  Gate (> 85%): {'PASS' if result.retention_pct > 85 else 'FAIL'}")

    return result


if __name__ == "__main__":
    run()
