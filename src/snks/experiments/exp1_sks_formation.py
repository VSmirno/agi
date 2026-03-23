"""Experiment 1: SKS Formation.

10 oriented gratings × 50 presentations, random order.
Gate: NMI > 0.7.
"""

from __future__ import annotations

import torch

from snks.daf.types import DafConfig, PipelineConfig, SKSConfig, PredictionConfig, EncoderConfig
from snks.data.stimuli import GratingGenerator
from snks.pipeline.runner import Pipeline, TrainResult


def make_config(device: str = "cpu", num_nodes: int = 10000) -> PipelineConfig:
    """Create experiment config."""
    return PipelineConfig(
        daf=DafConfig(
            num_nodes=num_nodes,
            avg_degree=30,
            oscillator_model="fhn",
            coupling_strength=0.05,
            dt=0.01,
            noise_sigma=0.005,
            fhn_I_base=0.0,  # quiescent without input
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
        steps_per_cycle=200,  # 200 × 0.01 = 2.0 time units (enough for FHN spike onset)
        device=device,
    )


def run(device: str = "cpu", num_nodes: int = 10000) -> TrainResult:
    """Run Experiment 1.

    Returns:
        TrainResult with final_nmi — must be > 0.7 to pass gate.
    """
    config = make_config(device, num_nodes)
    pipeline = Pipeline(config)

    gen = GratingGenerator(image_size=64, seed=42)
    images, labels = gen.generate_all(n_variations=50)  # 10 × 50 = 500

    result = pipeline.train_on_dataset(images, labels, epochs=1)

    print(f"Experiment 1: SKS Formation")
    print(f"  Images: {images.shape[0]}")
    print(f"  Final NMI: {result.final_nmi:.4f}")
    print(f"  Gate (NMI > 0.7): {'PASS' if result.final_nmi > 0.7 else 'FAIL'}")
    print(f"  Mean PE: {sum(result.mean_pe_history) / len(result.mean_pe_history):.4f}")
    print(f"  Final SKS count: {result.sks_count_history[-1] if result.sks_count_history else 0}")

    return result


if __name__ == "__main__":
    run()
