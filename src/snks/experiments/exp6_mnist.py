"""Experiment 6: MNIST Unsupervised Classification.

10 digit classes, n_per_class images each, resized to 64x64.
SDR size 8192 (pool 8x8) for better spatial discrimination.
Gate: NMI > 0.6.
"""

from __future__ import annotations

from snks.daf.types import DafConfig, PipelineConfig, SKSConfig, PredictionConfig, EncoderConfig
from snks.data.mnist import MnistLoader
from snks.pipeline.runner import Pipeline, TrainResult


def make_config(device: str = "cpu", num_nodes: int = 10000) -> PipelineConfig:
    """Create experiment config with larger SDR for MNIST."""
    return PipelineConfig(
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
        encoder=EncoderConfig(
            sdr_size=8192,
            pool_h=8,
            pool_w=8,
            sdr_sparsity=0.04,
            sdr_current_strength=1.0,
        ),
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


def run(
    device: str = "cpu",
    num_nodes: int = 10000,
    n_per_class: int = 100,
    epochs: int = 2,
) -> TrainResult:
    """Run Experiment 6.

    Returns:
        TrainResult with final_nmi — must be > 0.6 to pass gate.
    """
    config = make_config(device, num_nodes)
    pipeline = Pipeline(config)

    loader = MnistLoader(data_root="data/", target_size=64, seed=42)
    images, labels = loader.load("train", n_per_class=n_per_class)

    result = pipeline.train_on_dataset(images, labels, epochs=epochs)

    n_total = images.shape[0]
    print(f"Experiment 6: MNIST Unsupervised")
    print(f"  Images: {n_total} ({n_per_class}/class x 10)")
    print(f"  Epochs: {epochs}")
    print(f"  Final NMI: {result.final_nmi:.4f}")
    print(f"  Gate (NMI > 0.6): {'PASS' if result.final_nmi > 0.6 else 'FAIL'}")
    print(f"  Mean PE: {sum(result.mean_pe_history) / len(result.mean_pe_history):.4f}")
    print(f"  Final SKS count: {result.sks_count_history[-1] if result.sks_count_history else 0}")

    return result


if __name__ == "__main__":
    run()
