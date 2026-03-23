"""Experiment 4: Noise Robustness.

Trained pipeline + Gaussian noise sigma=0.1, 0.2, 0.3.
Gate: graceful degradation (not sudden drop).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import numpy as np
from sklearn.cluster import KMeans

from snks.daf.types import DafConfig, PipelineConfig, SKSConfig, PredictionConfig, EncoderConfig
from snks.data.stimuli import GratingGenerator
from snks.pipeline.runner import Pipeline
from snks.sks.metrics import compute_nmi


@dataclass
class NoiseResult:
    nmi_clean: float
    nmi_noise_01: float
    nmi_noise_02: float
    nmi_noise_03: float
    graceful: bool  # True if degradation is monotonic and not sudden


def _collect_rate_vectors(
    pipeline: Pipeline, images: torch.Tensor,
) -> np.ndarray:
    """Run images through pipeline and collect firing-rate vectors."""
    n_nodes = pipeline.engine.config.num_nodes
    vectors = np.zeros((len(images), n_nodes), dtype=np.float32)
    for i in range(len(images)):
        pipeline.perception_cycle(images[i])
        fired = pipeline.engine.get_fired_history()
        if fired is not None:
            vectors[i] = fired.float().mean(dim=0).cpu().numpy()
    return vectors


def _evaluate_nmi_kmeans(
    pipeline: Pipeline,
    images: torch.Tensor,
    labels: torch.Tensor,
    km: KMeans,
) -> float:
    """Classify images via k-means on firing-rate vectors and compute NMI."""
    vectors = _collect_rate_vectors(pipeline, images)
    pred_labels = km.predict(vectors)
    return compute_nmi(pred_labels, labels.numpy())


def run(device: str = "cpu", num_nodes: int = 10000) -> NoiseResult:
    """Run Experiment 4."""
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

    # Train
    images, labels = gen.generate_all(n_variations=50)
    result = pipeline.train_on_dataset(images, labels, epochs=1)
    n_classes = int(labels.max().item()) + 1

    # Build k-means from training data
    train_vectors = _collect_rate_vectors(pipeline, images)
    km = KMeans(n_clusters=n_classes, n_init=10, random_state=42)
    km.fit(train_vectors)

    # Test clean
    test_images, test_labels = gen.generate_all(n_variations=10)
    nmi_clean = _evaluate_nmi_kmeans(pipeline, test_images, test_labels, km)

    # Test with noise
    nmis = {}
    for sigma in [0.1, 0.2, 0.3]:
        torch.manual_seed(42)
        noisy = test_images + sigma * torch.randn_like(test_images)
        noisy = noisy.clamp(0.0, 1.0)
        nmis[sigma] = _evaluate_nmi_kmeans(pipeline, noisy, test_labels, km)

    # Graceful = monotonically decreasing or total drop < 50% of clean
    graceful = (
        nmi_clean >= nmis[0.1] >= nmis[0.2] >= nmis[0.3]
        or (nmi_clean - nmis[0.3]) / max(nmi_clean, 1e-8) < 0.5
    )

    result = NoiseResult(
        nmi_clean=nmi_clean,
        nmi_noise_01=nmis[0.1],
        nmi_noise_02=nmis[0.2],
        nmi_noise_03=nmis[0.3],
        graceful=graceful,
    )

    print(f"Experiment 4: Noise Robustness")
    print(f"  Clean NMI: {result.nmi_clean:.4f}")
    print(f"  sigma=0.1 NMI: {result.nmi_noise_01:.4f}")
    print(f"  sigma=0.2 NMI: {result.nmi_noise_02:.4f}")
    print(f"  sigma=0.3 NMI: {result.nmi_noise_03:.4f}")
    print(f"  Graceful degradation: {'YES' if result.graceful else 'NO'}")

    return result


if __name__ == "__main__":
    run()
