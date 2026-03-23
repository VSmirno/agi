"""Experiment 3: Sequence Prediction.

Deterministic sequences of length 3, 5, 7.
Gate: mean accuracy > 70%.

Approach:
  1. Training: present sequences, collect firing-rate vectors per step.
  2. Cluster training vectors with k-means (k = sequence length) to discover
     internal representations (unsupervised).
  3. Build cluster-space transition matrix from the assignment sequence.
  4. Test: classify each step via k-means, predict next cluster from transition
     matrix, map both predicted and actual clusters to labels, compare.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import numpy as np
from sklearn.cluster import KMeans

from snks.daf.types import DafConfig, PipelineConfig, SKSConfig, PredictionConfig, EncoderConfig
from snks.data.stimuli import GratingGenerator
from snks.data.sequences import SequenceGenerator
from snks.pipeline.runner import Pipeline


@dataclass
class PredictionResult:
    accuracy_3: float
    accuracy_5: float
    accuracy_7: float


def _collect_vectors(
    pipeline: Pipeline,
    images: torch.Tensor,
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


def _test_sequence(
    pipeline: Pipeline,
    seq_gen: SequenceGenerator,
    length: int,
    n_train_repeats: int = 20,
    n_test_repeats: int = 5,
) -> float:
    """Train and test on one sequence length. Returns prediction accuracy."""
    order = list(range(length))

    # --- Training phase ---
    images_train, labels_train, _ = seq_gen.deterministic(order, n_repeats=n_train_repeats)
    train_vectors = _collect_vectors(pipeline, images_train)
    train_labels = labels_train.numpy()

    # Cluster firing-rate vectors (unsupervised)
    km = KMeans(n_clusters=length, n_init=10, random_state=42)
    cluster_ids = km.fit_predict(train_vectors)

    # Build cluster→label mapping (majority vote)
    cluster_to_label = {}
    for c in range(length):
        mask = cluster_ids == c
        if mask.any():
            counts = np.bincount(train_labels[mask], minlength=length)
            cluster_to_label[c] = int(np.argmax(counts))
        else:
            cluster_to_label[c] = c

    # Build cluster-space transition matrix
    transition_counts = np.zeros((length, length), dtype=np.float64)
    for i in range(len(cluster_ids) - 1):
        transition_counts[cluster_ids[i], cluster_ids[i + 1]] += 1
    row_sums = transition_counts.sum(axis=1, keepdims=True)
    transition_probs = np.where(row_sums > 0, transition_counts / row_sums, 0)

    # --- Test phase ---
    images_test, labels_test, _ = seq_gen.deterministic(order, n_repeats=n_test_repeats)
    test_vectors = _collect_vectors(pipeline, images_test)
    test_labels = labels_test.numpy()

    # Classify test steps using k-means
    test_cluster_ids = km.predict(test_vectors)

    # Predict next: for step i, predict cluster at i+1 via transition matrix
    correct = 0
    total = 0
    for i in range(len(test_labels) - 1):
        current_cluster = test_cluster_ids[i]
        predicted_cluster = int(np.argmax(transition_probs[current_cluster]))
        predicted_label = cluster_to_label.get(predicted_cluster, predicted_cluster)
        actual_label = test_labels[i + 1]
        total += 1
        if predicted_label == actual_label:
            correct += 1

    return correct / max(total, 1)


def run(device: str = "cpu", num_nodes: int = 10000) -> PredictionResult:
    """Run Experiment 3."""
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
        prediction=PredictionConfig(
            causal_min_confidence=0.2,
        ),
        steps_per_cycle=200,
        device=device,
    )

    gen = GratingGenerator(image_size=64, seed=42)
    seq_gen = SequenceGenerator(gen, seed=42)

    results = {}
    for length in [3, 5, 7]:
        pipeline = Pipeline(config)
        acc = _test_sequence(pipeline, seq_gen, length)
        results[length] = acc
        print(f"  Length {length}: {acc:.1%}")

    result = PredictionResult(
        accuracy_3=results[3],
        accuracy_5=results[5],
        accuracy_7=results[7],
    )

    mean_acc = (result.accuracy_3 + result.accuracy_5 + result.accuracy_7) / 3
    print(f"Experiment 3: Sequence Prediction")
    print(f"  Mean: {mean_acc:.1%}")
    print(f"  Gate (> 70%): {'PASS' if mean_acc > 0.7 else 'FAIL'}")

    return result


if __name__ == "__main__":
    run()
