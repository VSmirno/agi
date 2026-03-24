"""Experiment 17: NoisePolicy adaptation.

3 категории × 20 предложений (животные, еда, техника).
100 циклов обучения (shuffle).

Два pipeline с одинаковым seed:
- pipeline_null:  policy="null"  (NullPolicy)
- pipeline_noise: policy="noise", strength=1.0

Gate:
  NMI(noise_policy) >= NMI(null) - 0.05
  AND std_confidence(noise_policy) < std_confidence(null)
"""

from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import torch
from sklearn.cluster import KMeans

from snks.daf.types import (
    DafConfig,
    EncoderConfig,
    GWSConfig,
    MetacogConfig,
    PipelineConfig,
    PredictionConfig,
    SKSConfig,
)
from snks.encoder.text_encoder import TextEncoder
from snks.pipeline.runner import Pipeline
from snks.sks.metrics import compute_nmi

# Те же предложения что в exp10
SENTENCES = {
    "animals": [
        "The cat sleeps on the warm sofa.",
        "Dogs love to run and play fetch.",
        "A bird sings early in the morning.",
        "The horse gallops across the open field.",
        "Fish swim silently in the deep ocean.",
        "Wolves hunt in coordinated packs.",
        "The elephant has a remarkable memory.",
        "Penguins waddle across the ice.",
        "A tiger prowls through the jungle undergrowth.",
        "Dolphins communicate using complex sounds.",
        "The rabbit nibbles on fresh carrots.",
        "An owl hunts mice at night.",
        "Bears hibernate during cold winter months.",
        "The fox is known for its cleverness.",
        "Whales migrate thousands of miles each year.",
        "A monkey swings from branch to branch.",
        "The snake slithers across the desert sand.",
        "Bees collect nectar from flowering plants.",
        "The lion roars to claim its territory.",
        "A deer leaps gracefully over the fence.",
    ],
    "food": [
        "Pizza is topped with melted mozzarella cheese.",
        "Fresh bread smells wonderful from the oven.",
        "Sushi combines rice with raw fish.",
        "A bowl of hot soup warms the body.",
        "Chocolate cake is a popular dessert.",
        "Ripe tomatoes add flavor to salad.",
        "Coffee is brewed from roasted beans.",
        "Pasta is boiled and served with sauce.",
        "Grilled chicken is a healthy meal option.",
        "Ice cream melts quickly on a warm day.",
        "Avocado toast has become a trendy breakfast.",
        "Spicy curry is cooked with many herbs.",
        "Fresh orange juice is rich in vitamin C.",
        "Pancakes are often served with maple syrup.",
        "A crisp apple is a satisfying snack.",
        "Bacon sizzles in a hot frying pan.",
        "Noodles are a staple food in Asia.",
        "Cheese is made from fermented milk.",
        "A smoothie blends fruit and yogurt together.",
        "Roasted vegetables taste sweet and savory.",
    ],
    "technology": [
        "The CPU executes billions of instructions per second.",
        "Machine learning models require large datasets.",
        "Cloud servers store data across distributed systems.",
        "A smartphone has more power than old computers.",
        "Neural networks learn patterns from training data.",
        "The GPU accelerates parallel matrix computations.",
        "Software bugs cause unexpected program crashes.",
        "Encryption protects sensitive data from hackers.",
        "Quantum computers use superposition and entanglement.",
        "A database stores structured information efficiently.",
        "Wi-Fi transmits data using radio waves.",
        "The operating system manages hardware resources.",
        "Algorithms solve problems step by step.",
        "Robots automate repetitive manufacturing tasks.",
        "Fiber optic cables transmit light at high speed.",
        "A compiler translates code into machine language.",
        "Virtual reality creates immersive digital environments.",
        "Sensors collect real-time environmental data.",
        "Cybersecurity prevents unauthorized access to networks.",
        "A solid-state drive reads data much faster than HDD.",
    ],
}

N_CYCLES = 100


def make_config(policy: str, strength: float = 1.0, device: str = "cpu") -> PipelineConfig:
    return PipelineConfig(
        daf=DafConfig(
            num_nodes=5000,
            avg_degree=20,
            oscillator_model="fhn",
            coupling_strength=0.08,
            dt=0.01,
            noise_sigma=0.003,
            fhn_I_base=0.0,
            stdp_a_plus=0.05,
            device=device,
        ),
        encoder=EncoderConfig(sdr_current_strength=1.5, image_size=32),
        sks=SKSConfig(
            top_k=2500,
            dbscan_eps=0.3,
            dbscan_min_samples=5,
            min_cluster_size=5,
            coherence_mode="rate",
        ),
        prediction=PredictionConfig(),
        gws=GWSConfig(enabled=True, w_size=1.0),
        metacog=MetacogConfig(
            enabled=True,
            policy=policy,
            policy_strength=strength,
        ),
        steps_per_cycle=50,
        device=device,
    )


def build_dataset() -> tuple[list[str], list[int]]:
    """Список (sentence, label) по всем категориям."""
    texts: list[str] = []
    labels: list[int] = []
    for label_idx, (_, sentences) in enumerate(SENTENCES.items()):
        for sentence in sentences:
            texts.append(sentence)
            labels.append(label_idx)
    return texts, labels


def train_and_eval(
    pipeline: Pipeline,
    texts: list[str],
    labels: list[int],
    n_cycles: int,
    seed: int = 42,
) -> tuple[float, float]:
    """Обучить pipeline на текстовых данных n_cycles циклов.

    Returns:
        (nmi, std_confidence)
    """
    n_total = len(texts)
    n_nodes = pipeline.engine.config.num_nodes

    # Для NMI: собираем firing patterns последнего прохода (или всех n_cycles)
    firing_patterns: list[np.ndarray] = []
    pattern_labels: list[int] = []
    confidences: list[float] = []

    rng = torch.Generator()
    rng.manual_seed(seed)

    cycles_done = 0
    epoch = 0
    while cycles_done < n_cycles:
        perm = torch.randperm(n_total, generator=rng).tolist()
        for idx in perm:
            if cycles_done >= n_cycles:
                break
            sentence = texts[idx]
            label = labels[idx]

            result = pipeline.perception_cycle(text=sentence)
            cycles_done += 1

            # Собираем firing pattern
            fired = pipeline.engine.get_fired_history()
            if fired is not None:
                pattern = fired.float().mean(dim=0).cpu().numpy()
                firing_patterns.append(pattern)
                pattern_labels.append(label)

            # Confidence
            if result.metacog is not None:
                confidences.append(result.metacog.confidence)

        epoch += 1

    # NMI: k-means на firing patterns
    if firing_patterns:
        X = np.array(firing_patterns)
        true_labels = np.array(pattern_labels)
        n_classes = len(SENTENCES)
        km = KMeans(n_clusters=n_classes, n_init=10, random_state=42)
        pred_labels = km.fit_predict(X)
        nmi = float(compute_nmi(pred_labels, true_labels))
    else:
        nmi = 0.0

    std_conf = float(np.std(confidences)) if len(confidences) > 1 else 0.0

    return nmi, std_conf


def run(device: str = "cpu") -> tuple[float, float]:
    """Run Exp 17: NoisePolicy adaptation.

    Returns:
        (nmi_noise, nmi_null) — основные метрики.
    """
    texts, labels = build_dataset()

    print("  Обучение pipeline_null (NullPolicy)...")
    pipeline_null = Pipeline(make_config("null", device=device))
    nmi_null, std_null = train_and_eval(pipeline_null, texts, labels, N_CYCLES, seed=42)

    print("  Обучение pipeline_noise (NoisePolicy)...")
    pipeline_noise = Pipeline(make_config("noise", strength=1.0, device=device))
    nmi_noise, std_noise = train_and_eval(pipeline_noise, texts, labels, N_CYCLES, seed=42)

    nmi_cond = nmi_noise >= nmi_null - 0.05
    std_cond = std_noise < std_null
    overall = nmi_cond and std_cond

    print("Exp 17: NoisePolicy adaptation")
    print(f"  NMI(null)  = {nmi_null:.3f}")
    print(f"  NMI(noise) = {nmi_noise:.3f}")
    print(f"  std_confidence(null)  = {std_null:.3f}")
    print(f"  std_confidence(noise) = {std_noise:.3f}")
    print(f"  NMI condition:  {'PASS' if nmi_cond else 'FAIL'} (NMI(noise) >= NMI(null) - 0.05)")
    print(f"  std condition:  {'PASS' if std_cond else 'FAIL'} (std_confidence(noise) < std_confidence(null))")
    print(f"  Overall: {'PASS' if overall else 'FAIL'}")

    return nmi_noise, nmi_null


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    run(device=args.device)
