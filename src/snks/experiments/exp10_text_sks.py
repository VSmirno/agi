"""Experiment 10: Текстовое восприятие.

3 категории × 20 предложений (животные, еда, техника).
Кластеризация SDR (k-means k=3) → NMI vs истинные категории.
Gate: NMI > 0.6.
"""

from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding="utf-8")

import torch
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score

from snks.daf.types import EncoderConfig
from snks.encoder.text_encoder import TextEncoder

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


def run() -> float:
    config = EncoderConfig()
    encoder = TextEncoder(config)

    sdrs = []
    labels = []
    for label_idx, (category, sentences) in enumerate(SENTENCES.items()):
        for sentence in sentences:
            sdr = encoder.encode(sentence)
            sdrs.append(sdr.cpu().numpy())
            labels.append(label_idx)

    import numpy as np
    X = np.array(sdrs)
    true_labels = np.array(labels)

    km = KMeans(n_clusters=3, n_init=20, random_state=42)
    pred_labels = km.fit_predict(X)

    nmi = normalized_mutual_info_score(true_labels, pred_labels)

    print("Exp 10: Текстовое восприятие")
    print(f"  Категорий: 3, предложений: {len(labels)}")
    print(f"  NMI: {nmi:.4f}")
    print(f"  Gate (NMI > 0.6): {'PASS' if nmi > 0.6 else 'FAIL'}")
    return nmi


if __name__ == "__main__":
    run()
