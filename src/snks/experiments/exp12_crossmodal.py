"""Experiment 12: Кросс-модальное связывание.

50 пар (image, text) → совместное обучение.
Метрика: cross_activation_ratio = mean(visual_nodes | paired_text) / mean(visual_nodes | random_text).
Gate: cross_activation_ratio > 2.0.
"""

from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding="utf-8")

import random

import torch

from snks.daf.types import DafConfig, EncoderConfig, PipelineConfig, PredictionConfig, SKSConfig
from snks.pipeline.runner import Pipeline


# 10 категорий × 5 пар = 50 пар (синтетические: геометрические фигуры + подписи)
CATEGORIES = [
    ("bright circle", "A bright circular shape in the center"),
    ("dark square", "A dark square shape in the image"),
    ("bright triangle", "A bright triangular figure visible here"),
    ("dark circle", "A dark round circle on a white background"),
    ("bright square", "A bright square centered in the frame"),
    ("dark triangle", "A dark triangular shape is shown"),
    ("striped pattern", "An image with horizontal striped lines"),
    ("dotted pattern", "A dotted pattern of small points"),
    ("gradient bright", "A smooth gradient from dark to bright"),
    ("gradient dark", "A smooth gradient from bright to dark"),
]

RANDOM_TEXTS = [
    "The stock market fluctuated wildly today",
    "Scientists discovered a new exoplanet",
    "The recipe requires two cups of flour",
    "Traffic was heavy on the highway",
    "The concert was sold out last night",
    "A cold front is approaching from the north",
    "The library opened a new digital collection",
    "Engineers built a faster microprocessor",
    "The election results were announced at midnight",
    "A new species of fish was found in the deep ocean",
]


def make_synthetic_image(category_idx: int, variation: int, size: int = 32) -> torch.Tensor:
    """Generate a synthetic grayscale image for a category."""
    torch.manual_seed(category_idx * 100 + variation)
    img = torch.zeros(size, size)

    if category_idx == 0:  # bright circle
        cx, cy = size // 2, size // 2
        for i in range(size):
            for j in range(size):
                if (i - cx) ** 2 + (j - cy) ** 2 < (size // 3) ** 2:
                    img[i, j] = 0.9
    elif category_idx == 1:  # dark square
        img[size // 4: 3 * size // 4, size // 4: 3 * size // 4] = 0.1
        img += 0.5
        img = img.clamp(0, 1)
    elif category_idx == 2:  # bright triangle
        for i in range(size):
            for j in range(size):
                if j > i * 0.5 and j < size - i * 0.5:
                    img[i, j] = 0.8
    elif category_idx == 3:  # dark circle
        img += 0.8
        cx, cy = size // 2, size // 2
        for i in range(size):
            for j in range(size):
                if (i - cx) ** 2 + (j - cy) ** 2 < (size // 3) ** 2:
                    img[i, j] = 0.1
    elif category_idx == 4:  # bright square
        img[size // 4: 3 * size // 4, size // 4: 3 * size // 4] = 0.95
    elif category_idx == 5:  # dark triangle
        img += 0.7
        for i in range(size):
            for j in range(size):
                if j > i * 0.5 and j < size - i * 0.5:
                    img[i, j] = 0.05
        img = img.clamp(0, 1)
    elif category_idx == 6:  # striped
        for i in range(size):
            img[i, :] = 0.9 if i % 4 < 2 else 0.1
    elif category_idx == 7:  # dotted
        for i in range(0, size, 4):
            for j in range(0, size, 4):
                img[i, j] = 0.9
    elif category_idx == 8:  # gradient bright
        for j in range(size):
            img[:, j] = j / size
    elif category_idx == 9:  # gradient dark
        for j in range(size):
            img[:, j] = 1.0 - j / size

    # Add small variation
    img += torch.randn(size, size) * 0.02
    return img.clamp(0, 1)


def make_config(device: str = "cpu") -> PipelineConfig:
    return PipelineConfig(
        daf=DafConfig(
            num_nodes=5000,
            avg_degree=20,
            oscillator_model="fhn",
            coupling_strength=0.05,
            dt=0.01,
            noise_sigma=0.005,
            fhn_I_base=0.0,
            device=device,
        ),
        encoder=EncoderConfig(sdr_current_strength=1.0, image_size=32),
        sks=SKSConfig(
            top_k=2500,
            dbscan_eps=0.3,
            dbscan_min_samples=5,
            min_cluster_size=5,
            coherence_mode="rate",
        ),
        prediction=PredictionConfig(),
        steps_per_cycle=200,
        device=device,
    )


def run(device: str = "cpu") -> float:
    config = make_config(device)
    pipeline = Pipeline(config)
    n_nodes = config.daf.num_nodes

    # Шаг 1: Генерируем 50 пар (image, caption), 5 вариаций на категорию
    pairs = []
    for cat_idx, (_, caption) in enumerate(CATEGORIES):
        for variation in range(5):
            img = make_synthetic_image(cat_idx, variation, size=32)
            pairs.append((img, caption))

    # Шаг 2: Обучение — совместные циклы image + text
    for img, caption in pairs:
        pipeline.perception_cycle(image=img, text=caption)

    # Шаг 3: Записываем, какие узлы активировались при image-only для каждой категории
    visual_node_activations = []  # список (n_nodes,) float
    for cat_idx, (_, caption) in enumerate(CATEGORIES):
        img = make_synthetic_image(cat_idx, 0, size=32)
        # Запускаем image-only, записываем firing history
        pipeline.perception_cycle(image=img)
        fired = pipeline.engine.get_fired_history()
        if fired is not None:
            rate = fired.float().mean(dim=0).cpu()  # (n_nodes,)
        else:
            rate = torch.zeros(n_nodes)
        visual_node_activations.append(rate)

    visual_rates = torch.stack(visual_node_activations)  # (10, n_nodes)
    # Топ-20% по средней активации → visual_nodes
    mean_visual = visual_rates.mean(dim=0)  # (n_nodes,)
    threshold = mean_visual.quantile(0.8)
    visual_mask = mean_visual > threshold  # (n_nodes,) bool

    # Шаг 4: Тест — текст из пары → activation visual_nodes
    test_activations = []
    for cat_idx, (_, caption) in enumerate(CATEGORIES):
        pipeline.perception_cycle(text=caption)
        fired = pipeline.engine.get_fired_history()
        if fired is not None:
            rate = fired.float().mean(dim=0).cpu()
        else:
            rate = torch.zeros(n_nodes)
        test_activations.append(rate[visual_mask].mean().item())

    # Шаг 5: Контроль — случайные тексты → activation visual_nodes
    rng = random.Random(42)
    control_activations = []
    for rtext in RANDOM_TEXTS:
        pipeline.perception_cycle(text=rtext)
        fired = pipeline.engine.get_fired_history()
        if fired is not None:
            rate = fired.float().mean(dim=0).cpu()
        else:
            rate = torch.zeros(n_nodes)
        control_activations.append(rate[visual_mask].mean().item())
    # Pad to 50 by repeating random texts
    for _ in range(40):
        rtext = rng.choice(RANDOM_TEXTS)
        pipeline.perception_cycle(text=rtext)
        fired = pipeline.engine.get_fired_history()
        if fired is not None:
            rate = fired.float().mean(dim=0).cpu()
        else:
            rate = torch.zeros(n_nodes)
        control_activations.append(rate[visual_mask].mean().item())

    mean_test = sum(test_activations) / len(test_activations)
    mean_control = sum(control_activations) / len(control_activations)

    if mean_control > 0:
        ratio = mean_test / mean_control
    else:
        ratio = float("inf")

    print("Exp 12: Кросс-модальное связывание")
    print(f"  Пар: {len(pairs)}, visual_nodes: {int(visual_mask.sum())}")
    print(f"  mean_test:    {mean_test:.6f}")
    print(f"  mean_control: {mean_control:.6f}")
    print(f"  cross_activation_ratio: {ratio:.4f}")
    print(f"  Gate (ratio > 2.0): {'PASS' if ratio > 2.0 else 'FAIL'}")
    return ratio


if __name__ == "__main__":
    run()
