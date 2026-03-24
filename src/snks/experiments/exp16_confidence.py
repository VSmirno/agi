"""Experiment 16: Confidence calibration.

Focused: 5 категорий × 2 вариации = 10 изображений.
Noise: 10 тензоров torch.rand(32, 32), seed 0..9.

Gate: mean_confidence(focused) / mean_confidence(noise) > 1.5
"""

from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding="utf-8")

import torch

from snks.daf.types import (
    DafConfig,
    EncoderConfig,
    GWSConfig,
    MetacogConfig,
    PipelineConfig,
    PredictionConfig,
    SKSConfig,
)
from snks.pipeline.runner import Pipeline


def make_synthetic_image(category_idx: int, variation: int, size: int = 32) -> torch.Tensor:
    torch.manual_seed(category_idx * 100 + variation)
    img = torch.zeros(size, size)

    if category_idx == 0:
        cx, cy = size // 2, size // 2
        for i in range(size):
            for j in range(size):
                if (i - cx) ** 2 + (j - cy) ** 2 < (size // 3) ** 2:
                    img[i, j] = 0.9
    elif category_idx == 1:
        img[size // 4:3 * size // 4, size // 4:3 * size // 4] = 0.1
        img += 0.5
        img = img.clamp(0, 1)
    elif category_idx == 2:
        for i in range(size):
            for j in range(size):
                if j > i * 0.5 and j < size - i * 0.5:
                    img[i, j] = 0.8
    elif category_idx == 3:
        img += 0.8
        cx, cy = size // 2, size // 2
        for i in range(size):
            for j in range(size):
                if (i - cx) ** 2 + (j - cy) ** 2 < (size // 3) ** 2:
                    img[i, j] = 0.1
    elif category_idx == 4:
        img[size // 4:3 * size // 4, size // 4:3 * size // 4] = 0.95

    img += torch.randn(size, size) * 0.02
    return img.clamp(0, 1)


def make_noise_image(seed: int, size: int = 32) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.rand(size, size).clamp(0, 1)


def make_config(device: str = "cpu") -> PipelineConfig:
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
        metacog=MetacogConfig(enabled=True, policy="null"),
        steps_per_cycle=50,
        device=device,
    )


def measure_confidence(pipeline: Pipeline, image: torch.Tensor, n_cycles: int = 3) -> float:
    """Среднее confidence за n_cycles циклов для одного стимула."""
    confidences: list[float] = []
    for _ in range(n_cycles):
        result = pipeline.perception_cycle(image=image)
        if result.metacog is not None:
            confidences.append(result.metacog.confidence)
    return sum(confidences) / len(confidences) if confidences else 0.0


def run(device: str = "cpu") -> float:
    """Run Exp 16: Confidence calibration.

    Returns:
        ratio = mean_confidence(focused) / mean_confidence(noise).
    """
    GATE = 1.5
    N_PRETRAIN = 50
    N_MEASURE_CYCLES = 3

    config = make_config(device)
    pipeline = Pipeline(config)

    # Датасет: focused стимулы
    focused_images: list[torch.Tensor] = [
        make_synthetic_image(cat_idx, variation)
        for cat_idx in range(5)
        for variation in [0, 1]
    ]  # 10 изображений

    # Noise стимулы
    noise_images: list[torch.Tensor] = [
        make_noise_image(seed)
        for seed in range(10)
    ]

    # Предобучение: 20 циклов каждого focused стимула
    print("  Предобучение...")
    for img in focused_images:
        for _ in range(N_PRETRAIN):
            pipeline.perception_cycle(image=img)

    # Измерение confidence на focused стимулах
    focused_confidences: list[float] = []
    for img in focused_images:
        c = measure_confidence(pipeline, img, n_cycles=N_MEASURE_CYCLES)
        focused_confidences.append(c)

    # Измерение confidence на noise стимулах
    noise_confidences: list[float] = []
    for img in noise_images:
        c = measure_confidence(pipeline, img, n_cycles=N_MEASURE_CYCLES)
        noise_confidences.append(c)

    mean_focused = sum(focused_confidences) / len(focused_confidences) if focused_confidences else 0.0
    mean_noise = sum(noise_confidences) / len(noise_confidences) if noise_confidences else 0.0

    if mean_noise > 0.0:
        ratio = mean_focused / mean_noise
    else:
        ratio = float("inf") if mean_focused > 0.0 else 1.0

    passed = ratio > GATE

    print("Exp 16: Confidence calibration")
    print(f"  mean_confidence(focused) = {mean_focused:.3f}")
    print(f"  mean_confidence(noise)   = {mean_noise:.3f}")
    print(f"  ratio = {ratio:.3f} (gate > {GATE})")
    print(f"  {'PASS' if passed else 'FAIL'}")

    return ratio


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    run(device=args.device)
