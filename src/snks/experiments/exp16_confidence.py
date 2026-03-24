"""Experiment 16: Confidence calibration.

Focused: 5 категорий × 2 вариации = 10 изображений.
Noise: 10 тензоров torch.rand(32, 32), seed 0..9.

Gate: mean_confidence(focused) / mean_confidence(noise) > 1.5
"""

from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding="utf-8")

import random
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
    # Use larger network on GPU (AMD/CUDA), smaller on CPU
    if device == "cpu":
        num_nodes = 5_000
        avg_degree = 20
        top_k = 2_500
    else:
        num_nodes = 50_000
        avg_degree = 50
        top_k = 5_000

    return PipelineConfig(
        daf=DafConfig(
            num_nodes=num_nodes,
            avg_degree=avg_degree,
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
            top_k=top_k,
            dbscan_eps=0.3,
            dbscan_min_samples=5,
            min_cluster_size=5,
            coherence_mode="cofiring",  # multi-cluster detection for meaningful dominance
        ),
        prediction=PredictionConfig(),
        gws=GWSConfig(enabled=True, w_size=1.0),
        metacog=MetacogConfig(enabled=True, policy="null"),
        steps_per_cycle=50,
        device=device,
    )


def measure_confidence(
    pipeline: Pipeline,
    image: torch.Tensor,
    n_warmup: int = 5,
    n_cycles: int = 3,
) -> tuple[float, float, float, float]:
    """Measure confidence after warm-up on the stimulus.

    Resets prev_winner before warm-up so stability is not contaminated
    by the previous stimulus.

    Returns:
        (mean_confidence, mean_dominance, mean_stability, mean_pred_error)
    """
    # Reset stability state to isolate this stimulus
    pipeline.metacog._prev_winner_nodes = None

    # Warm-up: let system settle on this image
    for _ in range(n_warmup):
        pipeline.perception_cycle(image=image)

    # Measure
    confidences: list[float] = []
    dominances: list[float] = []
    stabilities: list[float] = []
    pred_errors: list[float] = []

    for _ in range(n_cycles):
        result = pipeline.perception_cycle(image=image)
        if result.metacog is not None:
            confidences.append(result.metacog.confidence)
            dominances.append(result.metacog.dominance)
            stabilities.append(result.metacog.stability)
            pred_errors.append(result.metacog.pred_error)

    def mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    return mean(confidences), mean(dominances), mean(stabilities), mean(pred_errors)


def run(device: str = "cpu") -> float:
    """Run Exp 16: Confidence calibration.

    Returns:
        ratio = mean_confidence(focused) / mean_confidence(noise).
    """
    GATE = 1.5
    N_PRETRAIN_EPOCHS = 5   # interleaved epochs
    N_WARMUP = 5
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

    # Предобучение: interleaved эпохи по focused стимулам
    print("  Предобучение (interleaved)...")
    rng = random.Random(42)
    for epoch in range(N_PRETRAIN_EPOCHS):
        order = list(range(len(focused_images)))
        rng.shuffle(order)
        for idx in order:
            pipeline.perception_cycle(image=focused_images[idx])

    # Измерение confidence на focused стимулах
    print("  Измерение focused...")
    focused_confidences: list[float] = []
    focused_dominances: list[float] = []
    for img in focused_images:
        conf, dom, stab, pe = measure_confidence(pipeline, img, n_warmup=N_WARMUP, n_cycles=N_MEASURE_CYCLES)
        focused_confidences.append(conf)
        focused_dominances.append(dom)

    # Измерение confidence на noise стимулах
    print("  Измерение noise...")
    noise_confidences: list[float] = []
    noise_dominances: list[float] = []
    for img in noise_images:
        conf, dom, stab, pe = measure_confidence(pipeline, img, n_warmup=N_WARMUP, n_cycles=N_MEASURE_CYCLES)
        noise_confidences.append(conf)
        noise_dominances.append(dom)

    def mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    mean_focused = mean(focused_confidences)
    mean_noise = mean(noise_confidences)

    if mean_noise > 0.0:
        ratio = mean_focused / mean_noise
    else:
        ratio = float("inf") if mean_focused > 0.0 else 1.0

    passed = ratio > GATE

    print("Exp 16: Confidence calibration")
    print(f"  mean_confidence(focused) = {mean_focused:.3f}  dominance={mean(focused_dominances):.3f}")
    print(f"  mean_confidence(noise)   = {mean_noise:.3f}  dominance={mean(noise_dominances):.3f}")
    print(f"  ratio = {ratio:.3f} (gate > {GATE})")
    print(f"  {'PASS' if passed else 'FAIL'}")

    return ratio


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    else:
        device = args.device

    print(f"  device={device}")
    run(device=device)
