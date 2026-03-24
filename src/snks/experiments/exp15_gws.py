"""Experiment 15: GWS winner stability.

5 синтетических категорий × 10 повторов.
Для каждого повтора t >= 2:
    stability_t = |winner_t.nodes ∩ winner_{t-1}.nodes| / winner_t.size
mean_stability = mean(stability_t) по всем категориям и повторам t >= 2

Gate: mean_stability > 0.7
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


# Используем make_synthetic_image из exp12
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


def run(device: str = "cpu") -> float:
    """Run Exp 15: GWS winner stability.

    Returns:
        mean_stability across all categories and repeats t >= 2.
    """
    N_CATEGORIES = 5
    N_REPEATS = 10
    N_WARMUP = 5
    GATE = 0.7

    config = make_config(device)
    img = make_synthetic_image(0, 0)  # dummy, replaced per category

    all_stabilities: list[float] = []

    for cat_idx in range(N_CATEGORIES):
        # Создаём новый pipeline для каждой категории (сброс состояния между категориями)
        pipeline = Pipeline(config)
        cat_img = make_synthetic_image(cat_idx, variation=0)

        # Warm-up циклы (состояние не сбрасывается между повторами одной категории)
        for _ in range(N_WARMUP):
            pipeline.perception_cycle(image=cat_img)

        # 10 повторов, считаем stability начиная с t=2 (повтор 1)
        prev_winner_nodes: set[int] | None = None
        for t in range(N_REPEATS):
            result = pipeline.perception_cycle(image=cat_img)
            gws = result.gws

            if gws is not None and prev_winner_nodes is not None and gws.winner_size > 0:
                intersection = len(gws.winner_nodes & prev_winner_nodes)
                stability_t = intersection / gws.winner_size
                all_stabilities.append(stability_t)

            if gws is not None:
                prev_winner_nodes = set(gws.winner_nodes)
            else:
                prev_winner_nodes = None

    if all_stabilities:
        mean_stability = sum(all_stabilities) / len(all_stabilities)
    else:
        mean_stability = 0.0

    passed = mean_stability > GATE

    print("Exp 15: GWS winner stability")
    print(f"  samples collected: {len(all_stabilities)}")
    print(f"  mean_stability = {mean_stability:.3f} (gate > {GATE})")
    print(f"  {'PASS' if passed else 'FAIL'}")

    return mean_stability


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    run(device=args.device)
