"""Experiment 12: Кросс-модальное связывание.

50 пар (image, caption) → совместное обучение.
visual_nodes определяются по реальной firing-rate специфичности к изображениям.
Метрика: cross_activation_ratio = mean(visual_nodes | paired_text) / mean(visual_nodes | random_text).
Gate: cross_activation_ratio > 2.0.
"""

from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding="utf-8")

import torch

from snks.daf.types import DafConfig, EncoderConfig, PipelineConfig, PredictionConfig, SKSConfig
from snks.pipeline.runner import Pipeline

CATEGORIES = [
    ("bright circle", "A bright circular shape in the center of the image"),
    ("dark square", "A dark square shape visible in the frame"),
    ("bright triangle", "A bright triangular figure is shown here"),
    ("dark circle", "A dark round circle on a white background"),
    ("bright square", "A bright square centered in the frame"),
    ("dark triangle", "A dark triangular shape is displayed"),
    ("striped pattern", "Horizontal striped lines fill the image"),
    ("dotted pattern", "Small dots are arranged in a grid pattern"),
    ("gradient bright", "The image brightens from left to right"),
    ("gradient dark", "The image darkens from left to right"),
]

RANDOM_TEXTS = [
    "The stock market rose sharply this morning",
    "Scientists discovered a new species of bird",
    "The recipe calls for two cups of sugar",
    "Heavy traffic was reported on the highway",
    "The concert tickets sold out in minutes",
    "A cold front is moving in from the north",
    "The library expanded its digital collection",
    "Engineers designed a faster processor chip",
    "Election results were announced after midnight",
    "A rare fish was caught in the Pacific Ocean",
    "The train arrived three minutes early today",
    "Researchers published findings on sleep patterns",
    "The bakery opened a new branch downtown",
    "Weather forecasts predict rain this weekend",
    "Athletes trained hard for the upcoming season",
    "The museum acquired a new painting collection",
    "Astronomers observed a distant galaxy cluster",
    "The hospital introduced new treatment options",
    "Students gathered for the annual science fair",
    "The city council approved a new budget plan",
]


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
        img[size//4:3*size//4, size//4:3*size//4] = 0.1
        img += 0.5; img = img.clamp(0, 1)
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
        img[size//4:3*size//4, size//4:3*size//4] = 0.95
    elif category_idx == 5:
        img += 0.7
        for i in range(size):
            for j in range(size):
                if j > i * 0.5 and j < size - i * 0.5:
                    img[i, j] = 0.05
        img = img.clamp(0, 1)
    elif category_idx == 6:
        for i in range(size):
            img[i, :] = 0.9 if i % 4 < 2 else 0.1
    elif category_idx == 7:
        for i in range(0, size, 4):
            for j in range(0, size, 4):
                img[i, j] = 0.9
    elif category_idx == 8:
        for j in range(size):
            img[:, j] = j / size
    elif category_idx == 9:
        for j in range(size):
            img[:, j] = 1.0 - j / size

    img += torch.randn(size, size) * 0.02
    return img.clamp(0, 1)


def get_mean_rate(pipeline: Pipeline, n: int, **kw) -> torch.Tensor:
    """Среднее firing rate по n запускам."""
    total = torch.zeros(pipeline.engine.config.num_nodes)
    for _ in range(n):
        pipeline.perception_cycle(**kw)
        fired = pipeline.engine.get_fired_history()
        if fired is not None:
            total += fired.float().mean(dim=0).cpu()
    return total / n


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
        steps_per_cycle=200,
        device=device,
    )


def run(device: str = "cpu", n_train_reps: int = 10) -> float:
    config = make_config(device)
    pipeline = Pipeline(config)
    n_nodes = config.daf.num_nodes

    # Шаг 1: Генерируем 50 пар (image, caption): 10 категорий × 5 вариаций
    pairs: list[tuple[torch.Tensor, str]] = []
    for cat_idx, (_, caption) in enumerate(CATEGORIES):
        for variation in range(5):
            img = make_synthetic_image(cat_idx, variation, size=32)
            pairs.append((img, caption))

    # Шаг 2: Определяем visual_nodes по реальной firing-rate специфичности.
    # visual_nodes = узлы, которые стреляют значительно больше при image, чем при случайном тексте.
    print("  Определяем visual_nodes по firing-rate специфичности...")
    r_img_mean = torch.zeros(n_nodes)
    for cat_idx, _ in enumerate(CATEGORIES):
        img = make_synthetic_image(cat_idx, 0, size=32)
        r_img_mean += get_mean_rate(pipeline, n=2, image=img)
    r_img_mean /= len(CATEGORIES)

    r_rnd_baseline = torch.zeros(n_nodes)
    for rtext in RANDOM_TEXTS[:10]:
        r_rnd_baseline += get_mean_rate(pipeline, n=1, text=rtext)
    r_rnd_baseline /= 10

    # Специфичность: насколько сильнее узел реагирует на image vs случайный текст
    specificity = r_img_mean - r_rnd_baseline
    threshold = specificity.quantile(0.90)  # top 10% — истинно визуальные узлы
    visual_mask = specificity > threshold
    n_visual = int(visual_mask.sum().item())
    print(f"  visual_nodes: {n_visual} (image_rate={r_img_mean[visual_mask].mean():.4f}, rnd_rate={r_rnd_baseline[visual_mask].mean():.4f})")

    # Шаг 3: Baseline ratio ДО обучения
    r_cap_pre = torch.zeros(n_nodes)
    for cat_idx, (_, caption) in enumerate(CATEGORIES):
        r_cap_pre += get_mean_rate(pipeline, n=2, text=caption)
    r_cap_pre /= len(CATEGORIES)

    r_rnd_pre = torch.zeros(n_nodes)
    for rtext in RANDOM_TEXTS:
        r_rnd_pre += get_mean_rate(pipeline, n=1, text=rtext)
    r_rnd_pre /= len(RANDOM_TEXTS)

    pre_ratio = r_cap_pre[visual_mask].mean() / r_rnd_pre[visual_mask].mean()
    print(f"  Ratio ДО обучения: {pre_ratio:.4f}")

    # Шаг 4: Совместное обучение (n_train_reps × 50 пар)
    total_cycles = 0
    for _ in range(n_train_reps):
        perm = torch.randperm(len(pairs)).tolist()
        for idx in perm:
            img, caption = pairs[idx]
            pipeline.perception_cycle(image=img, text=caption)
            total_cycles += 1
    print(f"  Обучение: {total_cycles} совместных циклов")

    # Шаг 5: Ratio ПОСЛЕ обучения
    r_cap_post = torch.zeros(n_nodes)
    for cat_idx, (_, caption) in enumerate(CATEGORIES):
        r_cap_post += get_mean_rate(pipeline, n=2, text=caption)
    r_cap_post /= len(CATEGORIES)

    r_rnd_post = torch.zeros(n_nodes)
    for rtext in RANDOM_TEXTS:
        r_rnd_post += get_mean_rate(pipeline, n=1, text=rtext)
    r_rnd_post /= len(RANDOM_TEXTS)

    ratio = r_cap_post[visual_mask].mean() / r_rnd_post[visual_mask].mean()

    print("Exp 12: Кросс-модальное связывание")
    print(f"  visual_nodes (firing-based): {n_visual}")
    print(f"  paired_text activation:  {r_cap_post[visual_mask].mean():.5f}")
    print(f"  random_text activation:  {r_rnd_post[visual_mask].mean():.5f}")
    print(f"  cross_activation_ratio:  {ratio:.4f}")
    print(f"  Gate (ratio > 2.0): {'PASS' if ratio > 2.0 else 'FAIL'}")
    return ratio


if __name__ == "__main__":
    run()
