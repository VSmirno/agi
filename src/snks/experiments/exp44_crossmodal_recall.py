"""Experiment 44: Cross-modal recall with zonal DAF + complementary priming.

Stage 19 validation: after co-activation (image + word), presenting only the word
activates the visual zone via top-down priming through GroundingMap.

Mechanism: during co-activation, Pipeline registers visual SDR in GroundingMap.
On text-only recall, priming injects a fraction (priming_strength) of the learned
visual SDR into the visual zone, enabling cross-modal activation.

Metric: cross_modal_ratio = mean_firing(visual_zone | paired_word) /
                            mean_firing(visual_zone | random_word)
Gate: cross_modal_ratio > 2.0 for both configs.
"""

from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding="utf-8")

import torch

from snks.daf.types import (
    DafConfig, EncoderConfig, PipelineConfig, PredictionConfig,
    SKSConfig, ZoneConfig,
)
from snks.pipeline.runner import Pipeline

# --- Synthetic data (same categories as exp12) ---

CATEGORIES = [
    "bright circle", "dark square", "bright triangle", "dark circle",
    "bright square", "dark triangle", "striped pattern", "dotted pattern",
    "gradient bright", "gradient dark",
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
]


def make_synthetic_image(category_idx: int, variation: int, size: int = 32) -> torch.Tensor:
    """Generate synthetic image for category."""
    torch.manual_seed(category_idx * 100 + variation)
    img = torch.zeros(size, size)

    if category_idx == 0:  # bright circle
        cx, cy = size // 2, size // 2
        for i in range(size):
            for j in range(size):
                if (i - cx) ** 2 + (j - cy) ** 2 < (size // 3) ** 2:
                    img[i, j] = 0.9
    elif category_idx == 1:  # dark square
        img[size//4:3*size//4, size//4:3*size//4] = 0.1
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
        img[size//4:3*size//4, size//4:3*size//4] = 0.95
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

    img += torch.randn(size, size) * 0.02
    return img.clamp(0, 1)


def make_config(
    zones: dict[str, ZoneConfig],
    device: str = "cpu",
) -> PipelineConfig:
    total = sum(z.size for z in zones.values())
    return PipelineConfig(
        daf=DafConfig(
            num_nodes=total,
            avg_degree=20,
            inter_zone_avg_degree=30,
            zones=zones,
            oscillator_model="fhn",
            coupling_strength=0.2,
            dt=0.01,
            noise_sigma=0.003,
            fhn_I_base=0.0,
            stdp_a_plus=0.08,
            device=device,
            disable_csr=True,
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
        priming_strength=0.3,
    )


ZONES_A = {
    "visual": ZoneConfig(start=0, size=3000),
    "linguistic": ZoneConfig(start=3000, size=2000),
}

ZONES_B = {
    "visual": ZoneConfig(start=0, size=2200),
    "linguistic": ZoneConfig(start=2200, size=1800),
    "convergence": ZoneConfig(start=4000, size=1000),
}


def measure_visual_zone_activation(
    pipeline: Pipeline, text: str, vis_zone: ZoneConfig, n_runs: int = 3,
) -> float:
    """Mean firing rate in visual zone when only text is presented."""
    total = 0.0
    for _ in range(n_runs):
        pipeline.perception_cycle(text=text)
        fired = pipeline.engine.get_fired_history()
        if fired is not None:
            vis_rate = fired[:, vis_zone.start:vis_zone.start + vis_zone.size].float().mean()
            total += vis_rate.item()
    return total / n_runs


def run_variant(
    zones: dict[str, ZoneConfig],
    variant_name: str,
    device: str = "cpu",
    n_train_reps: int = 15,
    n_variations: int = 5,
) -> dict:
    """Run cross-modal recall experiment for one zone config."""
    print(f"\n{'='*60}")
    print(f"Exp 44 ({variant_name}): Cross-modal recall")
    print(f"{'='*60}")

    config = make_config(zones, device)
    pipeline = Pipeline(config)
    vis_zone = zones["visual"]

    # Generate pairs
    pairs = []
    for cat_idx in range(len(CATEGORIES)):
        for var in range(n_variations):
            img = make_synthetic_image(cat_idx, var)
            pairs.append((img, CATEGORIES[cat_idx]))

    # Co-activation training
    print(f"Training: {n_train_reps} reps × {len(pairs)} pairs = {n_train_reps * len(pairs)} cycles")
    for rep in range(n_train_reps):
        for img, text in pairs:
            pipeline.perception_cycle(image=img, text=text)
        print(f"  Rep {rep + 1}/{n_train_reps} done")

    # Measure: paired text → visual zone activation
    paired_rates = []
    for cat_idx, text in enumerate(CATEGORIES):
        rate = measure_visual_zone_activation(pipeline, text, vis_zone)
        paired_rates.append(rate)

    # Measure: random text → visual zone activation
    random_rates = []
    for text in RANDOM_TEXTS:
        rate = measure_visual_zone_activation(pipeline, text, vis_zone)
        random_rates.append(rate)

    mean_paired = sum(paired_rates) / len(paired_rates) if paired_rates else 0
    mean_random = sum(random_rates) / len(random_rates) if random_rates else 1e-9

    ratio = mean_paired / max(mean_random, 1e-9)

    print(f"\nResults ({variant_name}):")
    print(f"  Mean visual activation (paired text):  {mean_paired:.6f}")
    print(f"  Mean visual activation (random text):  {mean_random:.6f}")
    print(f"  Cross-modal ratio: {ratio:.3f}")
    print(f"  Gate (> 2.0): {'PASS' if ratio > 2.0 else 'FAIL'}")

    return {
        "variant": variant_name,
        "mean_paired": mean_paired,
        "mean_random": mean_random,
        "ratio": ratio,
        "pass": ratio > 2.0,
    }


def run(device: str = "cpu") -> dict:
    """Run exp44 for both zone configs."""
    result_a = run_variant(ZONES_A, "A (no convergence)", device)
    result_b = run_variant(ZONES_B, "B (with convergence)", device)

    print(f"\n{'='*60}")
    print("Exp 44 — Ablation summary:")
    print(f"  Config A ratio: {result_a['ratio']:.3f} ({'PASS' if result_a['pass'] else 'FAIL'})")
    print(f"  Config B ratio: {result_b['ratio']:.3f} ({'PASS' if result_b['pass'] else 'FAIL'})")

    # Decision rule from spec
    if result_b["ratio"] > 1.2 * result_a["ratio"]:
        chosen = "B"
        reason = f"ratio_B ({result_b['ratio']:.3f}) > 1.2 × ratio_A ({1.2 * result_a['ratio']:.3f})"
    else:
        chosen = "A"
        reason = f"ratio_B ({result_b['ratio']:.3f}) <= 1.2 × ratio_A ({1.2 * result_a['ratio']:.3f})"

    print(f"  Chosen config: {chosen} ({reason})")

    return {
        "config_a": result_a,
        "config_b": result_b,
        "chosen": chosen,
    }


# --- pytest entry point ---

def test_exp44_crossmodal_recall():
    results = run("cpu")
    assert results["config_a"]["pass"], f"Config A ratio {results['config_a']['ratio']:.3f} < 2.0"
    assert results["config_b"]["pass"], f"Config B ratio {results['config_b']['ratio']:.3f} < 2.0"


if __name__ == "__main__":
    run("cpu")
