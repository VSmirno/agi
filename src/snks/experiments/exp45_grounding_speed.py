"""Experiment 45: Grounding speed with zonal DAF.

Stage 19 validation: how many co-activations are needed for cross-modal ratio
to exceed 2.0. Tests both Config A and Config B.

Metric: number of co-activations until ratio > 2.0.
Gate: < 20 co-activations.
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
from snks.experiments.exp44_crossmodal_recall import (
    CATEGORIES, RANDOM_TEXTS, make_synthetic_image, ZONES_A, ZONES_B,
    measure_visual_zone_activation, make_config,
)


def run_variant(
    zones: dict[str, ZoneConfig],
    variant_name: str,
    device: str = "cpu",
    max_reps: int = 30,
    n_variations: int = 5,
) -> dict:
    """Measure how many co-activation reps needed for ratio > 2.0."""
    print(f"\n{'='*60}")
    print(f"Exp 45 ({variant_name}): Grounding speed")
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

    reps_to_threshold = max_reps  # default if never reached

    for rep in range(1, max_reps + 1):
        # One round of co-activation
        for img, text in pairs:
            pipeline.perception_cycle(image=img, text=text)

        # Measure ratio every rep
        paired_rates = []
        for text in CATEGORIES:
            rate = measure_visual_zone_activation(pipeline, text, vis_zone, n_runs=2)
            paired_rates.append(rate)

        random_rates = []
        for text in RANDOM_TEXTS[:5]:  # sample 5 for speed
            rate = measure_visual_zone_activation(pipeline, text, vis_zone, n_runs=2)
            random_rates.append(rate)

        mean_paired = sum(paired_rates) / len(paired_rates) if paired_rates else 0
        mean_random = sum(random_rates) / len(random_rates) if random_rates else 1e-9
        ratio = mean_paired / max(mean_random, 1e-9)

        print(f"  Rep {rep}: ratio = {ratio:.3f}", end="")
        if ratio > 2.0:
            reps_to_threshold = rep
            print(" ← THRESHOLD REACHED")
            break
        print()

    print(f"\nResults ({variant_name}):")
    print(f"  Reps to threshold: {reps_to_threshold}")
    print(f"  Final ratio: {ratio:.3f}")
    print(f"  Gate (< 20 reps): {'PASS' if reps_to_threshold < 20 else 'FAIL'}")

    return {
        "variant": variant_name,
        "reps_to_threshold": reps_to_threshold,
        "final_ratio": ratio,
        "pass": reps_to_threshold < 20,
    }


def run(device: str = "cpu") -> dict:
    """Run exp45 for both zone configs."""
    result_a = run_variant(ZONES_A, "A (no convergence)", device)
    result_b = run_variant(ZONES_B, "B (with convergence)", device)

    print(f"\n{'='*60}")
    print("Exp 45 — Ablation summary:")
    print(f"  Config A: {result_a['reps_to_threshold']} reps ({'PASS' if result_a['pass'] else 'FAIL'})")
    print(f"  Config B: {result_b['reps_to_threshold']} reps ({'PASS' if result_b['pass'] else 'FAIL'})")

    # Decision rule: if speed_B < 0.8 * speed_A → choose B
    if result_b["reps_to_threshold"] < 0.8 * result_a["reps_to_threshold"]:
        chosen = "B"
        reason = f"speed_B ({result_b['reps_to_threshold']}) < 0.8 × speed_A ({0.8 * result_a['reps_to_threshold']:.1f})"
    else:
        chosen = "A"
        reason = "convergence zone did not significantly speed up grounding"

    print(f"  Chosen config: {chosen} ({reason})")

    return {
        "config_a": result_a,
        "config_b": result_b,
        "chosen": chosen,
    }


# --- pytest entry point ---

def test_exp45_grounding_speed():
    results = run("cpu")
    assert results["config_a"]["pass"], f"Config A: {results['config_a']['reps_to_threshold']} reps >= 20"
    assert results["config_b"]["pass"], f"Config B: {results['config_b']['reps_to_threshold']} reps >= 20"


if __name__ == "__main__":
    run("cpu")
