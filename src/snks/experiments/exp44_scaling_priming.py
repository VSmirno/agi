"""Exp 44 scaling study WITH priming: cross-modal recall quality vs network size.

Compares priming ON vs OFF at N=10K, 15K, 25K, 50K.
Priming = top-down injection of learned visual SDR at fraction of full current.
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import torch

from snks.daf.types import (
    DafConfig, EncoderConfig, PipelineConfig, PredictionConfig,
    SKSConfig, ZoneConfig,
)
from snks.pipeline.runner import Pipeline
from snks.experiments.exp44_crossmodal_recall import (
    make_synthetic_image, CATEGORIES, RANDOM_TEXTS,
)

SEED = 42


def make_zones(n: int) -> dict[str, ZoneConfig]:
    vis = int(n * 0.6)
    ling = n - vis
    return {
        "visual": ZoneConfig(start=0, size=vis),
        "linguistic": ZoneConfig(start=vis, size=ling),
    }


def make_config(n: int, device: str, priming_strength: float = 0.0) -> PipelineConfig:
    zones = make_zones(n)
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
            top_k=n // 2,
            dbscan_eps=0.3,
            dbscan_min_samples=5,
            min_cluster_size=5,
            coherence_mode="rate",
        ),
        prediction=PredictionConfig(),
        steps_per_cycle=200,
        device=device,
        priming_strength=priming_strength,
    )


def measure_visual_activation(pipeline, text, vis_zone, n_runs=3):
    total = 0.0
    for _ in range(n_runs):
        pipeline.perception_cycle(text=text)
        fired = pipeline.engine.get_fired_history()
        if fired is not None:
            vis_rate = fired[:, vis_zone.start:vis_zone.start + vis_zone.size].float().mean()
            total += vis_rate.item()
    return total / n_runs


def run_scale(
    n: int, device: str, priming_strength: float,
    n_train_reps: int = 15, n_variations: int = 5,
) -> dict:
    torch.manual_seed(SEED)
    if device == "cuda":
        torch.cuda.manual_seed(SEED)
        torch.cuda.reset_peak_memory_stats()

    zones = make_zones(n)
    vis_zone = zones["visual"]
    label = f"N={n//1000}K priming={priming_strength}"

    print(f"\n{'='*60}")
    print(f"{label}")
    print(f"{'='*60}")

    t_init = time.time()
    config = make_config(n, device, priming_strength)
    pipeline = Pipeline(config)
    init_time = time.time() - t_init

    pairs = []
    for cat_idx in range(len(CATEGORIES)):
        for var in range(n_variations):
            img = make_synthetic_image(cat_idx, var)
            pairs.append((img, CATEGORIES[cat_idx]))

    total_train = n_train_reps * len(pairs)
    print(f"Training: {n_train_reps} reps x {len(pairs)} pairs = {total_train} cycles")
    t_train = time.time()
    for rep in range(n_train_reps):
        for img, text in pairs:
            pipeline.perception_cycle(image=img, text=text)
        elapsed_rep = time.time() - t_train
        eta = elapsed_rep / (rep + 1) * (n_train_reps - rep - 1)
        print(f"  Rep {rep+1}/{n_train_reps}  [{elapsed_rep:.1f}s, ~{eta:.0f}s left]")
    train_time = time.time() - t_train

    # Check grounding map state
    gm_size = pipeline.grounding_map.vocab_size
    gm_visual = len(pipeline.grounding_map._word_to_visual_sdr)
    print(f"  GroundingMap: {gm_size} words, {gm_visual} visual SDRs")

    # Measure: paired text -> visual zone activation
    t_test = time.time()
    paired_rates = []
    for cat_idx, text in enumerate(CATEGORIES):
        rate = measure_visual_activation(pipeline, text, vis_zone)
        paired_rates.append(rate)

    random_rates = []
    for text in RANDOM_TEXTS:
        rate = measure_visual_activation(pipeline, text, vis_zone)
        random_rates.append(rate)
    test_time = time.time() - t_test

    mean_paired = sum(paired_rates) / len(paired_rates)
    mean_random = sum(random_rates) / len(random_rates)
    ratio = mean_paired / max(mean_random, 1e-9)

    peak_mb = torch.cuda.max_memory_allocated() / 1024**2 if device == "cuda" else 0

    result = {
        "N": n,
        "priming_strength": priming_strength,
        "mean_paired": round(mean_paired, 6),
        "mean_random": round(mean_random, 6),
        "cross_modal_ratio": round(ratio, 4),
        "pass": ratio > 2.0,
        "grounding_map_visual_sdrs": gm_visual,
        "train_time_s": round(train_time, 1),
        "test_time_s": round(test_time, 1),
        "total_time_s": round(init_time + train_time + test_time, 1),
        "peak_gpu_mb": round(peak_mb, 0),
        "per_category_paired": [round(r, 6) for r in paired_rates],
        "per_category_random": [round(r, 6) for r in random_rates],
    }

    print(f"\nResults {label}:")
    print(f"  Paired activation:  {mean_paired:.6f}")
    print(f"  Random activation:  {mean_random:.6f}")
    print(f"  Cross-modal ratio:  {ratio:.4f}  {'PASS' if ratio > 2.0 else 'FAIL'}")
    print(f"  Time: train={train_time:.1f}s test={test_time:.1f}s")
    if peak_mb:
        print(f"  GPU peak: {peak_mb:.0f} MB")

    return result


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Seed: {SEED}")

    scales = [10_000, 15_000, 25_000, 50_000]
    priming_values = [0.0, 0.3]  # OFF vs ON
    results = []

    for n in scales:
        for ps in priming_values:
            result = run_scale(n, device, priming_strength=ps)
            results.append(result)

    # Summary table
    print(f"\n{'='*80}")
    print(f"SCALING SUMMARY WITH PRIMING (seed={SEED})")
    print(f"{'='*80}")
    print(f"{'N':>8} | {'Priming':>7} | {'Ratio':>8} | {'Pass':>4} | {'Paired':>10} | {'Random':>10} | {'Time':>8}")
    print("-" * 80)
    for r in results:
        ps = r['priming_strength']
        print(
            f"{r['N']//1000:>6}K | {ps:>7.1f} | {r['cross_modal_ratio']:>8.4f} | "
            f"{'YES' if r['pass'] else 'NO':>4} | {r['mean_paired']:>10.6f} | "
            f"{r['mean_random']:>10.6f} | {r['total_time_s']:>6.1f}s"
        )

    # Improvement summary
    print(f"\nPriming improvement:")
    for n in scales:
        no_p = next(r for r in results if r['N'] == n and r['priming_strength'] == 0.0)
        wi_p = next(r for r in results if r['N'] == n and r['priming_strength'] == 0.3)
        delta = wi_p['cross_modal_ratio'] - no_p['cross_modal_ratio']
        print(f"  N={n//1000}K: {no_p['cross_modal_ratio']:.4f} -> {wi_p['cross_modal_ratio']:.4f}  (delta={delta:+.4f})")

    out_path = "results/exp44_scaling_priming.json"
    os.makedirs("results", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"seed": SEED, "device": device, "results": results}, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
