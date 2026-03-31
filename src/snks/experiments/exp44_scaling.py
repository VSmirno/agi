"""Exp 44 scaling study: cross-modal recall quality vs network size.

Runs exp44 at N=10K, 15K, 25K, 50K with fixed seeds for reproducibility.
Reports cross_modal_ratio and timing at each scale.
"""

from __future__ import annotations

import json
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
    measure_visual_zone_activation,
)

SEED = 42


def make_zones(n: int) -> dict[str, ZoneConfig]:
    """Scale zones proportionally. 60% visual, 40% linguistic."""
    vis = int(n * 0.6)
    ling = n - vis
    return {
        "visual": ZoneConfig(start=0, size=vis),
        "linguistic": ZoneConfig(start=vis, size=ling),
    }


def make_config(n: int, device: str) -> PipelineConfig:
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
    )


def run_scale(n: int, device: str, n_train_reps: int = 15, n_variations: int = 5) -> dict:
    """Run cross-modal recall at given N, return metrics + timing."""
    torch.manual_seed(SEED)
    if device == "cuda":
        torch.cuda.manual_seed(SEED)
        torch.cuda.reset_peak_memory_stats()

    zones = make_zones(n)
    vis_zone = zones["visual"]

    print(f"\n{'='*60}")
    print(f"N={n//1000}K  (visual={vis_zone.size}, linguistic={zones['linguistic'].size})")
    print(f"{'='*60}")

    t_init = time.time()
    config = make_config(n, device)
    pipeline = Pipeline(config)
    init_time = time.time() - t_init

    # Generate pairs (deterministic via SEED)
    pairs = []
    for cat_idx in range(len(CATEGORIES)):
        for var in range(n_variations):
            img = make_synthetic_image(cat_idx, var)
            pairs.append((img, CATEGORIES[cat_idx]))

    # Co-activation training
    total_train = n_train_reps * len(pairs)
    print(f"Training: {n_train_reps} reps x {len(pairs)} pairs = {total_train} cycles")
    t_train = time.time()
    for rep in range(n_train_reps):
        for img, text in pairs:
            pipeline.perception_cycle(image=img, text=text)
        elapsed_rep = time.time() - t_train
        eta = elapsed_rep / (rep + 1) * (n_train_reps - rep - 1)
        print(f"  Rep {rep+1}/{n_train_reps}  [{elapsed_rep:.1f}s elapsed, ~{eta:.0f}s remaining]")
    train_time = time.time() - t_train

    # Check inter-zone weight growth (STDP effect)
    graph = pipeline.engine.graph
    zones_dict = make_zones(n)
    vis_end = zones_dict["visual"].start + zones_dict["visual"].size
    src = graph.edge_index[0]
    dst = graph.edge_index[1]
    cross_mask = ((src < vis_end) & (dst >= vis_end)) | ((src >= vis_end) & (dst < vis_end))
    intra_mask = ~cross_mask
    cross_w = graph.edge_attr[cross_mask, 0].mean().item()
    intra_w = graph.edge_attr[intra_mask, 0].mean().item()

    # Measure: paired text -> visual activation
    t_test = time.time()
    paired_rates = []
    for cat_idx, text in enumerate(CATEGORIES):
        rate = measure_visual_zone_activation(pipeline, text, vis_zone)
        paired_rates.append(rate)

    # Measure: random text -> visual activation
    random_rates = []
    for text in RANDOM_TEXTS:
        rate = measure_visual_zone_activation(pipeline, text, vis_zone)
        random_rates.append(rate)
    test_time = time.time() - t_test

    mean_paired = sum(paired_rates) / len(paired_rates)
    mean_random = sum(random_rates) / len(random_rates)
    ratio = mean_paired / max(mean_random, 1e-9)

    peak_mb = torch.cuda.max_memory_allocated() / 1024**2 if device == "cuda" else 0

    result = {
        "N": n,
        "visual_size": vis_zone.size,
        "linguistic_size": zones["linguistic"].size,
        "mean_paired": round(mean_paired, 6),
        "mean_random": round(mean_random, 6),
        "cross_modal_ratio": round(ratio, 4),
        "pass": ratio > 2.0,
        "cross_zone_weight": round(cross_w, 4),
        "intra_zone_weight": round(intra_w, 4),
        "init_time_s": round(init_time, 1),
        "train_time_s": round(train_time, 1),
        "test_time_s": round(test_time, 1),
        "total_time_s": round(init_time + train_time + test_time, 1),
        "peak_gpu_mb": round(peak_mb, 0),
        "per_category_paired": [round(r, 6) for r in paired_rates],
        "per_category_random": [round(r, 6) for r in random_rates],
    }

    print(f"\nResults N={n//1000}K:")
    print(f"  Paired activation:  {mean_paired:.6f}")
    print(f"  Random activation:  {mean_random:.6f}")
    print(f"  Cross-modal ratio:  {ratio:.4f}  {'PASS' if ratio > 2.0 else 'FAIL'}")
    print(f"  Inter-zone weight:  {cross_w:.4f}")
    print(f"  Intra-zone weight:  {intra_w:.4f}")
    print(f"  Time: init={init_time:.1f}s train={train_time:.1f}s test={test_time:.1f}s")
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
    results = []

    for n in scales:
        result = run_scale(n, device)
        results.append(result)

    # Summary table
    print(f"\n{'='*70}")
    print(f"SCALING SUMMARY (seed={SEED})")
    print(f"{'='*70}")
    print(f"{'N':>8} | {'Ratio':>8} | {'Pass':>4} | {'Paired':>10} | {'Random':>10} | {'XZone W':>8} | {'Time':>8} | {'GPU MB':>7}")
    print("-" * 70)
    for r in results:
        print(
            f"{r['N']//1000:>6}K | {r['cross_modal_ratio']:>8.4f} | "
            f"{'YES' if r['pass'] else 'NO':>4} | {r['mean_paired']:>10.6f} | "
            f"{r['mean_random']:>10.6f} | {r['cross_zone_weight']:>8.4f} | "
            f"{r['total_time_s']:>6.1f}s | {r['peak_gpu_mb']:>6.0f}"
        )

    # Save results
    out_path = "results/exp44_scaling.json"
    import os
    os.makedirs("results", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"seed": SEED, "device": device, "results": results}, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
