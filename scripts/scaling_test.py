"""Scaling test: run experiments 1-4 at multiple network sizes.

Measures how key metrics change with num_nodes to identify scaling trends.
Usage:
    python scripts/scaling_test.py [--device cuda] [--sizes 10000,20000,30000]
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import torch


@dataclass
class ScalePoint:
    num_nodes: int
    device: str
    # Exp 1
    exp1_nmi: float
    exp1_time_s: float
    # Exp 2
    exp2_retention_pct: float
    exp2_time_s: float
    # Exp 3
    exp3_mean_accuracy: float
    exp3_time_s: float
    # Exp 4
    exp4_nmi_clean: float
    exp4_graceful: bool
    exp4_time_s: float
    # Memory
    vram_peak_mb: float


def get_vram_mb(device: str) -> float:
    """Get peak VRAM usage in MB."""
    if device == "cpu":
        return 0.0
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0.0


def reset_vram(device: str) -> None:
    """Reset peak VRAM tracking."""
    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def run_scale_point(num_nodes: int, device: str) -> ScalePoint:
    """Run all 4 experiments at a given scale."""
    # Import here to avoid loading everything at parse time
    from snks.experiments import exp1_sks_formation, exp2_continual, exp3_prediction, exp4_noise

    reset_vram(device)
    print(f"\n{'='*60}")
    print(f"  Scale point: {num_nodes:,} nodes on {device}")
    print(f"{'='*60}")

    # Exp 1: SKS Formation
    print(f"\n--- Experiment 1: SKS Formation ---")
    t0 = time.time()
    r1 = exp1_sks_formation.run(device=device, num_nodes=num_nodes)
    t1 = time.time()
    exp1_time = t1 - t0
    exp1_nmi = r1.final_nmi
    print(f"  Time: {exp1_time:.1f}s")

    # Exp 2: Continual Learning
    print(f"\n--- Experiment 2: Continual Learning ---")
    t0 = time.time()
    r2 = exp2_continual.run(device=device, num_nodes=num_nodes)
    t2 = time.time()
    exp2_time = t2 - t0
    exp2_retention = r2.retention_pct
    print(f"  Time: {exp2_time:.1f}s")

    # Exp 3: Prediction
    print(f"\n--- Experiment 3: Sequence Prediction ---")
    t0 = time.time()
    r3 = exp3_prediction.run(device=device, num_nodes=num_nodes)
    t3 = time.time()
    exp3_time = t3 - t0
    exp3_acc = (r3.accuracy_3 + r3.accuracy_5 + r3.accuracy_7) / 3
    print(f"  Time: {exp3_time:.1f}s")

    # Exp 4: Noise Robustness
    print(f"\n--- Experiment 4: Noise Robustness ---")
    t0 = time.time()
    r4 = exp4_noise.run(device=device, num_nodes=num_nodes)
    t4 = time.time()
    exp4_time = t4 - t0
    print(f"  Time: {exp4_time:.1f}s")

    vram_peak = get_vram_mb(device)

    return ScalePoint(
        num_nodes=num_nodes,
        device=device,
        exp1_nmi=exp1_nmi,
        exp1_time_s=round(exp1_time, 1),
        exp2_retention_pct=round(exp2_retention, 1),
        exp2_time_s=round(exp2_time, 1),
        exp3_mean_accuracy=round(exp3_acc, 4),
        exp3_time_s=round(exp3_time, 1),
        exp4_nmi_clean=r4.nmi_clean,
        exp4_graceful=r4.graceful,
        exp4_time_s=round(exp4_time, 1),
        vram_peak_mb=round(vram_peak, 0),
    )


def print_summary(results: list[ScalePoint]) -> None:
    """Print a comparison table."""
    print(f"\n{'='*80}")
    print(f"  SCALING SUMMARY")
    print(f"{'='*80}")

    header = f"{'Nodes':>8} | {'NMI(1)':>7} | {'Ret%(2)':>7} | {'Acc(3)':>7} | {'NMI_c(4)':>8} | {'Grace':>5} | {'VRAM MB':>8} | {'Total s':>8}"
    print(header)
    print("-" * len(header))

    for r in results:
        total_time = r.exp1_time_s + r.exp2_time_s + r.exp3_time_s + r.exp4_time_s
        grace = "YES" if r.exp4_graceful else "NO"
        print(
            f"{r.num_nodes:>8,} | {r.exp1_nmi:>7.3f} | {r.exp2_retention_pct:>6.1f}% | {r.exp3_mean_accuracy:>7.1%} | {r.exp4_nmi_clean:>8.3f} | {grace:>5} | {r.vram_peak_mb:>7.0f} | {total_time:>7.1f}"
        )

    # Scaling trend
    if len(results) >= 2:
        first, last = results[0], results[-1]
        scale_ratio = last.num_nodes / first.num_nodes
        nmi_delta = last.exp1_nmi - first.exp1_nmi
        ret_delta = last.exp2_retention_pct - first.exp2_retention_pct
        acc_delta = last.exp3_mean_accuracy - first.exp3_mean_accuracy

        print(f"\n  Scale {first.num_nodes:,} -> {last.num_nodes:,} ({scale_ratio:.1f}x):")
        print(f"    NMI(Exp1):   {first.exp1_nmi:.3f} -> {last.exp1_nmi:.3f} (d {nmi_delta:+.3f})")
        print(f"    Retention:   {first.exp2_retention_pct:.1f}% -> {last.exp2_retention_pct:.1f}% (d {ret_delta:+.1f}%)")
        print(f"    Accuracy:    {first.exp3_mean_accuracy:.1%} -> {last.exp3_mean_accuracy:.1%} (d {acc_delta:+.1%})")


def main():
    parser = argparse.ArgumentParser(description="SNKS Scaling Test")
    parser.add_argument("--device", default="auto", help="Device: cpu, cuda, auto")
    parser.add_argument("--sizes", default="10000,20000,30000",
                        help="Comma-separated list of num_nodes to test")
    parser.add_argument("--output", default="results/scaling_test.json",
                        help="Path to save JSON results")
    args = parser.parse_args()

    # Resolve device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    else:
        device = args.device

    sizes = [int(s.strip()) for s in args.sizes.split(",")]

    print(f"SNKS Scaling Test")
    print(f"  Device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name()}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"  Sizes: {sizes}")

    results = []
    for n in sizes:
        point = run_scale_point(n, device)
        results.append(point)

    print_summary(results)

    # Save results
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
