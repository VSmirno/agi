"""Grid Search for Exp 9 C1: Decay Epsilon.

Search over epsilon_start and decay horizon.
denom=5.5, weights=(0.92, 0.08) fixed from prior grid search.

Usage:
    HSA_OVERRIDE_GFX_VERSION=11.0.0 PYTHONUNBUFFERED=1 \
        python -m snks.experiments.grid_search_exp9_c1
"""

from __future__ import annotations

import csv
import itertools
import time
from pathlib import Path

from snks.daf.types import (
    CausalAgentConfig,
    DafConfig,
    EncoderConfig,
    PipelineConfig,
    SKSConfig,
)
from snks.experiments.exp9_curiosity import _run_curious_agent, _run_random_agent

DEVICE = "hip"
NUM_NODES = 5000
N_STEPS = 2000
N_TRIALS = 3

EPSILON_STARTS = [0.20, 0.25, 0.30, 0.40]
EPSILON_HORIZONS = [1000, 2000, 3000]

RESULTS_PATH = Path("/opt/agi/grid_search_exp9_c1.csv") if Path("/opt/agi").exists() else Path("grid_search_exp9_c1.csv")


def make_config(device: str, eps_start: float, eps_horizon: int) -> CausalAgentConfig:
    return CausalAgentConfig(
        pipeline=PipelineConfig(
            daf=DafConfig(
                num_nodes=NUM_NODES,
                avg_degree=20,
                oscillator_model="fhn",
                coupling_strength=0.05,
                dt=0.01,
                noise_sigma=0.005,
                fhn_I_base=0.0,
                device=device,
            ),
            encoder=EncoderConfig(
                sdr_size=4096,
                sdr_sparsity=0.04,
            ),
            sks=SKSConfig(
                coherence_mode="rate",
                min_cluster_size=5,
                dbscan_min_samples=5,
            ),
            steps_per_cycle=100,
            device=device,
        ),
        motor_sdr_size=256,
        causal_min_observations=2,
        curiosity_epsilon=eps_start,
        curiosity_epsilon_min=0.05,
        curiosity_epsilon_horizon=eps_horizon,
    )


def main() -> None:
    configs = list(itertools.product(EPSILON_STARTS, EPSILON_HORIZONS))
    print(f"Grid search C1: {len(configs)} configs × {N_TRIALS} trials each")
    print(f"Device: {DEVICE}, num_nodes: {NUM_NODES}, n_steps: {N_STEPS}")
    print("-" * 60)


    results = []

    for i, (eps_start, eps_horizon) in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] eps_start={eps_start}, horizon={eps_horizon}")
        t0 = time.time()

        config = make_config(DEVICE, eps_start, eps_horizon)

        curious_coverages = []
        random_coverages = []

        for trial in range(N_TRIALS):
            rand_cov, _ = _run_random_agent(N_STEPS)
            cur_cov, _, _ = _run_curious_agent(config, N_STEPS)
            curious_coverages.append(cur_cov)
            random_coverages.append(rand_cov)
            print(f"  trial {trial+1}: curious={cur_cov:.3f} random={rand_cov:.3f}")

        import numpy as np
        avg_curious = float(np.mean(curious_coverages))
        avg_random = float(np.mean(random_coverages))
        ratio = avg_curious / max(avg_random, 0.001)
        elapsed = time.time() - t0

        print(f"  => ratio={ratio:.3f} (curious={avg_curious:.3f} random={avg_random:.3f}) [{elapsed:.0f}s]")
        print(f"  {'✅ PASS' if ratio > 1.5 else '❌ FAIL'}")

        results.append({
            "eps_start": eps_start,
            "eps_horizon": eps_horizon,
            "coverage_ratio": ratio,
            "curious_coverage": avg_curious,
            "random_coverage": avg_random,
            "elapsed_s": elapsed,
        })

        # Save after each config
        with open(RESULTS_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    print("\n" + "=" * 60)
    print("GRID SEARCH C1 COMPLETE")
    print("=" * 60)
    results.sort(key=lambda r: r["coverage_ratio"], reverse=True)
    print(f"{'eps_start':>10} {'horizon':>8} {'ratio':>8} {'pass':>6}")
    print("-" * 40)
    for r in results:
        flag = "✅" if r["coverage_ratio"] > 1.5 else "❌"
        print(f"{r['eps_start']:>10.2f} {r['eps_horizon']:>8} {r['coverage_ratio']:>8.3f} {flag}")
    print(f"\nBest: eps_start={results[0]['eps_start']}, horizon={results[0]['eps_horizon']}, ratio={results[0]['coverage_ratio']:.3f}")
    print(f"Results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
