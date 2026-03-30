"""Generate demo data for Stage 19 grounding visualization.

Trains zonal DAF with priming, saves checkpoint, exports JSON for HTML demo.
Usage: PYTHONPATH=src python src/snks/experiments/stage19_demo_gen.py [--device cuda]
"""

from __future__ import annotations

import argparse
import json
import math
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
N = 5_000  # small N is fine with priming
N_TRAIN_REPS = 15
N_VARIATIONS = 5
CHECKPOINT_DIR = "checkpoints/stage19_demo"
OUTPUT_DIR = "demo_output"


def make_zones() -> dict[str, ZoneConfig]:
    """Config A (no convergence) — cleaner activations at demo scale N=5K."""
    return {
        "visual": ZoneConfig(start=0, size=3000),
        "linguistic": ZoneConfig(start=3000, size=2000),
    }


def make_config(device: str) -> PipelineConfig:
    zones = make_zones()
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


def collect_activation(pipeline: Pipeline, text: str, vis_zone: ZoneConfig, n_runs: int = 5) -> torch.Tensor:
    """Run text-only cycles, return averaged visual zone firing rates."""
    total = torch.zeros(vis_zone.size)
    for _ in range(n_runs):
        pipeline.perception_cycle(text=text)
        fired = pipeline.engine.get_fired_history()
        if fired is not None:
            total += fired[:, vis_zone.start:vis_zone.start + vis_zone.size].float().mean(dim=0).cpu()
    return total / n_runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    if args.device == "cuda":
        torch.cuda.manual_seed(SEED)

    device = args.device
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")

    zones = make_zones()
    vis_zone = zones["visual"]

    # --- Train ---
    print("\n1. Training...")
    config = make_config(device)
    pipeline = Pipeline(config)

    pairs = []
    images = {}  # category -> image tensor
    for cat_idx in range(len(CATEGORIES)):
        img = make_synthetic_image(cat_idx, 0)
        images[CATEGORIES[cat_idx]] = img
        for var in range(N_VARIATIONS):
            img_var = make_synthetic_image(cat_idx, var)
            pairs.append((img_var, CATEGORIES[cat_idx]))

    t0 = time.time()
    for rep in range(N_TRAIN_REPS):
        for img, text in pairs:
            pipeline.perception_cycle(image=img, text=text)
        print(f"  Rep {rep+1}/{N_TRAIN_REPS}  [{time.time()-t0:.1f}s]")
    train_time = time.time() - t0
    print(f"Training done in {train_time:.1f}s")

    # --- Save checkpoint ---
    print("\n2. Saving checkpoint...")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    ckpt_path = os.path.join(CHECKPOINT_DIR, "pipeline")
    pipeline.save_checkpoint(ckpt_path)
    print(f"  Saved to {ckpt_path}")

    # --- Collect demo data ---
    print("\n3. Collecting demo data...")

    # Per-category heatmaps
    heatmaps = {}
    paired_activations = []
    for cat_idx, cat in enumerate(CATEGORIES):
        activation = collect_activation(pipeline, cat, vis_zone)
        mean_act = activation.mean().item()
        paired_activations.append(mean_act)

        heatmaps[cat] = {
            "visual_activation": activation.tolist(),
            "original_image": images[cat].tolist(),
            "cross_modal_ratio": 0.0,  # filled after random
        }
        print(f"  {cat}: activation={mean_act:.6f}")

    # Random activations
    random_activations = []
    for text in RANDOM_TEXTS:
        activation = collect_activation(pipeline, text, vis_zone)
        random_activations.append(activation.mean().item())

    mean_random = sum(random_activations) / len(random_activations) if random_activations else 1e-9

    # Fill ratios
    for cat_idx, cat in enumerate(CATEGORIES):
        ratio = paired_activations[cat_idx] / max(mean_random, 1e-9)
        heatmaps[cat]["cross_modal_ratio"] = round(ratio, 2)

    # Discrimination matrix: row=word presented, col=category pattern activated
    print("\n  Building discrimination matrix...")
    disc_matrix = []
    for word_idx, word in enumerate(CATEGORIES):
        row = []
        # Present word, measure activation overlap with each category's typical pattern
        activation = collect_activation(pipeline, word, vis_zone)
        for cat_idx, cat in enumerate(CATEGORIES):
            # Use correlation with that category's activation as similarity
            cat_act = torch.tensor(heatmaps[cat]["visual_activation"])
            act_cpu = activation.cpu()
            if cat_act.norm() > 0 and act_cpu.norm() > 0:
                sim = torch.nn.functional.cosine_similarity(
                    act_cpu.unsqueeze(0), cat_act.unsqueeze(0)
                ).item()
            else:
                sim = 0.0
            row.append(round(max(sim, 0.0), 4))
        disc_matrix.append(row)

    # --- Export JSON ---
    print("\n4. Exporting JSON...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    demo_data = {
        "categories": CATEGORIES,
        "discrimination_matrix": disc_matrix,
        "paired_activations": [round(v, 8) for v in paired_activations],
        "random_activations": [round(v, 8) for v in random_activations],
        "random_texts": RANDOM_TEXTS,
        "heatmaps": heatmaps,
        "config": {
            "visual_zone_size": vis_zone.size,
            "linguistic_zone_size": zones["linguistic"].size,
            "convergence_zone_size": zones["convergence"].size if "convergence" in zones else 0,
            "total_nodes": sum(z.size for z in zones.values()),
            "priming_strength": config.priming_strength,
            "chosen_config": "B (with convergence)",
            "n_train_reps": N_TRAIN_REPS,
            "n_categories": len(CATEGORIES),
            "seed": SEED,
        },
    }

    out_path = os.path.join(OUTPUT_DIR, "stage19_demo_data.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(demo_data, f, ensure_ascii=False, indent=2)

    print(f"  Saved to {out_path}")
    print(f"\nDone! Open demo:")
    print(f"  cd {OUTPUT_DIR} && python -m http.server 8080")
    print(f"  http://localhost:8080/stage19_grounding_demo.html")


if __name__ == "__main__":
    main()
