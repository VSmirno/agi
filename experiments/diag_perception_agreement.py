"""Evaluate pixel/CNN VisualField agreement against semantic VisualField.

Supports both:
- legacy full-frame segmenters
- crop-trained segmenters (exp137) via --crop-world

Run on minipc:
  ./scripts/minipc-run.sh diagagree "from diag_perception_agreement import main; main()"
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from snks.agent.perception import perceive_semantic_field, perceive_tile_field
from snks.encoder.tile_segmenter import crop_world_pixels, load_tile_segmenter, pick_device

ROOT = Path(__file__).parent.parent


class CroppedSegmenter:
    def __init__(self, segmenter):
        self.segmenter = segmenter

    def classify_tiles(self, pixels):
        return self.segmenter.classify_tiles(crop_world_pixels(pixels))


def _norm_detections(vf) -> set[tuple[str, int, int]]:
    return {(cid, int(gy), int(gx)) for cid, _conf, gy, gx in vf.detections}


def _compare_fields(pixel_vf, symbolic_vf) -> dict:
    pixel_set = _norm_detections(pixel_vf)
    symbolic_set = _norm_detections(symbolic_vf)
    overlap = pixel_set & symbolic_set
    union = pixel_set | symbolic_set
    return {
        "near_match": pixel_vf.near_concept == symbolic_vf.near_concept,
        "pixel_near": pixel_vf.near_concept,
        "symbolic_near": symbolic_vf.near_concept,
        "jaccard": round(len(overlap) / max(len(union), 1), 3),
        "pixel_only": sorted(pixel_set - symbolic_set),
        "symbolic_only": sorted(symbolic_set - pixel_set),
    }


def _aggregate(rows: list[dict], args) -> dict:
    near_matches = sum(1 for row in rows if row["near_match"])
    mean_jaccard = float(np.mean([row["jaccard"] for row in rows])) if rows else 0.0
    pixel_only_by_concept: Counter[str] = Counter()
    symbolic_only_by_concept: Counter[str] = Counter()
    pixel_only_by_row: Counter[int] = Counter()
    symbolic_only_by_row: Counter[int] = Counter()
    near_mismatch_pairs: Counter[str] = Counter()

    for row in rows:
        if not row["near_match"]:
            near_mismatch_pairs[f"{row['pixel_near']}->{row['symbolic_near']}"] += 1
        for cid, gy, _gx in row["pixel_only"]:
            pixel_only_by_concept[cid] += 1
            pixel_only_by_row[int(gy)] += 1
        for cid, gy, _gx in row["symbolic_only"]:
            symbolic_only_by_concept[cid] += 1
            symbolic_only_by_row[int(gy)] += 1

    return {
        "checkpoint": str(args.checkpoint),
        "crop_world": bool(args.crop_world),
        "seed_start": args.seed,
        "n_seeds": args.n_seeds,
        "max_steps": args.max_steps,
        "sample_every": args.sample_every,
        "n_samples": len(rows),
        "near_match_rate": round(near_matches / max(len(rows), 1), 3),
        "mean_jaccard": round(mean_jaccard, 3),
        "pixel_only_by_concept": dict(pixel_only_by_concept.most_common()),
        "symbolic_only_by_concept": dict(symbolic_only_by_concept.most_common()),
        "pixel_only_by_row": {str(k): v for k, v in sorted(pixel_only_by_row.items())},
        "symbolic_only_by_row": {str(k): v for k, v in sorted(symbolic_only_by_row.items())},
        "near_mismatch_pairs": dict(near_mismatch_pairs.most_common()),
    }


def main() -> None:
    from snks.agent.crafter_pixel_env import CrafterPixelEnv

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "demos" / "checkpoints" / "exp136" / "segmenter_9x9.pt",
    )
    parser.add_argument("--crop-world", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-seeds", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--sample-every", type=int, default=5)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "_docs" / "diag_perception_agreement.json",
    )
    args = parser.parse_args()

    device = torch.device(args.device) if args.device is not None else torch.device(pick_device())
    segmenter = load_tile_segmenter(str(args.checkpoint), device=device)
    encoder = CroppedSegmenter(segmenter) if args.crop_world else segmenter

    rows: list[dict] = []
    for seed in range(args.seed, args.seed + args.n_seeds):
        env = CrafterPixelEnv(seed=seed)
        pixels, info = env.reset()
        for step in range(args.max_steps):
            if step % args.sample_every == 0:
                pixel_vf = perceive_tile_field(pixels, encoder)
                symbolic_vf = perceive_semantic_field(info)
                rows.append({"seed": seed, "step": step, **_compare_fields(pixel_vf, symbolic_vf)})
            action = np.random.randint(0, env.n_actions)
            pixels, _reward, done, info = env.step(action)
            if done:
                break

    summary = _aggregate(rows, args)
    args.out.write_text(json.dumps({"summary": summary, "samples": rows}, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"saved={args.out}")


if __name__ == "__main__":
    main()
