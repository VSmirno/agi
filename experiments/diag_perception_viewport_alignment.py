"""Diagnostic: compare pixel vs symbolic VisualField on the same Crafter frames.

Run on minipc:
  ./scripts/minipc-run.sh vfalign \
    "import sys; sys.argv=['diag_perception_viewport_alignment.py']; from diag_perception_viewport_alignment import main; main()"

Writes:
  _docs/diag_perception_viewport_alignment.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from snks.agent.perception import perceive_semantic_field, perceive_tile_field

torch.backends.cudnn.enabled = False

ROOT = Path(__file__).parent.parent
OUT_PATH = ROOT / "_docs" / "diag_perception_viewport_alignment.json"
CASE_DIR = ROOT / "_docs" / "diag_perception_viewport_cases"


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
        "pixel_count": len(pixel_set),
        "symbolic_count": len(symbolic_set),
        "overlap_count": len(overlap),
        "union_count": len(union),
        "jaccard": round(len(overlap) / max(len(union), 1), 3),
        "pixel_only": sorted(pixel_set - symbolic_set),
        "symbolic_only": sorted(symbolic_set - pixel_set),
    }


def _case_priority(row: dict) -> tuple[int, float]:
    return (
        0 if not row["near_match"] else 1,
        float(row["jaccard"]),
    )


def _write_case_dump(
    *,
    seed: int,
    step: int,
    pixels: np.ndarray,
    semantic: np.ndarray,
    info: dict,
    pixel_vf,
    symbolic_vf,
) -> str:
    CASE_DIR.mkdir(parents=True, exist_ok=True)
    path = CASE_DIR / f"seed{seed}_step{step}.npz"
    np.savez_compressed(
        path,
        pixels=np.asarray(pixels),
        semantic=np.asarray(semantic),
        player_pos=np.asarray(info.get("player_pos", (0, 0))),
        inventory=np.asarray(list(dict(info.get("inventory", {})).items()), dtype=object),
        pixel_detections=np.asarray(pixel_vf.detections, dtype=object),
        symbolic_detections=np.asarray(symbolic_vf.detections, dtype=object),
        pixel_near=np.asarray(pixel_vf.near_concept, dtype=object),
        symbolic_near=np.asarray(symbolic_vf.near_concept, dtype=object),
    )
    return str(path)


def _aggregate(rows: list[dict], seed_start: int, n_seeds: int, max_steps: int, sample_every: int) -> dict:
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
        "seed_start": seed_start,
        "n_seeds": n_seeds,
        "max_steps": max_steps,
        "sample_every": sample_every,
        "n_samples": len(rows),
        "near_match_rate": round(near_matches / max(len(rows), 1), 3),
        "mean_jaccard": round(mean_jaccard, 3),
        "mismatch_steps": [
            {"seed": row["seed"], "step": row["step"]}
            for row in rows
            if not row["near_match"]
        ],
        "pixel_only_by_concept": dict(pixel_only_by_concept.most_common()),
        "symbolic_only_by_concept": dict(symbolic_only_by_concept.most_common()),
        "pixel_only_by_row": {str(k): v for k, v in sorted(pixel_only_by_row.items())},
        "symbolic_only_by_row": {str(k): v for k, v in sorted(symbolic_only_by_row.items())},
        "near_mismatch_pairs": dict(near_mismatch_pairs.most_common()),
    }


def main() -> None:
    from snks.agent.crafter_pixel_env import CrafterPixelEnv
    from snks.encoder.tile_segmenter import load_tile_segmenter, pick_device

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-seeds", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--sample-every", type=int, default=5)
    parser.add_argument("--dump-cases", action="store_true")
    parser.add_argument("--max-cases", type=int, default=12)
    parser.add_argument("--jaccard-threshold", type=float, default=0.4)
    args = parser.parse_args()

    checkpoint_path = ROOT / "demos" / "checkpoints" / "exp136" / "segmenter_9x9.pt"
    device = torch.device(pick_device())
    segmenter = load_tile_segmenter(str(checkpoint_path), device=device)
    rows: list[dict] = []
    case_rows: list[dict] = []

    for seed in range(args.seed, args.seed + args.n_seeds):
        env = CrafterPixelEnv(seed=seed)
        pixels, info = env.reset()

        for step in range(args.max_steps):
            if step % args.sample_every == 0:
                pixel_vf = perceive_tile_field(pixels, segmenter)
                symbolic_vf = perceive_semantic_field(info)
                row = {
                    "seed": seed,
                    "step": step,
                    **_compare_fields(pixel_vf, symbolic_vf),
                }
                rows.append(row)

                if (
                    args.dump_cases
                    and (not row["near_match"] or row["jaccard"] <= args.jaccard_threshold)
                ):
                    semantic = np.asarray(info.get("semantic")) if info.get("semantic") is not None else None
                    if semantic is not None:
                        row = dict(row)
                        row["case_dump"] = _write_case_dump(
                            seed=seed,
                            step=step,
                            pixels=np.asarray(pixels),
                            semantic=semantic,
                            info=info,
                            pixel_vf=pixel_vf,
                            symbolic_vf=symbolic_vf,
                        )
                        case_rows.append(row)

            action = np.random.randint(0, env.n_actions)
            pixels, _reward, done, info = env.step(action)
            if done:
                break

    if args.dump_cases:
        case_rows = sorted(case_rows, key=_case_priority)[: args.max_cases]

    summary = _aggregate(
        rows,
        seed_start=args.seed,
        n_seeds=args.n_seeds,
        max_steps=args.max_steps,
        sample_every=args.sample_every,
    )
    summary["case_dump_count"] = len(case_rows)
    summary["case_dump_dir"] = str(CASE_DIR) if case_rows else None

    OUT_PATH.write_text(
        json.dumps(
            {"summary": summary, "samples": rows, "case_samples": case_rows},
            indent=2,
        )
    )
    print(json.dumps(summary, indent=2))
    print(f"Saved alignment diagnostics to {OUT_PATH}")


if __name__ == "__main__":
    main()
