"""Stage 90R local-only online evaluation."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

from stage90_quick_slice import _build_runtime, _json_default, load_stage89_baseline_reference

ROOT = Path(__file__).parent.parent
DOCS_DIR = ROOT / "_docs"
OUT_PATH = DOCS_DIR / "stage90r_local_only_eval.json"


def _allowed_primitives() -> list[str]:
    return ["move_left", "move_right", "move_up", "move_down", "do", "sleep"]


def main() -> None:
    from snks.agent.crafter_pixel_env import CrafterPixelEnv
    from snks.agent.perception import perceive_semantic_field, perceive_tile_field
    from snks.agent.post_mortem import DamageEvent, PostMortemAnalyzer, dominant_cause
    from snks.agent.stage90r_local_model import (
        load_local_evaluator_checkpoint,
        stage90r_action_utility,
    )
    from snks.agent.stage90r_local_policy import build_local_observation_package
    from snks.agent.vector_mpc_agent import DynamicEntityTracker
    from snks.agent.crafter_pixel_env import ACTION_TO_IDX

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-episodes", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=220)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--crop-world", action="store_true")
    parser.add_argument("--perception-mode", choices=("pixel", "symbolic"), default="pixel")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "demos" / "checkpoints" / "exp137" / "segmenter_9x9.pt",
    )
    parser.add_argument(
        "--local-evaluator",
        type=Path,
        default=DOCS_DIR / "stage90r_local_evaluator.pt",
    )
    args = parser.parse_args()

    baseline_reference = load_stage89_baseline_reference()
    model, segmenter, tb, _tracker_unused, runtime = _build_runtime(
        seed=args.seed,
        checkpoint=args.checkpoint if args.perception_mode == "pixel" else None,
        crop_world=args.crop_world,
    )
    del model, tb, runtime  # local-only eval does not use planner/world-model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluator = load_local_evaluator_checkpoint(args.local_evaluator, device=device)

    results: list[dict[str, Any]] = []
    death_causes: Counter[str] = Counter()
    t0 = time.time()

    for ep in range(args.n_episodes):
        env = CrafterPixelEnv(seed=args.seed + ep)
        pixels, info = env.reset()
        entity_tracker = DynamicEntityTracker()
        for cid in ("zombie", "skeleton", "cow", "arrow"):
            entity_tracker.register_dynamic_concept(cid)
        action_counts: Counter[str] = Counter()
        damage_log: list[DamageEvent] = []
        prev_body = None
        death_cause = "alive"
        steps_taken = 0

        for step in range(args.max_steps):
            steps_taken = step + 1
            raw_inv = dict(info.get("inventory", {}))
            body = {
                key: float(raw_inv.get(key, 9.0))
                for key in ("health", "food", "drink", "energy")
            }
            inv = {
                key: value
                for key, value in raw_inv.items()
                if key not in ("health", "food", "drink", "energy")
            }
            if prev_body is not None:
                health_delta = body.get("health", 0.0) - prev_body.get("health", 0.0)
                if health_delta < 0:
                    player_pos = tuple(info.get("player_pos", (32, 32)))
                    nearby_cids = []
                    for entity_cid, entity_pos in entity_tracker.visible_entities():
                        ex, ey = entity_pos
                        dist = abs(ex - player_pos[0]) + abs(ey - player_pos[1])
                        nearby_cids.append((entity_cid, dist))
                    damage_log.append(
                        DamageEvent(
                            step=step,
                            health_delta=float(health_delta),
                            vitals={
                                key: prev_body.get(key, 9.0)
                                for key in ("food", "drink", "energy")
                            },
                            nearby_cids=nearby_cids,
                        )
                    )
            if args.perception_mode == "pixel":
                vf = perceive_tile_field(pixels, segmenter)
            else:
                vf = perceive_semantic_field(info)
            player_pos = tuple(info.get("player_pos", (32, 32)))
            entity_tracker.update(vf, player_pos)

            obs = build_local_observation_package(vf, body, inv)
            class_ids = torch.tensor([obs["viewport_class_ids"]], dtype=torch.long, device=device)
            confidences = torch.tensor([obs["viewport_confidences"]], dtype=torch.float32, device=device)
            body_vec = torch.tensor([obs["body_vector"]], dtype=torch.float32, device=device)
            inv_vec = torch.tensor([obs["inventory_vector"]], dtype=torch.float32, device=device)

            best_primitive = None
            best_score = None
            for primitive in _allowed_primitives():
                action_idx = torch.tensor([ACTION_TO_IDX[primitive]], dtype=torch.long, device=device)
                preds = evaluator(class_ids, confidences, body_vec, inv_vec, action_idx)
                utility = stage90r_action_utility(**preds).item()
                if best_score is None or utility > best_score:
                    best_score = utility
                    best_primitive = primitive

            primitive = best_primitive or "move_right"
            action_counts[primitive] += 1
            if step % 20 == 0:
                print(
                    f"s{step:3d} H{body.get('health', 0):.0f} "
                    f"F{body.get('food', 0):.0f} D{body.get('drink', 0):.0f} "
                    f"near={vf.near_concept:9s} → {primitive:12s} local_only"
                )

            prev_body = dict(body)
            pixels, _reward, done, info = env.step(primitive)
            if done:
                final_raw_inv = dict(info.get("inventory", {}))
                final_body = {
                    key: float(final_raw_inv.get(key, 0.0))
                    for key in ("health", "food", "drink", "energy")
                }
                if any(final_body.get(key, 0.0) <= 0.0 for key in final_body):
                    health_delta = final_body.get("health", 0.0) - body.get("health", 9.0)
                    if health_delta < 0:
                        nearby_cids = []
                        for entity_cid, entity_pos in entity_tracker.visible_entities():
                            ex, ey = entity_pos
                            dist = abs(ex - player_pos[0]) + abs(ey - player_pos[1])
                            nearby_cids.append((entity_cid, dist))
                        damage_log.append(
                            DamageEvent(
                                step=step,
                                health_delta=float(health_delta),
                                vitals={k: body.get(k, 9.0) for k in ("food", "drink", "energy")},
                                nearby_cids=nearby_cids,
                            )
                        )
                break

        attribution = PostMortemAnalyzer().attribute(damage_log, steps_taken)
        death_cause = dominant_cause(attribution)
        death_causes[death_cause] += 1
        results.append(
            {
                "episode_id": ep,
                "seed": args.seed + ep,
                "episode_steps": steps_taken,
                "death_cause": death_cause,
                "action_counts": dict(action_counts),
            }
        )
        elapsed = time.time() - t0
        print(
            f"ep{ep:3d}: len={steps_taken:4d} death={death_cause:12s} "
            f"[{elapsed:.0f}s]"
        )

    avg_survival = round(
        sum(result["episode_steps"] for result in results) / max(len(results), 1),
        2,
    )
    payload = {
        "stage": "stage90r_local_only_eval",
        "baseline_reference": baseline_reference,
        "config": {
            "n_episodes": args.n_episodes,
            "max_steps": args.max_steps,
            "seed": args.seed,
            "perception_mode": args.perception_mode,
            "local_evaluator": str(args.local_evaluator),
        },
        "summary": {
            "avg_survival": avg_survival,
            "death_cause_breakdown": dict(death_causes),
        },
        "episodes": results,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2, default=_json_default))
    print(f"saved eval: {OUT_PATH}")


if __name__ == "__main__":
    main()
