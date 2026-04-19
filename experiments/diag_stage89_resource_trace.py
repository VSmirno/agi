"""Trace adjacent resource interaction for one Stage 89 episode.

Focus: understand why the agent fails to gather tree/wood when near trees.

Writes:
  - _docs/diag_stage89_resource_trace_seedXXXX.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

ROOT = Path(__file__).parent.parent
OUT_DIR = ROOT / "_docs"


def _json_default(value):
    try:
        import numpy as np

        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.bool_):
            return bool(value)
    except Exception:
        pass
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def main() -> None:
    from stage89_eval import _build_model_and_segmenter, _stage89_mode_config
    from snks.agent.crafter_pixel_env import CrafterPixelEnv
    from snks.agent.crafter_textbook import CrafterTextbook
    from snks.agent.perception import HomeostaticTracker
    from snks.agent.post_mortem import PostMortemLearner
    from snks.agent.vector_mpc_agent import run_vector_mpc_episode
    from snks.encoder.tile_segmenter import pick_device

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "demos" / "checkpoints" / "exp137" / "segmenter_9x9.pt",
    )
    parser.add_argument("--crop-world", action="store_true")
    parser.add_argument("--perception-mode", choices=("pixel", "symbolic"), default="pixel")
    args = parser.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / f"diag_stage89_resource_trace_seed{args.seed}.json"
    device = torch.device(pick_device())
    model_dim = 16384
    n_locations = 50000
    config = _stage89_mode_config("current")

    if args.perception_mode == "pixel":
        model, segmenter, textbook_path = _build_model_and_segmenter(
            model_dim=model_dim,
            n_locations=n_locations,
            seed=42,
            device=device,
            checkpoint_path=args.checkpoint,
            crop_world=args.crop_world,
        )
    else:
        from snks.agent.vector_bootstrap import load_from_textbook
        from snks.agent.vector_world_model import VectorWorldModel

        textbook_path = ROOT / "configs" / "crafter_textbook.yaml"
        model = VectorWorldModel(dim=model_dim, n_locations=n_locations, seed=42, device=device)
        load_from_textbook(model, textbook_path)
        segmenter = None

    tb = CrafterTextbook(str(textbook_path))
    tracker = HomeostaticTracker()
    tracker.init_from_textbook(tb.body_block)
    learner = PostMortemLearner()
    stimuli = learner.build_stimuli(
        ["health", "food", "drink", "energy"],
        include_vital_delta=config["include_vital_delta"],
    )
    env = CrafterPixelEnv(seed=args.seed)

    metrics = run_vector_mpc_episode(
        env=env,
        segmenter=segmenter,
        model=model,
        tracker=tracker,
        max_steps=args.max_steps,
        stimuli=stimuli,
        textbook=tb,
        verbose=True,
        enable_dynamic_threat_model=config["enable_dynamic_threat_model"],
        enable_dynamic_threat_goals=config["enable_dynamic_threat_goals"],
        enable_motion_plans=config["enable_motion_plans"],
        enable_motion_chains=config["enable_motion_chains"],
        enable_post_plan_passive_rollout=config["enable_post_plan_passive_rollout"],
        perception_mode=args.perception_mode,
        record_step_trace=True,
    )

    trace = metrics.get("step_trace", [])
    tree_focus = [
        event for event in trace
        if event.get("plan_target") == "tree"
        or event.get("primitive") == "do"
        or event.get("near_concept") == "tree"
    ]
    frustrated_tree_do = [
        event for event in tree_focus
        if event.get("primitive") == "do" and int(event.get("wood_gain", 0)) <= 0
    ]
    successful_tree_do = [
        event for event in tree_focus
        if event.get("primitive") == "do" and int(event.get("wood_gain", 0)) > 0
    ]

    payload = {
        "seed": args.seed,
        "max_steps": args.max_steps,
        "perception_mode": args.perception_mode,
        "checkpoint": str(args.checkpoint),
        "summary": {
            "episode_steps": metrics.get("episode_steps", 0),
            "death_cause": metrics.get("death_cause", "unknown"),
            "n_tree_focus_steps": len(tree_focus),
            "n_frustrated_tree_do": len(frustrated_tree_do),
            "n_successful_tree_do": len(successful_tree_do),
        },
        "tree_focus_steps": tree_focus,
        "frustrated_tree_do_steps": frustrated_tree_do,
        "successful_tree_do_steps": successful_tree_do,
    }
    out_path.write_text(json.dumps(payload, indent=2, default=_json_default))
    print(
        f"saved trace: {out_path} "
        f"(tree_focus={len(tree_focus)} frustrated_do={len(frustrated_tree_do)} successful_do={len(successful_tree_do)})"
    )


if __name__ == "__main__":
    main()
