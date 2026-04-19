"""Stage 90 quick-slice death collection.

Collect a fast bundle of post-Stage-89 death episodes with passive planner
diagnostics attached. This script does not modify planner behavior.

Writes:
  - _docs/stage90_baseline_reference.json
  - _docs/stage90_quick_slice.json
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from stage89_eval import _build_model_and_segmenter, _stage89_mode_config

ROOT = Path(__file__).parent.parent
DOCS_DIR = ROOT / "_docs"
BASELINE_REF_PATH = DOCS_DIR / "stage90_baseline_reference.json"
QUICK_SLICE_PATH = DOCS_DIR / "stage90_quick_slice.json"
STAGE89_REFERENCE_PATHS = (
    DOCS_DIR / "stage89_eval.json",
    DOCS_DIR / "stage89_baseline.json",
)


def _json_default(value: Any) -> Any:
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


def load_stage89_baseline_reference() -> dict[str, Any]:
    reference = {
        "preferred_paths": [str(path) for path in STAGE89_REFERENCE_PATHS],
        "available": False,
        "path": None,
        "summary": None,
    }
    for path in STAGE89_REFERENCE_PATHS:
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        reference["available"] = True
        reference["path"] = str(path)
        reference["summary"] = payload.get("summary", payload)
        break
    return reference


def _build_runtime(seed: int, checkpoint: Path, crop_world: bool) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
    from snks.agent.crafter_textbook import CrafterTextbook
    from snks.agent.post_mortem import PostMortemLearner
    from snks.agent.vector_bootstrap import load_from_textbook
    from snks.agent.vector_world_model import VectorWorldModel
    from snks.encoder.tile_segmenter import pick_device

    device = torch.device(pick_device())
    config = _stage89_mode_config("current")
    model_dim = 16384
    n_locations = 50000

    if checkpoint:
        model, segmenter, textbook_path = _build_model_and_segmenter(
            model_dim=model_dim,
            n_locations=n_locations,
            seed=seed,
            device=device,
            checkpoint_path=checkpoint,
            crop_world=crop_world,
        )
    else:
        textbook_path = ROOT / "configs" / "crafter_textbook.yaml"
        model = VectorWorldModel(dim=model_dim, n_locations=n_locations, seed=seed, device=device)
        load_from_textbook(model, textbook_path)
        segmenter = None

    tb = CrafterTextbook(str(textbook_path))
    learner = PostMortemLearner()
    stimuli = learner.build_stimuli(
        ["health", "food", "drink", "energy"],
        include_vital_delta=config["include_vital_delta"],
    )
    return model, segmenter, tb, None, {"config": config, "stimuli": stimuli, "learner": learner}


def main() -> None:
    from snks.agent.crafter_pixel_env import CrafterPixelEnv
    from snks.agent.perception import HomeostaticTracker
    from snks.agent.post_mortem import PostMortemAnalyzer
    from snks.agent.vector_mpc_agent import run_vector_mpc_episode

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-deaths", type=int, default=30)
    parser.add_argument("--max-episodes", type=int, default=120)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--capture-steps", type=int, default=8)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "demos" / "checkpoints" / "exp137" / "segmenter_9x9.pt",
    )
    parser.add_argument("--crop-world", action="store_true")
    parser.add_argument("--perception-mode", choices=("pixel", "symbolic"), default="pixel")
    args = parser.parse_args()

    DOCS_DIR.mkdir(exist_ok=True)
    baseline_reference = load_stage89_baseline_reference()
    BASELINE_REF_PATH.write_text(json.dumps(baseline_reference, indent=2, default=_json_default))

    model, segmenter, tb, _tracker_unused, runtime = _build_runtime(
        seed=args.seed,
        checkpoint=args.checkpoint if args.perception_mode == "pixel" else None,
        crop_world=args.crop_world,
    )
    config = runtime["config"]
    learner = runtime["learner"]
    stimuli = runtime["stimuli"]
    analyzer = PostMortemAnalyzer()

    bundles: list[dict[str, Any]] = []
    death_causes: Counter[str] = Counter()
    t0 = time.time()

    for ep in range(args.max_episodes):
        if len(bundles) >= args.n_deaths:
            break

        ep_seed = args.seed + ep
        env = CrafterPixelEnv(seed=ep_seed)
        tracker = HomeostaticTracker()
        tracker.init_from_textbook(tb.body_block)
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
            record_death_bundle=True,
            death_capture_steps=args.capture_steps,
        )

        damage_log = metrics.get("damage_log", [])
        attribution = analyzer.attribute(damage_log, metrics.get("episode_steps", 0))
        learner.update(attribution)
        stimuli = learner.build_stimuli(
            ["health", "food", "drink", "energy"],
            include_vital_delta=config["include_vital_delta"],
        )

        death_cause = metrics.get("death_cause", "alive")
        death_causes[death_cause] += 1
        bundle = metrics.get("death_trace_bundle")
        if bundle:
            bundle["seed"] = ep_seed
            bundle["episode_id"] = ep
            bundle["action_entropy"] = metrics.get("action_entropy", 0.0)
            bundle["total_surprise"] = metrics.get("total_surprise", 0.0)
            bundles.append(bundle)

        elapsed = time.time() - t0
        print(
            f"ep{ep:3d}: len={metrics.get('episode_steps', 0):4.0f} "
            f"death={death_cause:12s} bundles={len(bundles):2d}/{args.n_deaths} "
            f"[{elapsed:.0f}s]"
        )

    payload = {
        "stage": "stage90_quick_slice",
        "mode": "current_stage89_stack",
        "baseline_reference": baseline_reference,
        "config": {
            "seed": args.seed,
            "n_deaths": args.n_deaths,
            "max_episodes": args.max_episodes,
            "max_steps": args.max_steps,
            "capture_steps": args.capture_steps,
            "checkpoint": str(args.checkpoint),
            "crop_world": bool(args.crop_world),
            "perception_mode": args.perception_mode,
        },
        "summary": {
            "episodes_run": ep + 1 if args.max_episodes > 0 else 0,
            "deaths_captured": len(bundles),
            "death_cause_breakdown": dict(death_causes),
            "unknown_bundle_count": sum(
                1 for bundle in bundles if bundle.get("primary_error_label") == "unknown"
            ),
        },
        "bundles": bundles,
    }
    QUICK_SLICE_PATH.write_text(json.dumps(payload, indent=2, default=_json_default))
    print(f"saved quick slice: {QUICK_SLICE_PATH}")


if __name__ == "__main__":
    main()
