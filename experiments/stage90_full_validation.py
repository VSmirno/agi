"""Stage 90 full-validation run for the chosen death diagnosis.

Reads:
  - optional _docs/stage90_quick_slice_summary.json

Writes:
  - _docs/stage90_full_validation.json
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from analyze_stage90_deaths import DEFAULT_OUTPUT_PATH as QUICK_SUMMARY_PATH
from stage90_quick_slice import _build_runtime, _json_default, load_stage89_baseline_reference

ROOT = Path(__file__).parent.parent
DOCS_DIR = ROOT / "_docs"
OUT_PATH = DOCS_DIR / "stage90_full_validation.json"


def main() -> None:
    from snks.agent.crafter_pixel_env import CrafterPixelEnv
    from snks.agent.perception import HomeostaticTracker
    from snks.agent.post_mortem import PostMortemAnalyzer
    from snks.agent.stage90_diagnostics import summarize_failure_buckets
    from snks.agent.vector_mpc_agent import run_vector_mpc_episode

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=142)
    parser.add_argument("--capture-steps", type=int, default=8)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "demos" / "checkpoints" / "exp137" / "segmenter_9x9.pt",
    )
    parser.add_argument("--crop-world", action="store_true")
    parser.add_argument("--perception-mode", choices=("pixel", "symbolic"), default="pixel")
    parser.add_argument("--quick-summary", type=Path, default=QUICK_SUMMARY_PATH)
    args = parser.parse_args()

    DOCS_DIR.mkdir(exist_ok=True)
    baseline_reference = load_stage89_baseline_reference()
    expected_dominant = None
    if args.quick_summary.exists():
        quick_summary = json.loads(args.quick_summary.read_text())
        expected_dominant = (
            quick_summary.get("analysis_summary", {}) or {}
        ).get("dominant_bucket")

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

    for ep in range(args.n_episodes):
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
            f"ep{ep:2d}: len={metrics.get('episode_steps', 0):4.0f} "
            f"death={death_cause:12s} bundles={len(bundles):2d} [{elapsed:.0f}s]"
        )

    analysis_summary = summarize_failure_buckets(bundles)
    payload = {
        "stage": "stage90_full_validation",
        "mode": "current_stage89_stack",
        "baseline_reference": baseline_reference,
        "expected_dominant_bucket": expected_dominant,
        "config": {
            "seed": args.seed,
            "n_episodes": args.n_episodes,
            "max_steps": args.max_steps,
            "capture_steps": args.capture_steps,
            "checkpoint": str(args.checkpoint),
            "crop_world": bool(args.crop_world),
            "perception_mode": args.perception_mode,
        },
        "collection_summary": {
            "episodes_run": args.n_episodes,
            "deaths_captured": len(bundles),
            "death_cause_breakdown": dict(death_causes),
        },
        "analysis_summary": analysis_summary,
        "diagnosis_agreement": (
            analysis_summary.get("dominant_bucket") == expected_dominant
            if expected_dominant is not None
            else None
        ),
        "bundles": bundles,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2, default=_json_default))
    print(
        f"saved full validation: {OUT_PATH} "
        f"(dominant={analysis_summary['dominant_bucket']} agreement={payload['diagnosis_agreement']})"
    )


if __name__ == "__main__":
    main()
