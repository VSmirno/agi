"""Stage 90R viewport-first local dataset collection.

Collect passive per-step local observations from the current Stage 89 stack and
post-process them into fixed-horizon local training examples.

Writes:
  - _docs/stage90r_baseline_reference.json
  - _docs/stage90r_local_dataset.json
  - _docs/stage90r_local_dataset_summary.json
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from stage90_quick_slice import _build_runtime, _json_default, load_stage89_baseline_reference

ROOT = Path(__file__).parent.parent
DOCS_DIR = ROOT / "_docs"
BASELINE_REF_PATH = DOCS_DIR / "stage90r_baseline_reference.json"
DATASET_PATH = DOCS_DIR / "stage90r_local_dataset.json"
SUMMARY_PATH = DOCS_DIR / "stage90r_local_dataset_summary.json"


def _summarize_action_distribution(samples: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter(sample.get("action", "unknown") for sample in samples)
    return dict(counts)


def main() -> None:
    from snks.agent.crafter_pixel_env import CrafterPixelEnv
    from snks.agent.perception import HomeostaticTracker
    from snks.agent.post_mortem import PostMortemAnalyzer
    from snks.agent.stage90r_local_policy import (
        build_local_training_examples,
        local_dataset_metadata,
    )
    from snks.agent.vector_mpc_agent import run_vector_mpc_episode

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-episodes", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--crop-world", action="store_true")
    parser.add_argument("--perception-mode", choices=("pixel", "symbolic"), default="pixel")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "demos" / "checkpoints" / "exp137" / "segmenter_9x9.pt",
    )
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

    episode_artifacts: list[dict[str, Any]] = []
    all_samples: list[dict[str, Any]] = []
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
            record_local_trace=True,
        )

        damage_log = metrics.get("damage_log", [])
        attribution = analyzer.attribute(damage_log, metrics.get("episode_steps", 0))
        learner.update(attribution)
        stimuli = learner.build_stimuli(
            ["health", "food", "drink", "energy"],
            include_vital_delta=config["include_vital_delta"],
        )

        local_trace = metrics.get("local_trace", [])
        samples = build_local_training_examples(
            local_trace=local_trace,
            final_body=metrics.get("final_body", {}),
            final_inventory=metrics.get("final_inv", {}),
            seed=ep_seed,
            episode_id=ep,
            horizon=args.horizon,
        )
        all_samples.extend(samples)

        death_cause = metrics.get("death_cause", "alive")
        death_causes[death_cause] += 1
        episode_artifacts.append(
            {
                "episode_id": ep,
                "seed": ep_seed,
                "episode_steps": int(metrics.get("episode_steps", 0)),
                "death_cause": death_cause,
                "n_local_trace_steps": len(local_trace),
                "n_samples": len(samples),
            }
        )

        elapsed = time.time() - t0
        print(
            f"ep{ep:3d}: len={metrics.get('episode_steps', 0):4.0f} "
            f"death={death_cause:12s} samples={len(samples):4d} "
            f"total={len(all_samples):5d} [{elapsed:.0f}s]"
        )

    payload = {
        "stage": "stage90r_local_dataset",
        "mode": "current_stage89_stack",
        "baseline_reference": baseline_reference,
        "metadata": local_dataset_metadata(args.horizon),
        "config": {
            "seed": args.seed,
            "n_episodes": args.n_episodes,
            "max_steps": args.max_steps,
            "horizon": args.horizon,
            "checkpoint": str(args.checkpoint),
            "crop_world": bool(args.crop_world),
            "perception_mode": args.perception_mode,
        },
        "episodes": episode_artifacts,
        "samples": all_samples,
    }
    DATASET_PATH.write_text(json.dumps(payload, indent=2, default=_json_default))

    summary = {
        "stage": "stage90r_local_dataset_summary",
        "dataset_path": str(DATASET_PATH),
        "baseline_reference": baseline_reference,
        "config": payload["config"],
        "metadata": payload["metadata"],
        "summary": {
            "episodes_run": len(episode_artifacts),
            "samples_collected": len(all_samples),
            "avg_episode_steps": round(
                sum(ep["episode_steps"] for ep in episode_artifacts) / max(len(episode_artifacts), 1),
                2,
            ),
            "death_cause_breakdown": dict(death_causes),
            "action_distribution": _summarize_action_distribution(all_samples),
        },
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, default=_json_default))
    print(f"saved dataset: {DATASET_PATH}")
    print(f"saved summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
