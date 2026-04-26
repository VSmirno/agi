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
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from stage90_quick_slice import _build_runtime, _json_default, load_stage89_baseline_reference

ROOT = Path(__file__).parent.parent
DOCS_DIR = ROOT / "_docs"
BASELINE_REF_PATH = DOCS_DIR / "stage90r_baseline_reference.json"
DATASET_PATH = DOCS_DIR / "stage90r_local_dataset.json"
SUMMARY_PATH = DOCS_DIR / "stage90r_local_dataset_summary.json"

_FULL_COLLECTION_PROFILE = {
    "name": "full",
    "runtime_model_dim": 16384,
    "runtime_n_locations": 50000,
    "planner_horizon": 10,
    "beam_width": 5,
    "max_depth": 3,
    "enable_post_plan_passive_rollout": True,
    "record_local_counterfactuals": True,
    "local_counterfactual_horizon": 3,
    "verbose": True,
}
_SMOKE_LITE_COLLECTION_PROFILE = {
    "name": "smoke_lite",
    "runtime_model_dim": 2048,
    "runtime_n_locations": 5000,
    "planner_horizon": 4,
    "beam_width": 2,
    "max_depth": 2,
    "enable_post_plan_passive_rollout": False,
    "record_local_counterfactuals": "salient_only",
    "local_counterfactual_horizon": 1,
    "verbose": False,
}


def _collection_profile(*, smoke_lite: bool) -> dict[str, Any]:
    return dict(_SMOKE_LITE_COLLECTION_PROFILE if smoke_lite else _FULL_COLLECTION_PROFILE)


def _episode_rng(seed: int) -> np.random.RandomState:
    return np.random.RandomState(int(seed))


def _summarize_action_distribution(samples: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter(sample.get("action", "unknown") for sample in samples)
    return dict(counts)


def _summarize_action_distribution_by_regime(samples: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    by_regime: dict[str, Counter[str]] = defaultdict(Counter)
    for sample in samples:
        regime = str(sample.get("primary_regime", "unknown"))
        by_regime[regime][str(sample.get("action", "unknown"))] += 1
    return {
        regime: dict(sorted(counter.items()))
        for regime, counter in sorted(by_regime.items())
    }


def _escape_label_coverage(samples: list[dict[str, Any]]) -> dict[str, Any]:
    valid = sum(1 for sample in samples if sample["label"].get("escape_delta_h") is not None)
    total = len(samples)
    return {
        "valid_count": valid,
        "masked_count": total - valid,
        "valid_fraction": round(valid / max(total, 1), 3),
    }


def _belief_target_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        return {
            "progress_delta_mean": 0.0,
            "stall_risk_mean": 0.0,
            "affordance_persistence_mean": 0.0,
            "threat_trend_mean": 0.0,
        }
    return {
        "progress_delta_mean": round(
            sum(float(sample["label"].get("progress_delta_h", 0.0)) for sample in samples) / len(samples),
            3,
        ),
        "stall_risk_mean": round(
            sum(float(sample["label"].get("stall_risk_h", 0.0)) for sample in samples) / len(samples),
            3,
        ),
        "affordance_persistence_mean": round(
            sum(float(sample["label"].get("affordance_persistence_h", 0.0)) for sample in samples) / len(samples),
            3,
        ),
        "threat_trend_mean": round(
            sum(float(sample["label"].get("threat_trend_h", 0.0)) for sample in samples) / len(samples),
            3,
        ),
    }


def _counterfactual_coverage(samples: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(samples)
    supported_samples = sum(1 for sample in samples if sample.get("counterfactual_outcomes"))
    action_counts: Counter[str] = Counter()
    for sample in samples:
        for outcome in sample.get("counterfactual_outcomes", []):
            action_counts[str(outcome.get("action", "unknown"))] += 1
    return {
        "state_samples_with_counterfactuals": supported_samples,
        "state_sample_fraction": round(supported_samples / max(total, 1), 3),
        "counterfactual_action_support": dict(sorted(action_counts.items())),
    }


def _state_centered_summary(state_samples: list[dict[str, Any]]) -> dict[str, Any]:
    candidate_histogram: Counter[int] = Counter()
    action_support: Counter[str] = Counter()
    regime_counts: Counter[str] = Counter()
    comparison_ready = 0
    counterfactual_ready = 0
    for state in state_samples:
        n_candidates = int(state["comparison_coverage"]["n_candidate_actions"])
        candidate_histogram[n_candidates] += 1
        if n_candidates >= 2:
            comparison_ready += 1
        if int(state["comparison_coverage"].get("n_counterfactual_actions", 0)) >= 2:
            counterfactual_ready += 1
        regime_counts[str(state.get("primary_regime", "unknown"))] += 1
        for action, support in state.get("chosen_action_support", {}).items():
            action_support[str(action)] += int(support)
    return {
        "n_state_samples": len(state_samples),
        "n_comparison_ready_states": comparison_ready,
        "n_counterfactual_ready_states": counterfactual_ready,
        "candidate_action_histogram": {
            str(count): value
            for count, value in sorted(candidate_histogram.items())
        },
        "candidate_support_by_action": dict(sorted(action_support.items())),
        "state_regime_counts": dict(sorted(regime_counts.items())),
    }


def _learner_transition_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    regime_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    planner_origin_counts: Counter[str] = Counter()
    controller_counts: Counter[str] = Counter()
    next_obs_missing = 0
    terminated = 0
    for record in records:
        regime_counts[str(record.get("primary_regime", "unknown"))] += 1
        action_counts[str(record.get("action", "unknown"))] += 1
        planner_origin_counts[str(record.get("plan_origin", "unknown"))] += 1
        controller_counts[str(record.get("controller", "unknown"))] += 1
        if not bool(record.get("next_observation_available", False)):
            next_obs_missing += 1
        if bool(record.get("terminated", False)):
            terminated += 1
    return {
        "n_records": len(records),
        "regime_counts": dict(sorted(regime_counts.items())),
        "action_distribution": dict(sorted(action_counts.items())),
        "plan_origin_distribution": dict(sorted(planner_origin_counts.items())),
        "controller_distribution": dict(sorted(controller_counts.items())),
        "terminal_transition_count": terminated,
        "next_observation_missing_count": next_obs_missing,
    }


def _planner_teacher_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    action_counts: Counter[str] = Counter()
    regime_counts: Counter[str] = Counter()
    planner_origin_counts: Counter[str] = Counter()
    teacher_mode_counts: Counter[str] = Counter()
    for record in records:
        action_counts[str(record.get("planner_action", "unknown"))] += 1
        regime_counts[str(record.get("primary_regime", "unknown"))] += 1
        planner_origin_counts[str(record.get("planner_plan_origin", "unknown"))] += 1
        teacher_mode_counts[str(record.get("teacher_mode", "unknown"))] += 1
    return {
        "n_records": len(records),
        "planner_action_distribution": dict(sorted(action_counts.items())),
        "regime_counts": dict(sorted(regime_counts.items())),
        "planner_plan_origin_distribution": dict(sorted(planner_origin_counts.items())),
        "teacher_mode_distribution": dict(sorted(teacher_mode_counts.items())),
        "bootstrap_only": set(teacher_mode_counts.keys()) <= {"planner_controlled_bootstrap"},
    }


def _rescue_record_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    trigger_counts: Counter[str] = Counter()
    improved_count = 0
    for record in records:
        trigger_counts[str(record.get("trigger", "unknown"))] += 1
        if record.get("rescue_improved_outcome") is True:
            improved_count += 1
    return {
        "n_records": len(records),
        "trigger_distribution": dict(sorted(trigger_counts.items())),
        "improved_count": improved_count,
    }


def _auxiliary_counterfactual_probe_summary(probes: list[dict[str, Any]]) -> dict[str, Any]:
    action_counts: Counter[str] = Counter()
    regime_counts: Counter[str] = Counter()
    for probe in probes:
        action_counts[str(probe.get("action", "unknown"))] += 1
        regime_counts[str(probe.get("primary_regime", "unknown"))] += 1
    return {
        "n_records": len(probes),
        "action_distribution": dict(sorted(action_counts.items())),
        "regime_counts": dict(sorted(regime_counts.items())),
    }


def main() -> None:
    from snks.agent.crafter_pixel_env import CrafterPixelEnv
    from snks.agent.perception import HomeostaticTracker
    from snks.agent.post_mortem import PostMortemAnalyzer
    from snks.agent.stage90r_local_model import load_local_evaluator_artifact
    from snks.agent.stage90r_local_policy import (
        build_auxiliary_counterfactual_probe_records,
        build_learner_transition_records,
        build_local_training_examples,
        build_planner_teacher_records,
        build_rescue_records,
        build_state_centered_training_examples,
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
        "--smoke-lite",
        action="store_true",
        help="Use a reduced world-model and search budget for a bounded local smoke dataset run.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "demos" / "checkpoints" / "exp137" / "segmenter_9x9.pt",
    )
    parser.add_argument("--baseline-ref-out", type=Path, default=BASELINE_REF_PATH)
    parser.add_argument("--dataset-out", type=Path, default=DATASET_PATH)
    parser.add_argument("--summary-out", type=Path, default=SUMMARY_PATH)
    parser.add_argument(
        "--control-mode",
        choices=("planner_bootstrap", "mixed_control"),
        default="planner_bootstrap",
    )
    parser.add_argument("--actor-checkpoint", type=Path, default=None)
    parser.add_argument("--actor-share", type=float, default=0.0)
    parser.add_argument("--enable-planner-rescue", action="store_true")
    parser.add_argument("--allow-offline-gate-failure", action="store_true")
    args = parser.parse_args()

    DOCS_DIR.mkdir(exist_ok=True)
    baseline_reference = load_stage89_baseline_reference()
    args.baseline_ref_out.write_text(json.dumps(baseline_reference, indent=2, default=_json_default))
    collection_profile = _collection_profile(smoke_lite=bool(args.smoke_lite))
    actor_model = None
    actor_artifact = None
    if args.control_mode == "mixed_control":
        if args.actor_checkpoint is None:
            raise ValueError("--actor-checkpoint is required for --control-mode mixed_control")
        actor_model, actor_artifact = load_local_evaluator_artifact(args.actor_checkpoint, device="cpu")
        offline_gate = dict(actor_artifact.get("offline_gate", {}))
        if not bool(offline_gate.get("passed", False)) and not bool(args.allow_offline_gate_failure):
            raise ValueError(
                "Actor checkpoint offline gate failed; pass --allow-offline-gate-failure for diagnostic mixed-control collection."
            )

    model, segmenter, tb, _tracker_unused, runtime = _build_runtime(
        seed=args.seed,
        checkpoint=args.checkpoint if args.perception_mode == "pixel" else None,
        crop_world=args.crop_world,
        model_dim=int(collection_profile["runtime_model_dim"]),
        n_locations=int(collection_profile["runtime_n_locations"]),
    )
    config = runtime["config"]
    learner = runtime["learner"]
    stimuli = runtime["stimuli"]
    analyzer = PostMortemAnalyzer()

    episode_artifacts: list[dict[str, Any]] = []
    learner_transition_records: list[dict[str, Any]] = []
    planner_teacher_records: list[dict[str, Any]] = []
    rescue_records: list[dict[str, Any]] = []
    all_samples: list[dict[str, Any]] = []
    death_causes: Counter[str] = Counter()
    t0 = time.time()

    for ep in range(args.n_episodes):
        ep_seed = args.seed + ep
        env = CrafterPixelEnv(seed=ep_seed)
        tracker = HomeostaticTracker()
        tracker.init_from_textbook(tb.body_block)
        episode_rng = _episode_rng(ep_seed)
        metrics = run_vector_mpc_episode(
            env=env,
            segmenter=segmenter,
            model=model,
            tracker=tracker,
            rng=episode_rng,
            max_steps=args.max_steps,
            horizon=int(collection_profile["planner_horizon"]),
            beam_width=int(collection_profile["beam_width"]),
            max_depth=int(collection_profile["max_depth"]),
            stimuli=stimuli,
            textbook=tb,
            verbose=bool(collection_profile["verbose"]),
            enable_dynamic_threat_model=config["enable_dynamic_threat_model"],
            enable_dynamic_threat_goals=config["enable_dynamic_threat_goals"],
            enable_motion_plans=config["enable_motion_plans"],
            enable_motion_chains=config["enable_motion_chains"],
            enable_post_plan_passive_rollout=bool(collection_profile["enable_post_plan_passive_rollout"]),
            perception_mode=args.perception_mode,
            record_local_trace=True,
            record_local_counterfactuals=collection_profile["record_local_counterfactuals"],
            local_counterfactual_horizon=int(collection_profile["local_counterfactual_horizon"]),
            local_actor_policy=actor_model,
            mixed_control_actor_share=float(args.actor_share) if args.control_mode == "mixed_control" else 0.0,
            enable_planner_rescue=bool(args.enable_planner_rescue and args.control_mode == "mixed_control"),
        )

        damage_log = metrics.get("damage_log", [])
        attribution = analyzer.attribute(damage_log, metrics.get("episode_steps", 0))
        learner.update(attribution)
        stimuli = learner.build_stimuli(
            ["health", "food", "drink", "energy"],
            include_vital_delta=config["include_vital_delta"],
        )

        local_trace = metrics.get("local_trace", [])
        rescue_trace = metrics.get("rescue_trace", [])
        transition_records = build_learner_transition_records(
            local_trace=local_trace,
            final_body=metrics.get("final_body", {}),
            final_inventory=metrics.get("final_inv", {}),
            seed=ep_seed,
            episode_id=ep,
            horizon=args.horizon,
        )
        learner_transition_records.extend(transition_records)
        planner_teacher_records.extend(build_planner_teacher_records(transition_records))
        rescue_records.extend(
            build_rescue_records(
                rescue_trace=rescue_trace,
                seed=ep_seed,
                episode_id=ep,
            )
        )
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

    state_samples = build_state_centered_training_examples(all_samples)
    auxiliary_counterfactual_probes = build_auxiliary_counterfactual_probe_records(learner_transition_records)
    payload = {
        "stage": "stage90r_planner_teacher_hybrid_dataset",
        "mode": args.control_mode,
        "baseline_reference": baseline_reference,
        "metadata": local_dataset_metadata(args.horizon),
        "config": {
            "seed": args.seed,
            "n_episodes": args.n_episodes,
            "max_steps": args.max_steps,
            "horizon": args.horizon,
            "checkpoint": str(args.checkpoint),
            "control_mode": args.control_mode,
            "actor_checkpoint": str(args.actor_checkpoint) if args.actor_checkpoint is not None else None,
            "actor_share": float(args.actor_share),
            "enable_planner_rescue": bool(args.enable_planner_rescue),
            "allow_offline_gate_failure": bool(args.allow_offline_gate_failure),
            "crop_world": bool(args.crop_world),
            "perception_mode": args.perception_mode,
            "collection_profile": collection_profile,
        },
        "episodes": episode_artifacts,
        "learner_transition_records": learner_transition_records,
        "planner_teacher_records": planner_teacher_records,
        "rescue_records": rescue_records,
        "auxiliary_counterfactual_probes": auxiliary_counterfactual_probes,
        "auxiliary_action_samples": all_samples,
        "auxiliary_state_samples": state_samples,
        "samples": all_samples,
        "state_samples": state_samples,
    }
    args.dataset_out.write_text(json.dumps(payload, indent=2, default=_json_default))

    summary = {
        "stage": "stage90r_planner_teacher_hybrid_dataset_summary",
        "dataset_path": str(args.dataset_out),
        "baseline_reference": baseline_reference,
        "config": payload["config"],
        "metadata": payload["metadata"],
        "actor_checkpoint_offline_gate": actor_artifact.get("offline_gate") if actor_artifact is not None else None,
        "summary": {
            "episodes_run": len(episode_artifacts),
            "samples_collected": len(all_samples),
            "avg_episode_steps": round(
                sum(ep["episode_steps"] for ep in episode_artifacts) / max(len(episode_artifacts), 1),
                2,
            ),
            "death_cause_breakdown": dict(death_causes),
            "learner_transitions": _learner_transition_summary(learner_transition_records),
            "planner_teacher_records": _planner_teacher_summary(planner_teacher_records),
            "rescue_records": _rescue_record_summary(rescue_records),
            "auxiliary_counterfactual_probes": _auxiliary_counterfactual_probe_summary(auxiliary_counterfactual_probes),
            "auxiliary_state_centered": _state_centered_summary(state_samples),
            "auxiliary_action_samples": {
                "n_records": len(all_samples),
                "action_distribution": _summarize_action_distribution(all_samples),
                "action_distribution_by_regime": _summarize_action_distribution_by_regime(all_samples),
                "escape_label_coverage": _escape_label_coverage(all_samples),
                "belief_target_summary": _belief_target_summary(all_samples),
                "counterfactual_coverage": _counterfactual_coverage(all_samples),
            },
        },
    }
    args.summary_out.write_text(json.dumps(summary, indent=2, default=_json_default))
    print(f"saved dataset: {args.dataset_out}")
    print(f"saved summary: {args.summary_out}")


if __name__ == "__main__":
    main()
