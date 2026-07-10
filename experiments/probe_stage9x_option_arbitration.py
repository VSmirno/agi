"""Stage9X-B paired option-arbitration probe.

This is a non-video diagnostic gate for the learned option arbitration loop:

  gen1: run one episode with option outcome learning enabled and stimulus off;
        persist the VectorWorldModel memory for this seed.
  gen2: run the same seed with learning + stimulus enabled, loading gen1 memory;
        compare traces for option-outcome recall/scoring and first divergence.

The probe intentionally does not use the local actor/evaluator checkpoint. It
exercises the planner/world-model/option-outcome path through
``run_vector_mpc_episode`` directly.
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(ROOT / "experiments"))

from stage90_quick_slice import _build_runtime, _json_default  # noqa: E402
from stage90r_eval_local_policy import _runtime_profile  # noqa: E402


def _summarize_trace(trace: list[dict[str, Any]]) -> dict[str, Any]:
    option_counts: Counter[str] = Counter()
    plan_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    recall_count = 0
    used_for_scoring = 0
    negative_by_died_to: Counter[str] = Counter()
    scored_rows: list[dict[str, Any]] = []

    for row in trace:
        option = row.get("strategy_option") or {}
        option_id = option.get("option_id") or option.get("id")
        if option_id:
            option_counts[str(option_id)] += 1
        plan_origin = row.get("plan_origin")
        if plan_origin:
            plan_counts[str(plan_origin)] += 1
        action = row.get("action")
        if action:
            action_counts[str(action)] += 1
        recall = row.get("option_outcome_recall")
        if not recall:
            continue
        recall_count += 1
        decoded = recall.get("decoded") or {}
        if bool(recall.get("used_for_scoring")):
            used_for_scoring += 1
            died_to = str(decoded.get("died_to") or "unknown")
            negative_by_died_to[died_to] += 1
            if len(scored_rows) < 20:
                scored_rows.append(
                    {
                        "step": row.get("step"),
                        "action": action,
                        "plan_origin": plan_origin,
                        "option_id": recall.get("option_id"),
                        "confidence": recall.get("confidence"),
                        "decoded": decoded,
                        "goal": row.get("goal"),
                        "option_context": row.get("option_context"),
                    }
                )

    return {
        "steps": len(trace),
        "strategy_option_counts": dict(option_counts.most_common()),
        "plan_origin_counts": dict(plan_counts.most_common()),
        "action_counts": dict(action_counts.most_common()),
        "option_outcome_recall": int(recall_count),
        "option_outcome_used_for_scoring": int(used_for_scoring),
        "negative_recall_by_died_to": dict(negative_by_died_to.most_common()),
        "first_scored_rows": scored_rows,
    }


def _first_divergence(
    trace_a: list[dict[str, Any]],
    trace_b: list[dict[str, Any]],
) -> dict[str, Any] | None:
    limit = min(len(trace_a), len(trace_b))
    keys = ("action", "plan_origin", "selected_controller")
    for idx in range(limit):
        a = trace_a[idx]
        b = trace_b[idx]
        option_a_obj = (a.get("strategy_option") or {})
        option_b_obj = (b.get("strategy_option") or {})
        option_a = option_a_obj.get("option_id") or option_a_obj.get("id")
        option_b = option_b_obj.get("option_id") or option_b_obj.get("id")
        changed = {key: (a.get(key), b.get(key)) for key in keys if a.get(key) != b.get(key)}
        if option_a != option_b:
            changed["strategy_option"] = (option_a, option_b)
        if changed:
            return {
                "step_index": idx,
                "gen1_step": a.get("step"),
                "gen2_step": b.get("step"),
                "changed": changed,
                "gen1": {
                    "body": a.get("body"),
                    "goal": a.get("goal"),
                    "option_context": a.get("option_context"),
                    "option_outcome_recall": a.get("option_outcome_recall"),
                },
                "gen2": {
                    "body": b.get("body"),
                    "goal": b.get("goal"),
                    "option_context": b.get("option_context"),
                    "option_outcome_recall": b.get("option_outcome_recall"),
                },
            }
    if len(trace_a) != len(trace_b):
        return {
            "step_index": limit,
            "gen1_len": len(trace_a),
            "gen2_len": len(trace_b),
            "changed": {"trace_length": (len(trace_a), len(trace_b))},
        }
    return None


def _run_one(
    *,
    label: str,
    seed: int,
    max_steps: int,
    smoke_lite: bool,
    world_model_dir: Path,
    enable_stimulus: bool,
    option_outcome_horizon: int,
    option_outcome_weight: float,
    option_outcome_confidence_floor: float,
) -> dict[str, Any]:
    from snks.agent.crafter_pixel_env import CrafterPixelEnv
    from snks.agent.perception import HomeostaticTracker
    from snks.agent.vector_mpc_agent import run_vector_mpc_episode

    profile = _runtime_profile(smoke_lite=smoke_lite)
    model, segmenter, textbook, _tracker_unused, runtime = _build_runtime(
        seed=seed,
        checkpoint=None,
        crop_world=False,
        model_dim=int(profile["model_dim"]),
        n_locations=int(profile["n_locations"]),
    )
    tracker = HomeostaticTracker()
    tracker.init_from_textbook(textbook.body_block)
    metrics = run_vector_mpc_episode(
        env=CrafterPixelEnv(seed=seed),
        segmenter=segmenter,
        model=model,
        tracker=tracker,
        max_steps=max_steps,
        horizon=int(profile["planner_horizon"]),
        beam_width=int(profile["beam_width"]),
        max_depth=int(profile["max_depth"]),
        stimuli=runtime["stimuli"],
        textbook=textbook,
        verbose=False,
        enable_dynamic_threat_model=runtime["config"]["enable_dynamic_threat_model"],
        enable_dynamic_threat_goals=runtime["config"]["enable_dynamic_threat_goals"],
        enable_motion_plans=runtime["config"]["enable_motion_plans"],
        enable_motion_chains=runtime["config"]["enable_motion_chains"],
        enable_post_plan_passive_rollout=bool(profile["enable_post_plan_passive_rollout"]),
        perception_mode="symbolic",
        record_local_trace=True,
        # The causal question here is option recall/scoring/divergence, not
        # local counterfactual inspection. Full-profile counterfactual payloads
        # materially increase memory pressure and have killed HyperPC runs with
        # exit 137, so this probe keeps the trace focused.
        record_local_counterfactuals=False,
        local_counterfactual_horizon=1,
        record_death_bundle=True,
        death_capture_steps=20,
        enable_option_outcome_learning=True,
        option_outcome_horizon=option_outcome_horizon,
        enable_option_outcome_stimulus=enable_stimulus,
        option_outcome_stimulus_weight=option_outcome_weight,
        option_outcome_confidence_floor=option_outcome_confidence_floor,
        world_model_path=world_model_dir / f"seed{seed}.pt",
        rng=np.random.RandomState(seed),
    )
    trace = list(metrics.get("local_trace", []))
    last_trace_row = trace[-1] if trace else {}
    result = {
        "label": label,
        "metrics": {
            "episode_steps": metrics.get("episode_steps"),
            "death_cause": metrics.get("death_cause"),
            "terminated_done": bool(last_trace_row.get("done_after_step", False)),
            "final_body": dict(last_trace_row.get("body_after") or {}),
            "action_counts": metrics.get("action_counts"),
            "controller_distribution": metrics.get("controller_distribution"),
        },
        "summary": _summarize_trace(trace),
        "local_trace": trace,
        "death_trace_bundle": metrics.get("death_trace_bundle"),
    }
    del metrics, model, segmenter, tracker
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-steps", type=int, default=220)
    parser.add_argument("--gen1-max-steps", type=int, default=None)
    parser.add_argument("--gen2-max-steps", type=int, default=None)
    parser.add_argument("--smoke-lite", action="store_true")
    parser.add_argument("--out", type=Path, default=ROOT / "output_to_user" / "stage9x_option_arbitration_probe" / "paired_seed17.json")
    parser.add_argument("--world-model-dir", type=Path, default=None)
    parser.add_argument("--option-outcome-horizon", type=int, default=5)
    parser.add_argument("--option-outcome-weight", type=float, default=1.0)
    parser.add_argument("--option-outcome-confidence-floor", type=float, default=0.25)
    args = parser.parse_args()

    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    world_model_dir = args.world_model_dir or (out_path.parent / "world_model")
    if world_model_dir.exists():
        shutil.rmtree(world_model_dir)
    world_model_dir.mkdir(parents=True, exist_ok=True)

    gen1 = _run_one(
        label="gen1_writer_only",
        seed=int(args.seed),
        max_steps=int(args.gen1_max_steps or args.max_steps),
        smoke_lite=bool(args.smoke_lite),
        world_model_dir=world_model_dir,
        enable_stimulus=False,
        option_outcome_horizon=int(args.option_outcome_horizon),
        option_outcome_weight=float(args.option_outcome_weight),
        option_outcome_confidence_floor=float(args.option_outcome_confidence_floor),
    )
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gen2 = _run_one(
        label="gen2_reader_writer",
        seed=int(args.seed),
        max_steps=int(args.gen2_max_steps or args.max_steps),
        smoke_lite=bool(args.smoke_lite),
        world_model_dir=world_model_dir,
        enable_stimulus=True,
        option_outcome_horizon=int(args.option_outcome_horizon),
        option_outcome_weight=float(args.option_outcome_weight),
        option_outcome_confidence_floor=float(args.option_outcome_confidence_floor),
    )
    payload = {
        "stage": "stage9x_option_arbitration_probe",
        "claim_scope": "non-video paired trace validation; not a Stage PASS or AGI claim",
        "config": {
            "seed": int(args.seed),
            "max_steps": int(args.max_steps),
            "gen1_max_steps": int(args.gen1_max_steps or args.max_steps),
            "gen2_max_steps": int(args.gen2_max_steps or args.max_steps),
            "smoke_lite": bool(args.smoke_lite),
            "world_model_dir": str(world_model_dir),
            "option_outcome_horizon": int(args.option_outcome_horizon),
            "option_outcome_weight": float(args.option_outcome_weight),
            "option_outcome_confidence_floor": float(args.option_outcome_confidence_floor),
        },
        "gen1": gen1,
        "gen2": gen2,
        "comparison": {
            "first_divergence": _first_divergence(gen1["local_trace"], gen2["local_trace"]),
            "causal_signal_present": bool(
                gen2["summary"]["option_outcome_used_for_scoring"] > 0
            ),
        },
    }
    out_path.write_text(json.dumps(payload, indent=2, default=_json_default))
    print(json.dumps({
        "out": str(out_path),
        "gen1": gen1["metrics"],
        "gen1_summary": gen1["summary"],
        "gen2": gen2["metrics"],
        "gen2_summary": gen2["summary"],
        "comparison": payload["comparison"],
    }, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
