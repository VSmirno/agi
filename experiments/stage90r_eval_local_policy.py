"""Stage 90R local-only canary and planner-advisory evaluation."""

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
LOCAL_ONLY_OUT_PATH = DOCS_DIR / "stage90r_local_only_eval.json"
PLANNER_ADVISORY_OUT_PATH = DOCS_DIR / "stage90r_planner_advisory_eval.json"
STAGE90R_RESCUE_BASELINE_COMPARE_PATH = (
    DOCS_DIR / "hyper_stage90r_mixed_control_rescue_compare" / "comparison_summary.json"
)
_FULL_RUNTIME_PROFILE = {
    "model_dim": 16384,
    "n_locations": 50000,
    "planner_horizon": 10,
    "beam_width": 5,
    "max_depth": 3,
    "enable_post_plan_passive_rollout": True,
}
_SMOKE_LITE_RUNTIME_PROFILE = {
    "model_dim": 2048,
    "n_locations": 5000,
    "planner_horizon": 4,
    "beam_width": 2,
    "max_depth": 2,
    "enable_post_plan_passive_rollout": False,
}


def _allowed_primitives() -> list[str]:
    return ["move_left", "move_right", "move_up", "move_down", "do", "sleep"]


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _runtime_profile(*, smoke_lite: bool) -> dict[str, Any]:
    return dict(_SMOKE_LITE_RUNTIME_PROFILE if smoke_lite else _FULL_RUNTIME_PROFILE)


def _eval_episode_rng(*, base_seed: int, episode_index: int) -> np.random.RandomState:
    return np.random.RandomState(base_seed + episode_index)


def _action_summary(action_counts: Counter[str]) -> dict[str, Any]:
    return {
        "action_counts": dict(action_counts),
        "dominant_action": action_counts.most_common(1)[0][0] if action_counts else None,
        "dominant_action_share": round(
            action_counts.most_common(1)[0][1] / max(sum(action_counts.values()), 1),
            3,
        )
        if action_counts
        else 0.0,
    }


def _load_stage90r_rescue_baseline_reference() -> dict[str, Any]:
    reference = {
        "baseline_commit": "9083357",
        "comparison_path": str(STAGE90R_RESCUE_BASELINE_COMPARE_PATH),
        "available": False,
        "summary": None,
    }
    if not STAGE90R_RESCUE_BASELINE_COMPARE_PATH.exists():
        return reference
    payload = json.loads(STAGE90R_RESCUE_BASELINE_COMPARE_PATH.read_text())
    baseline = dict(payload.get("baseline_9083357", {}))
    reference["available"] = True
    reference["summary"] = baseline.get("summary")
    reference["eval_json"] = baseline.get("eval_json")
    reference["compare_metadata"] = {
        "mode": payload.get("mode"),
        "perception_mode": payload.get("perception_mode"),
        "smoke_lite": payload.get("smoke_lite"),
        "seed": payload.get("seed"),
        "n_episodes": payload.get("n_episodes"),
        "max_steps": payload.get("max_steps"),
    }
    return reference


def _summarize_planner_advisory_trace(trace: list[dict[str, Any]]) -> dict[str, Any]:
    base = _summarize_planner_advisory_trace_base(trace)
    threat_slice = [
        entry
        for entry in trace
        if "hostile_contact" in entry.get("regime_labels", [])
        or "hostile_near" in entry.get("regime_labels", [])
    ]
    resource_slice = [
        entry
        for entry in trace
        if "local_resource_facing" in entry.get("regime_labels", [])
    ]
    return {
        "n_advisory_steps": base["n_steps"],
        "agreement_rate": base["agreement_rate"],
        "disagreement_rate": base["disagreement_rate"],
        "mean_planner_rank_by_local_predictor": base["mean_planner_rank_by_local_predictor"],
        "mean_score_gap_to_advisory_best": base["mean_score_gap_to_advisory_best"],
        "planner_action_distribution": base["planner_action_distribution"],
        "advisory_best_action_distribution": base["advisory_best_action_distribution"],
        "threat_slice": _summarize_planner_advisory_slice(threat_slice),
        "resource_slice": _summarize_planner_advisory_slice(resource_slice),
    }


def _summarize_planner_advisory_slice(trace: list[dict[str, Any]]) -> dict[str, Any]:
    if not trace:
        return {
            "n_steps": 0,
            "agreement_rate": 0.0,
            "mean_planner_rank_by_local_predictor": 0.0,
            "mean_score_gap_to_advisory_best": 0.0,
            "planner_action_distribution": {},
            "advisory_best_action_distribution": {},
        }
    base = _summarize_planner_advisory_trace_base(trace)
    return {
        "n_steps": base["n_steps"],
        "agreement_rate": base["agreement_rate"],
        "mean_planner_rank_by_local_predictor": base["mean_planner_rank_by_local_predictor"],
        "mean_score_gap_to_advisory_best": base["mean_score_gap_to_advisory_best"],
        "planner_action_distribution": base["planner_action_distribution"],
        "advisory_best_action_distribution": base["advisory_best_action_distribution"],
    }


def _summarize_planner_advisory_trace_base(trace: list[dict[str, Any]]) -> dict[str, Any]:
    planner_actions: Counter[str] = Counter()
    advisory_actions: Counter[str] = Counter()
    agreement = 0
    planner_rank_total = 0.0
    planner_rank_count = 0
    score_gap_total = 0.0
    score_gap_count = 0

    for entry in trace:
        planner_action = str(entry.get("planner_action"))
        advisory_action = str(entry.get("advisory_best_action"))
        planner_actions[planner_action] += 1
        if advisory_action and advisory_action != "None":
            advisory_actions[advisory_action] += 1
        if bool(entry.get("advisory_agrees_with_planner")):
            agreement += 1
        planner_rank = entry.get("planner_rank_by_local_predictor")
        if planner_rank is not None:
            planner_rank_total += float(planner_rank)
            planner_rank_count += 1
        score_gap = entry.get("score_gap_to_advisory_best")
        if score_gap is not None:
            score_gap_total += float(score_gap)
            score_gap_count += 1

    total = len(trace)
    return {
        "n_steps": total,
        "disagreement_rate": round((total - agreement) / max(total, 1), 3),
        "agreement_rate": round(agreement / max(total, 1), 3),
        "mean_planner_rank_by_local_predictor": round(
            planner_rank_total / max(planner_rank_count, 1),
            3,
        ),
        "mean_score_gap_to_advisory_best": round(
            score_gap_total / max(score_gap_count, 1),
            4,
        ),
        "planner_action_distribution": dict(sorted(planner_actions.items())),
        "advisory_best_action_distribution": dict(sorted(advisory_actions.items())),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("local_only_canary", "planner_advisory", "mixed_control_rescue"),
        default="local_only_canary",
    )
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
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--allow-offline-gate-failure", action="store_true")
    parser.add_argument("--actor-share", type=float, default=0.4)
    parser.add_argument("--enable-planner-rescue", action="store_true")
    parser.add_argument("--smoke-lite", action="store_true")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-explanations-per-episode", type=int, default=8)
    return parser


def _enforce_offline_gate(
    *,
    mode: str,
    offline_gate: dict[str, Any] | None,
    allow_override: bool,
) -> dict[str, Any]:
    gate = dict(offline_gate or {})
    passed = bool(gate.get("passed", False))
    status = str(gate.get("status", "missing"))
    if not passed and not allow_override:
        raise SystemExit(
            f"offline gate blocked {mode}: status={status}. "
            "Pass --allow-offline-gate-failure to override for diagnostic use."
        )
    gate["override_used"] = bool(not passed and allow_override)
    return gate


def _run_local_only_canary(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    from snks.agent.crafter_pixel_env import ACTION_TO_IDX, CrafterPixelEnv
    from snks.agent.perception import perceive_semantic_field, perceive_tile_field
    from snks.agent.post_mortem import DamageEvent, PostMortemAnalyzer, dominant_cause
    from snks.agent.stage90r_local_model import (
        load_local_evaluator_artifact,
        rank_local_action_candidates,
    )
    from snks.agent.stage90r_local_policy import (
        BeliefStateEncoder,
        build_local_observation_package,
    )
    from snks.agent.vector_mpc_agent import DynamicEntityTracker

    device = _device()
    evaluator, evaluator_artifact = load_local_evaluator_artifact(args.local_evaluator, device=device)
    offline_gate = _enforce_offline_gate(
        mode="local_only_canary",
        offline_gate=evaluator_artifact.get("offline_gate"),
        allow_override=bool(args.allow_offline_gate_failure),
    )
    baseline_reference = load_stage89_baseline_reference()
    runtime_profile = _runtime_profile(smoke_lite=bool(args.smoke_lite))
    model, segmenter, tb, _tracker_unused, runtime = _build_runtime(
        seed=args.seed,
        checkpoint=args.checkpoint if args.perception_mode == "pixel" else None,
        crop_world=args.crop_world,
        model_dim=int(runtime_profile["model_dim"]),
        n_locations=int(runtime_profile["n_locations"]),
    )
    del model, tb, runtime  # local-only eval does not use planner/world-model

    results: list[dict[str, Any]] = []
    death_causes: Counter[str] = Counter()
    overall_action_counts: Counter[str] = Counter()
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
        steps_taken = 0
        explanation_log: list[dict[str, Any]] = []
        belief_tracker = BeliefStateEncoder()

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

            obs = build_local_observation_package(
                vf,
                body,
                inv,
                belief_context=belief_tracker.build_context(
                    near_concept=str(vf.near_concept)
                ),
            )
            ranked_candidates = rank_local_action_candidates(
                evaluator=evaluator,
                observation=obs,
                allowed_actions=_allowed_primitives(),
                action_to_idx=ACTION_TO_IDX,
                device=device,
            )
            primitive = str(ranked_candidates[0]["action"]) if ranked_candidates else "move_right"
            action_counts[primitive] += 1
            overall_action_counts[primitive] += 1
            if len(explanation_log) < args.max_explanations_per_episode:
                explanation_log.append(
                    {
                        "step": int(step),
                        "body": {key: round(float(value), 3) for key, value in body.items()},
                        "near_concept": str(vf.near_concept),
                        "selected_action": primitive,
                        "belief_state_signature": dict(obs.get("belief_state_signature", {})),
                        "top_candidates": ranked_candidates[: max(1, args.top_k)],
                    }
                )
            if step % 20 == 0:
                summary = ", ".join(
                    f"{candidate['action']}:{candidate['score']:.2f}"
                    for candidate in ranked_candidates[: max(1, args.top_k)]
                )
                near_concept = str(vf.near_concept)
                print(
                    f"s{step:3d} H{body.get('health', 0):.0f} "
                    f"F{body.get('food', 0):.0f} D{body.get('drink', 0):.0f} "
                    f"near={near_concept:9s} → {primitive:12s} "
                    f"[{summary}]"
                )

            prev_body = dict(body)
            pixels, _reward, done, info = env.step(primitive)
            raw_inv_after = dict(info.get("inventory", {}))
            body_after = {
                key: float(raw_inv_after.get(key, 0.0))
                for key in ("health", "food", "drink", "energy")
            }
            inv_after = {
                key: value
                for key, value in raw_inv_after.items()
                if key not in ("health", "food", "drink", "energy")
            }
            belief_tracker.observe_transition(
                near_concept=str(vf.near_concept),
                player_pos_before=player_pos,
                player_pos_after=tuple(info.get("player_pos", player_pos)),
                body_before=body,
                body_after=body_after,
                inventory_before=inv,
                inventory_after=inv_after,
            )
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

        death_cause = dominant_cause(PostMortemAnalyzer().attribute(damage_log, steps_taken))
        death_causes[death_cause] += 1
        episode_summary = _action_summary(action_counts)
        results.append(
            {
                "episode_id": ep,
                "seed": args.seed + ep,
                "episode_steps": steps_taken,
                "death_cause": death_cause,
                **episode_summary,
                "explanation_log": explanation_log,
            }
        )
        elapsed = time.time() - t0
        print(f"ep{ep:3d}: len={steps_taken:4d} death={death_cause:12s} [{elapsed:.0f}s]")

    avg_survival = round(
        sum(result["episode_steps"] for result in results) / max(len(results), 1),
        2,
    )
    overall_summary = _action_summary(overall_action_counts)
    payload = {
        "stage": "stage90r_local_only_eval",
        "mode": "local_only_canary",
        "architecture_role": "diagnostic_canary_not_direct_policy_target",
        "advisory_semantics": (
            "Scores are logged as predicted local consequences for falsification "
            "and inspectability, not as evidence that local argmax is the target architecture."
        ),
        "baseline_reference": baseline_reference,
        "config": {
            "n_episodes": args.n_episodes,
            "max_steps": args.max_steps,
            "seed": args.seed,
            "perception_mode": args.perception_mode,
            "local_evaluator": str(args.local_evaluator),
            "allow_offline_gate_failure": bool(args.allow_offline_gate_failure),
            "top_k": args.top_k,
        },
        "summary": {
            "avg_survival": avg_survival,
            "death_cause_breakdown": dict(death_causes),
            **overall_summary,
        },
        "offline_gate": offline_gate,
        "episodes": results,
    }
    return payload, LOCAL_ONLY_OUT_PATH


def _run_planner_advisory_analysis(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    from snks.agent.crafter_pixel_env import CrafterPixelEnv
    from snks.agent.perception import HomeostaticTracker
    from snks.agent.post_mortem import PostMortemAnalyzer
    from snks.agent.stage90r_local_model import load_local_evaluator_artifact
    from snks.agent.vector_mpc_agent import run_vector_mpc_episode

    device = _device()
    evaluator, evaluator_artifact = load_local_evaluator_artifact(args.local_evaluator, device=device)
    offline_gate = _enforce_offline_gate(
        mode="planner_advisory",
        offline_gate=evaluator_artifact.get("offline_gate"),
        allow_override=bool(args.allow_offline_gate_failure),
    )
    baseline_reference = load_stage89_baseline_reference()
    model, segmenter, tb, _tracker_unused, runtime = _build_runtime(
        seed=args.seed,
        checkpoint=args.checkpoint if args.perception_mode == "pixel" else None,
        crop_world=args.crop_world,
    )
    config = runtime["config"]
    learner = runtime["learner"]
    stimuli = runtime["stimuli"]
    analyzer = PostMortemAnalyzer()

    results: list[dict[str, Any]] = []
    death_causes: Counter[str] = Counter()
    overall_action_counts: Counter[str] = Counter()
    overall_trace: list[dict[str, Any]] = []
    t0 = time.time()

    for ep in range(args.n_episodes):
        env = CrafterPixelEnv(seed=args.seed + ep)
        tracker = HomeostaticTracker()
        tracker.init_from_textbook(tb.body_block)
        metrics = run_vector_mpc_episode(
            env=env,
            segmenter=segmenter,
            model=model,
            tracker=tracker,
            max_steps=args.max_steps,
            horizon=int(runtime_profile["planner_horizon"]),
            beam_width=int(runtime_profile["beam_width"]),
            max_depth=int(runtime_profile["max_depth"]),
            stimuli=stimuli,
            textbook=tb,
            verbose=True,
            enable_dynamic_threat_model=config["enable_dynamic_threat_model"],
            enable_dynamic_threat_goals=config["enable_dynamic_threat_goals"],
            enable_motion_plans=config["enable_motion_plans"],
            enable_motion_chains=config["enable_motion_chains"],
            enable_post_plan_passive_rollout=bool(runtime_profile["enable_post_plan_passive_rollout"]),
            perception_mode=args.perception_mode,
            local_action_advisor=evaluator,
            local_advisory_allowed_actions=_allowed_primitives(),
            record_local_advisory_trace=True,
            local_advisory_top_k=args.top_k,
            local_advisory_device=device,
        )

        attribution = analyzer.attribute(metrics.get("damage_log", []), metrics.get("episode_steps", 0))
        learner.update(attribution)
        stimuli = learner.build_stimuli(
            ["health", "food", "drink", "energy"],
            include_vital_delta=config["include_vital_delta"],
        )

        death_cause = metrics.get("death_cause", "alive")
        death_causes[death_cause] += 1
        action_counts = Counter(metrics.get("action_counts", {}))
        overall_action_counts.update(action_counts)
        advisory_trace = list(metrics.get("local_advisory_trace", []))
        overall_trace.extend(advisory_trace)
        advisory_summary = _summarize_planner_advisory_trace(advisory_trace)
        results.append(
            {
                "episode_id": ep,
                "seed": args.seed + ep,
                "episode_steps": int(metrics.get("episode_steps", 0)),
                "death_cause": death_cause,
                **_action_summary(action_counts),
                "planner_advisory_summary": advisory_summary,
                "planner_advisory_trace": advisory_trace[: args.max_explanations_per_episode],
            }
        )
        elapsed = time.time() - t0
        print(f"ep{ep:3d}: len={metrics.get('episode_steps', 0):4.0f} death={death_cause:12s} [{elapsed:.0f}s]")

    avg_survival = round(
        sum(result["episode_steps"] for result in results) / max(len(results), 1),
        2,
    )
    payload = {
        "stage": "stage90r_planner_advisory_eval",
        "mode": "planner_advisory_analysis",
        "architecture_role": "planner_canonical_local_predictor_advisory_only",
        "advisory_semantics": (
            "The planner remains the canonical decision layer. The local evaluator "
            "only scores short-horizon consequences for comparison and inspection."
        ),
        "baseline_reference": baseline_reference,
        "config": {
            "n_episodes": args.n_episodes,
            "max_steps": args.max_steps,
            "seed": args.seed,
            "perception_mode": args.perception_mode,
            "local_evaluator": str(args.local_evaluator),
            "allow_offline_gate_failure": bool(args.allow_offline_gate_failure),
            "smoke_lite": bool(args.smoke_lite),
            "top_k": args.top_k,
        },
        "summary": {
            "avg_survival": avg_survival,
            "death_cause_breakdown": dict(death_causes),
            **_action_summary(overall_action_counts),
            "planner_advisory": _summarize_planner_advisory_trace(overall_trace),
        },
        "offline_gate": offline_gate,
        "episodes": results,
    }
    return payload, PLANNER_ADVISORY_OUT_PATH


def _run_mixed_control_rescue_eval(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    from snks.agent.crafter_pixel_env import CrafterPixelEnv
    from snks.agent.perception import HomeostaticTracker
    from snks.agent.post_mortem import PostMortemAnalyzer
    from snks.agent.stage90r_local_model import load_local_evaluator_artifact
    from snks.agent.vector_mpc_agent import run_vector_mpc_episode

    device = _device()
    evaluator, evaluator_artifact = load_local_evaluator_artifact(args.local_evaluator, device=device)
    offline_gate = _enforce_offline_gate(
        mode="mixed_control_rescue",
        offline_gate=evaluator_artifact.get("offline_gate"),
        allow_override=bool(args.allow_offline_gate_failure),
    )
    baseline_reference = _load_stage90r_rescue_baseline_reference()
    runtime_profile = _runtime_profile(smoke_lite=bool(args.smoke_lite))
    model, segmenter, tb, _tracker_unused, runtime = _build_runtime(
        seed=args.seed,
        checkpoint=args.checkpoint if args.perception_mode == "pixel" else None,
        crop_world=args.crop_world,
        model_dim=int(runtime_profile["model_dim"]),
        n_locations=int(runtime_profile["n_locations"]),
    )
    config = runtime["config"]
    learner = runtime["learner"]
    stimuli = runtime["stimuli"]
    analyzer = PostMortemAnalyzer()

    controller_counts: Counter[str] = Counter()
    rescue_trigger_counts: Counter[str] = Counter()
    rescue_override_source_counts: Counter[str] = Counter()
    rescue_activation_reason_counts: Counter[str] = Counter()
    death_causes: Counter[str] = Counter()
    early_hostile_deaths_without_rescue = 0
    hostile_deaths_without_rescue = 0
    results: list[dict[str, Any]] = []
    t0 = time.time()

    for ep in range(args.n_episodes):
        env = CrafterPixelEnv(seed=args.seed + ep)
        tracker = HomeostaticTracker()
        tracker.init_from_textbook(tb.body_block)
        metrics = run_vector_mpc_episode(
            env=env,
            segmenter=segmenter,
            model=model,
            tracker=tracker,
            max_steps=args.max_steps,
            horizon=int(runtime_profile["planner_horizon"]),
            beam_width=int(runtime_profile["beam_width"]),
            max_depth=int(runtime_profile["max_depth"]),
            stimuli=stimuli,
            textbook=tb,
            verbose=False,
            enable_dynamic_threat_model=config["enable_dynamic_threat_model"],
            enable_dynamic_threat_goals=config["enable_dynamic_threat_goals"],
            enable_motion_plans=config["enable_motion_plans"],
            enable_motion_chains=config["enable_motion_chains"],
            enable_post_plan_passive_rollout=bool(runtime_profile["enable_post_plan_passive_rollout"]),
            perception_mode=args.perception_mode,
            local_actor_policy=evaluator,
            local_advisory_device=device,
            mixed_control_actor_share=float(args.actor_share),
            enable_planner_rescue=bool(args.enable_planner_rescue),
            record_local_trace=True,
            record_local_counterfactuals="salient_only",
            local_counterfactual_horizon=1,
            rng=_eval_episode_rng(base_seed=args.seed, episode_index=ep),
        )
        attribution = analyzer.attribute(metrics.get("damage_log", []), metrics.get("episode_steps", 0))
        learner.update(attribution)
        stimuli = learner.build_stimuli(
            ["health", "food", "drink", "energy"],
            include_vital_delta=config["include_vital_delta"],
        )
        local_trace = list(metrics.get("local_trace", []))
        rescue_trace = list(metrics.get("rescue_trace", []))
        local_controller_counts = Counter(str(step.get("controller", "unknown")) for step in local_trace)
        controller_counts.update(local_controller_counts)
        rescue_trigger_counts.update(str(event.get("trigger", "unknown")) for event in rescue_trace)
        rescue_override_source_counts.update(
            str(event.get("override_source", event.get("rescue_policy", "unknown")))
            for event in rescue_trace
        )
        rescue_activation_reason_counts.update(
            str(event.get("activation_reason", event.get("trigger", "unknown")))
            for event in rescue_trace
        )
        death_cause = metrics.get("death_cause", "alive")
        death_causes[death_cause] += 1
        hostile_death_without_rescue = death_cause in {"zombie", "skeleton", "arrow"} and not rescue_trace
        if hostile_death_without_rescue:
            hostile_deaths_without_rescue += 1
            if int(metrics.get("episode_steps", 0)) <= 50:
                early_hostile_deaths_without_rescue += 1
        results.append(
            {
                "episode_id": ep,
                "seed": args.seed + ep,
                "episode_steps": int(metrics.get("episode_steps", 0)),
                "death_cause": death_cause,
                "controller_distribution": dict(sorted(local_controller_counts.items())),
                "n_rescue_events": len(rescue_trace),
                "hostile_death_without_rescue": bool(hostile_death_without_rescue),
                "rescue_trace": rescue_trace[: args.max_explanations_per_episode],
                "local_trace_excerpt": local_trace[: args.max_explanations_per_episode],
            }
        )
        elapsed = time.time() - t0
        print(f"ep{ep:3d}: len={metrics.get('episode_steps', 0):4.0f} death={death_cause:12s} rescues={len(rescue_trace):2d} [{elapsed:.0f}s]")

    total_steps = sum(sum(ep["controller_distribution"].values()) for ep in results)
    planner_aligned_emergency_steps = rescue_override_source_counts.get("planner_aligned_safety", 0)
    baseline_avg_survival = None
    if baseline_reference.get("summary"):
        baseline_avg_survival = baseline_reference["summary"].get("avg_survival")
    avg_survival = round(sum(ep["episode_steps"] for ep in results) / max(len(results), 1), 2)
    fallback_status = "UNKNOWN"
    if baseline_avg_survival is not None:
        fallback_status = "PASS" if avg_survival >= float(baseline_avg_survival) else "PARTIAL"
    payload = {
        "stage": "stage90r_mixed_control_rescue_eval",
        "mode": "mixed_control_rescue",
        "architecture_role": "actor_bootstrap_with_first_class_emergency_safety_controller",
        "baseline_reference": baseline_reference,
        "config": {
            "n_episodes": args.n_episodes,
            "max_steps": args.max_steps,
            "seed": args.seed,
            "perception_mode": args.perception_mode,
            "local_evaluator": str(args.local_evaluator),
            "actor_share": float(args.actor_share),
            "enable_planner_rescue": bool(args.enable_planner_rescue),
            "allow_offline_gate_failure": bool(args.allow_offline_gate_failure),
            "smoke_lite": bool(args.smoke_lite),
        },
        "summary": {
            "avg_survival": avg_survival,
            "death_cause_breakdown": dict(death_causes),
            "controller_distribution": dict(sorted(controller_counts.items())),
            "planner_dependence": round(
                (
                    controller_counts.get("planner_bootstrap", 0)
                    + controller_counts.get("planner_rescue", 0)
                    + planner_aligned_emergency_steps
                ) / max(total_steps, 1),
                3,
            ),
            "learner_control_fraction": round(controller_counts.get("learner_actor", 0) / max(total_steps, 1), 3),
            "rescue_rate": round(sum(rescue_trigger_counts.values()) / max(total_steps, 1), 3),
            "rescue_trigger_distribution": dict(sorted(rescue_trigger_counts.items())),
            "rescue_activation_reason_distribution": dict(sorted(rescue_activation_reason_counts.items())),
            "rescue_override_source_distribution": dict(sorted(rescue_override_source_counts.items())),
            "early_hostile_deaths_without_rescue": int(early_hostile_deaths_without_rescue),
            "hostile_deaths_without_rescue": int(hostile_deaths_without_rescue),
            "fallback_criterion": {
                "baseline_commit": "9083357",
                "baseline_avg_survival": baseline_avg_survival,
                "candidate_avg_survival": avg_survival,
                "status": fallback_status,
            },
        },
        "offline_gate": offline_gate,
        "episodes": results,
    }
    return payload, DOCS_DIR / "stage90r_mixed_control_rescue_eval.json"


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.mode == "planner_advisory":
        payload, out_path = _run_planner_advisory_analysis(args)
    elif args.mode == "mixed_control_rescue":
        payload, out_path = _run_mixed_control_rescue_eval(args)
    else:
        payload, out_path = _run_local_only_canary(args)
    if args.out is not None:
        out_path = Path(args.out)
    out_path.write_text(json.dumps(payload, indent=2, default=_json_default))
    print(f"saved eval: {out_path}")


if __name__ == "__main__":
    main()
