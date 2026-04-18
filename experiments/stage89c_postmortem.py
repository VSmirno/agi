"""Stage 89c diagnostic: compare fair baseline vs current on matching seeds.

Run on minipc:
  ./scripts/minipc-run.sh stage89c "from stage89c_postmortem import main; main()"

Writes:
  _docs/stage89c_episode_buckets.json
  _docs/stage89c_regression_cases.json
  _docs/stage89c_counterfactuals.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

torch.backends.cudnn.enabled = False

ROOT = Path(__file__).parent.parent
DOCS_DIR = ROOT / "_docs"
BUCKETS_PATH = DOCS_DIR / "stage89c_episode_buckets.json"
REGRESSIONS_PATH = DOCS_DIR / "stage89c_regression_cases.json"
COUNTERFACTUALS_PATH = DOCS_DIR / "stage89c_counterfactuals.json"


def _bucket_for_delta(delta_steps: int, threshold: int = 20) -> str:
    if delta_steps >= threshold:
        return "improved"
    if delta_steps <= -threshold:
        return "regressed"
    return "neutral"


def _mode_config(mode: str) -> dict:
    from stage89_eval import _stage89_mode_config

    return _stage89_mode_config(mode)


def _run_one_episode(
    *,
    mode: str,
    seed: int,
    max_steps: int,
    device,
    segmenter,
    textbook,
) -> dict:
    from snks.agent.crafter_pixel_env import CrafterPixelEnv
    from snks.agent.perception import HomeostaticTracker
    from snks.agent.post_mortem import PostMortemLearner
    from snks.agent.vector_bootstrap import load_from_textbook
    from snks.agent.vector_mpc_agent import run_vector_mpc_episode
    from snks.agent.vector_world_model import VectorWorldModel

    config = _mode_config(mode)
    model = VectorWorldModel(dim=16384, n_locations=50000, seed=seed, device=device)
    textbook_path = ROOT / "configs" / "crafter_textbook.yaml"
    load_from_textbook(model, textbook_path)
    stimuli = PostMortemLearner().build_stimuli(
        ["health", "food", "drink", "energy"],
        include_vital_delta=config["include_vital_delta"],
    )

    env = CrafterPixelEnv(seed=seed)
    tracker = HomeostaticTracker()
    tracker.init_from_textbook(textbook.body_block)

    metrics = run_vector_mpc_episode(
        env=env,
        segmenter=segmenter,
        model=model,
        tracker=tracker,
        max_steps=max_steps,
        stimuli=stimuli,
        textbook=textbook,
        verbose=False,
        enable_dynamic_threat_model=config["enable_dynamic_threat_model"],
        enable_dynamic_threat_goals=config["enable_dynamic_threat_goals"],
        enable_motion_plans=config["enable_motion_plans"],
        enable_motion_chains=config["enable_motion_chains"],
        enable_post_plan_passive_rollout=config["enable_post_plan_passive_rollout"],
        record_stage89c_trace=True,
    )
    metrics["mode"] = mode
    metrics["seed"] = seed
    metrics["mode_config"] = config
    return metrics


def _counterfactual_from_pair(seed: int, baseline: dict, current: dict) -> dict:
    base_event = baseline.get("defensive_events", [])
    curr_event = current.get("defensive_events", [])
    return {
        "seed": seed,
        "baseline_steps": baseline.get("episode_steps", 0),
        "current_steps": current.get("episode_steps", 0),
        "delta_steps": current.get("episode_steps", 0) - baseline.get("episode_steps", 0),
        "baseline_death_cause": baseline.get("death_cause", "alive"),
        "current_death_cause": current.get("death_cause", "alive"),
        "baseline_first_defensive_action_step": baseline.get("first_defensive_action_step"),
        "current_first_defensive_action_step": current.get("first_defensive_action_step"),
        "baseline_first_event": base_event[0] if base_event else None,
        "current_first_event": curr_event[0] if curr_event else None,
    }


def main() -> None:
    from snks.agent.crafter_textbook import CrafterTextbook
    from snks.encoder.tile_segmenter import load_tile_segmenter, pick_device

    DOCS_DIR.mkdir(exist_ok=True)
    device = torch.device(pick_device())
    checkpoint_path = ROOT / "demos" / "checkpoints" / "exp136" / "segmenter_9x9.pt"
    textbook_path = ROOT / "configs" / "crafter_textbook.yaml"
    segmenter = load_tile_segmenter(str(checkpoint_path), device=device)
    textbook = CrafterTextbook(str(textbook_path))

    n_episodes = 20
    max_steps = 1000
    start_seed = 42

    pairs: list[dict] = []
    regressions: list[dict] = []
    counterfactuals: list[dict] = []

    for ep in range(n_episodes):
        seed = start_seed + ep
        baseline = _run_one_episode(
            mode="baseline",
            seed=seed,
            max_steps=max_steps,
            device=device,
            segmenter=segmenter,
            textbook=textbook,
        )
        current = _run_one_episode(
            mode="current",
            seed=seed,
            max_steps=max_steps,
            device=device,
            segmenter=segmenter,
            textbook=textbook,
        )
        delta_steps = int(current.get("episode_steps", 0) - baseline.get("episode_steps", 0))
        bucket = _bucket_for_delta(delta_steps)
        pair = {
            "seed": seed,
            "episode": ep,
            "episode_bucket": bucket,
            "baseline_steps": baseline.get("episode_steps", 0),
            "current_steps": current.get("episode_steps", 0),
            "delta_steps": delta_steps,
            "baseline_death_cause": baseline.get("death_cause", "alive"),
            "current_death_cause": current.get("death_cause", "alive"),
            "baseline_first_arrow_threat_step": baseline.get("first_arrow_threat_step"),
            "current_first_arrow_threat_step": current.get("first_arrow_threat_step"),
            "baseline_first_defensive_action_step": baseline.get("first_defensive_action_step"),
            "current_first_defensive_action_step": current.get("first_defensive_action_step"),
            "baseline_defensive_action_rate": baseline.get("defensive_action_rate", 0.0),
            "current_defensive_action_rate": current.get("defensive_action_rate", 0.0),
            "baseline_danger_prediction_error": baseline.get("danger_prediction_error", 0.0),
            "current_danger_prediction_error": current.get("danger_prediction_error", 0.0),
        }
        pairs.append(pair)

        if bucket == "regressed":
            regression_case = {
                "seed": seed,
                "baseline_steps": baseline.get("episode_steps", 0),
                "current_steps": current.get("episode_steps", 0),
                "baseline_death_cause": baseline.get("death_cause", "alive"),
                "current_death_cause": current.get("death_cause", "alive"),
                "first_arrow_threat_step": current.get("first_arrow_threat_step"),
                "first_defensive_action_step": current.get("first_defensive_action_step"),
                "predicted_best_loss": (
                    current.get("defensive_events", [{}])[0].get("predicted_best_loss")
                    if current.get("defensive_events")
                    else None
                ),
                "predicted_baseline_loss": (
                    current.get("defensive_events", [{}])[0].get("predicted_baseline_loss")
                    if current.get("defensive_events")
                    else None
                ),
                "post_defense_vitals": (
                    current.get("defensive_events", [{}])[0].get("post_defense_vitals")
                    if current.get("defensive_events")
                    else None
                ),
                "post_defense_resource_access": (
                    current.get("defensive_events", [{}])[0].get("resource_access_loss")
                    if current.get("defensive_events")
                    else None
                ),
                "current_defensive_events": current.get("defensive_events", []),
                "baseline_defensive_events": baseline.get("defensive_events", []),
                "notes": "",
            }
            regressions.append(regression_case)
            counterfactuals.append(_counterfactual_from_pair(seed, baseline, current))

        print(
            f"ep{ep:2d} seed={seed} "
            f"baseline={baseline.get('episode_steps', 0):4.0f}/{baseline.get('death_cause', '?'):10s} "
            f"current={current.get('episode_steps', 0):4.0f}/{current.get('death_cause', '?'):10s} "
            f"bucket={bucket:9s} "
            f"def={current.get('defensive_action_rate', 0.0):.2f}"
        )

    summary = {
        "n_episodes": n_episodes,
        "baseline_avg_survival": round(float(np.mean([p["baseline_steps"] for p in pairs])), 2),
        "current_avg_survival": round(float(np.mean([p["current_steps"] for p in pairs])), 2),
        "avg_delta_steps": round(float(np.mean([p["delta_steps"] for p in pairs])), 2),
        "bucket_counts": {
            "improved": sum(1 for p in pairs if p["episode_bucket"] == "improved"),
            "neutral": sum(1 for p in pairs if p["episode_bucket"] == "neutral"),
            "regressed": sum(1 for p in pairs if p["episode_bucket"] == "regressed"),
        },
    }

    BUCKETS_PATH.write_text(json.dumps({"summary": summary, "episodes": pairs}, indent=2))
    REGRESSIONS_PATH.write_text(json.dumps({"summary": summary, "cases": regressions}, indent=2))
    COUNTERFACTUALS_PATH.write_text(
        json.dumps({"summary": summary, "pairs": counterfactuals}, indent=2)
    )

    print(summary)
    print(f"Saved {BUCKETS_PATH}")
    print(f"Saved {REGRESSIONS_PATH}")
    print(f"Saved {COUNTERFACTUALS_PATH}")


if __name__ == "__main__":
    main()
