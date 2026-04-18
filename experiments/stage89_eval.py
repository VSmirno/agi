"""Stage 89: Arrow Trajectory Modeling — targeted eval on Crafter.

Run on minipc ONLY:
  ./scripts/minipc-run.sh stage89 "from stage89_eval import main; main()"
  ./scripts/minipc-run.sh stage89_baseline \
    "from stage89_eval import main; import sys; sys.argv=['stage89_eval.py','--mode','baseline']; main()"

Outputs:
  - _docs/stage89_baseline.json
  - _docs/stage89_eval.json
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

torch.backends.cudnn.enabled = False

ROOT = Path(__file__).parent.parent
DOCS_DIR = ROOT / "_docs"
BASELINE_PATH = DOCS_DIR / "stage89_baseline.json"
EVAL_PATH = DOCS_DIR / "stage89_eval.json"


def _stage89_mode_config(mode: str) -> dict:
    if mode == "current":
        return {
            "include_vital_delta": True,
            "enable_dynamic_threat_model": True,
            "enable_dynamic_threat_goals": True,
            "enable_motion_plans": True,
            "enable_motion_chains": True,
            "enable_post_plan_passive_rollout": True,
        }
    if mode == "baseline":
        return {
            "include_vital_delta": False,
            "enable_dynamic_threat_model": False,
            "enable_dynamic_threat_goals": False,
            "enable_motion_plans": False,
            "enable_motion_chains": False,
            "enable_post_plan_passive_rollout": False,
        }
    raise ValueError(f"Unknown stage89 mode: {mode}")


def _build_model_and_segmenter(model_dim: int, n_locations: int, seed: int, device):
    from snks.agent.vector_bootstrap import load_from_textbook
    from snks.agent.vector_world_model import VectorWorldModel
    from snks.encoder.tile_segmenter import load_tile_segmenter

    checkpoint_path = ROOT / "demos" / "checkpoints" / "exp136" / "segmenter_9x9.pt"
    textbook_path = ROOT / "configs" / "crafter_textbook.yaml"

    segmenter = load_tile_segmenter(str(checkpoint_path), device=device)
    model = VectorWorldModel(dim=model_dim, n_locations=n_locations, seed=seed, device=device)
    stats = load_from_textbook(model, textbook_path)
    print(f"Segmenter: {checkpoint_path.name}  Textbook seeded: {stats}")
    return model, segmenter, textbook_path


def _summarize_episode_metrics(results: list[dict]) -> dict:
    avg_survival = float(np.mean([r.get("episode_steps", 0) for r in results])) if results else 0.0
    death_causes = Counter(r.get("death_cause", "alive") for r in results)
    n_results = max(len(results), 1)
    arrow_death_pct = 100.0 * death_causes.get("arrow", 0) / n_results
    defensive_action_rate = float(
        np.mean([r.get("defensive_action_rate", 0.0) for r in results])
    ) if results else 0.0
    danger_prediction_error = float(
        np.mean([r.get("danger_prediction_error", 0.0) for r in results])
    ) if results else 0.0
    arrow_threat_steps = int(sum(r.get("arrow_threat_steps", 0) for r in results))
    defensive_action_steps = int(sum(r.get("defensive_action_steps", 0) for r in results))
    arrow_visible_steps = int(sum(r.get("arrow_visible_steps", 0) for r in results))
    arrow_velocity_known_steps = int(
        sum(r.get("arrow_velocity_known_steps", 0) for r in results)
    )
    arrow_velocity_unknown_steps = int(
        sum(r.get("arrow_velocity_unknown_steps", 0) for r in results)
    )

    return {
        "avg_survival": round(avg_survival, 2),
        "arrow_death_pct": round(arrow_death_pct, 2),
        "danger_prediction_error": round(danger_prediction_error, 3),
        "defensive_action_rate": round(defensive_action_rate, 3),
        "arrow_threat_steps": arrow_threat_steps,
        "defensive_action_steps": defensive_action_steps,
        "arrow_visible_steps": arrow_visible_steps,
        "arrow_velocity_known_steps": arrow_velocity_known_steps,
        "arrow_velocity_unknown_steps": arrow_velocity_unknown_steps,
        "arrow_velocity_known_rate": round(
            arrow_velocity_known_steps / max(arrow_visible_steps, 1), 3
        ),
        "death_causes": dict(death_causes),
    }


def run_eval(
    n_episodes: int,
    max_steps: int,
    model_dim: int,
    n_locations: int,
    seed: int,
    device,
    mode: str = "current",
) -> dict:
    from snks.agent.crafter_pixel_env import CrafterPixelEnv
    from snks.agent.crafter_textbook import CrafterTextbook
    from snks.agent.perception import HomeostaticTracker
    from snks.agent.post_mortem import PostMortemAnalyzer, PostMortemLearner
    from snks.agent.vector_mpc_agent import run_vector_mpc_episode

    config = _stage89_mode_config(mode)
    model, segmenter, textbook_path = _build_model_and_segmenter(
        model_dim, n_locations, seed, device
    )
    tb = CrafterTextbook(str(textbook_path))
    vitals = ["health", "food", "drink", "energy"]
    learner = PostMortemLearner()
    analyzer = PostMortemAnalyzer()
    stimuli = learner.build_stimuli(
        vitals,
        include_vital_delta=config["include_vital_delta"],
    )

    results: list[dict] = []
    t0 = time.time()
    for ep in range(n_episodes):
        ep_seed = seed + ep
        env = CrafterPixelEnv(seed=ep_seed)
        tracker = HomeostaticTracker()
        tracker.init_from_textbook(tb.body_block)

        metrics = run_vector_mpc_episode(
            env=env,
            segmenter=segmenter,
            model=model,
            tracker=tracker,
            max_steps=max_steps,
            stimuli=stimuli,
            textbook=tb,
            verbose=True,
            enable_dynamic_threat_model=config["enable_dynamic_threat_model"],
            enable_dynamic_threat_goals=config["enable_dynamic_threat_goals"],
            enable_motion_plans=config["enable_motion_plans"],
            enable_motion_chains=config["enable_motion_chains"],
            enable_post_plan_passive_rollout=config["enable_post_plan_passive_rollout"],
        )
        results.append(metrics)

        damage_log = metrics.get("damage_log", [])
        attribution = analyzer.attribute(damage_log, metrics.get("episode_steps", 0))
        learner.update(attribution)
        stimuli = learner.build_stimuli(
            vitals,
            include_vital_delta=config["include_vital_delta"],
        )

        elapsed = time.time() - t0
        eta = elapsed / (ep + 1) * (n_episodes - ep - 1)
        print(
            f"ep{ep:2d}: len={metrics.get('episode_steps', 0):4.0f} "
            f"death={metrics.get('death_cause', '?'):10s} "
            f"arrow_threat={metrics.get('arrow_threat_steps', 0):3d} "
            f"def_rate={metrics.get('defensive_action_rate', 0.0):.2f} "
            f"pred_err={metrics.get('danger_prediction_error', 0.0):.3f} "
            f"[{elapsed:.0f}s eta={eta:.0f}s]"
        )

    summary = _summarize_episode_metrics(results)
    return {
        "mode": mode,
        "mode_config": config,
        "summary": summary,
        "episodes": [{k: v for k, v in r.items() if k != "damage_log"} for r in results],
    }


def _compare_to_baseline(current: dict, baseline: dict | None) -> dict:
    if baseline is None:
        return {"baseline_available": False}

    current_summary = current["summary"]
    baseline_summary = baseline.get("summary", baseline)
    arrow_reduction = baseline_summary["arrow_death_pct"] - current_summary["arrow_death_pct"]
    relative_reduction = (
        arrow_reduction / baseline_summary["arrow_death_pct"]
        if baseline_summary["arrow_death_pct"] > 0
        else 0.0
    )
    survival_delta = current_summary["avg_survival"] - baseline_summary["avg_survival"]

    return {
        "baseline_available": True,
        "baseline": baseline,
        "survival_delta": round(survival_delta, 2),
        "arrow_death_pct_delta": round(-arrow_reduction, 2),
        "arrow_death_reduction_ratio": round(relative_reduction, 3),
        "gate_arrow_deaths_halved": relative_reduction >= 0.5,
    }


def main() -> None:
    import argparse
    from snks.encoder.tile_segmenter import pick_device

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("current", "baseline"), default="current")
    parser.add_argument("--n-episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    baseline = json.loads(BASELINE_PATH.read_text()) if BASELINE_PATH.exists() else None
    device = torch.device(pick_device())
    model_dim = 16384
    n_locations = 50000

    print(
        f"device={device}, dim={model_dim}, locs={n_locations}, "
        f"episodes={args.n_episodes}, max_steps={args.max_steps}, mode={args.mode}"
    )
    if args.mode == "current" and baseline is None:
        print("Baseline reference missing. Eval will still run, but comparison will be omitted.")

    run_data = run_eval(
        n_episodes=args.n_episodes,
        max_steps=args.max_steps,
        model_dim=model_dim,
        n_locations=n_locations,
        seed=args.seed,
        device=device,
        mode=args.mode,
    )
    if args.mode == "baseline":
        out = {
            **run_data,
            "comparison": {"baseline_available": False},
            "n_episodes": args.n_episodes,
            "max_steps": args.max_steps,
        }
        DOCS_DIR.mkdir(exist_ok=True)
        BASELINE_PATH.write_text(json.dumps(out, indent=2, default=str))
        print(f"Saved baseline reference to {BASELINE_PATH}")
        print(json.dumps(out["summary"], indent=2))
        return

    comparison = _compare_to_baseline(run_data, baseline)
    out = {
        **run_data,
        "comparison": comparison,
        "n_episodes": args.n_episodes,
        "max_steps": args.max_steps,
    }

    DOCS_DIR.mkdir(exist_ok=True)
    EVAL_PATH.write_text(json.dumps(out, indent=2, default=str))
    print(f"Saved eval results to {EVAL_PATH}")
    print(json.dumps(out["summary"], indent=2))
    if comparison.get("baseline_available"):
        print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
