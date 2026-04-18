"""Diagnostic ablation: compare pixel vs symbolic perception on Stage 89 stack.

Run on minipc ONLY:
  ./scripts/minipc-run.sh stage89sym \
    "from stage89_symbolic_ablation import main; main()"

Outputs:
  - _docs/stage89_symbolic_ablation.json
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

torch.backends.cudnn.enabled = False

ROOT = Path(__file__).parent.parent
OUT_PATH = ROOT / "_docs" / "stage89_symbolic_ablation.json"


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def _summarize(results: list[dict]) -> dict:
    avg_survival = float(np.mean([r.get("episode_steps", 0) for r in results])) if results else 0.0
    death_causes = Counter(r.get("death_cause", "alive") for r in results)
    n_results = max(len(results), 1)
    return {
        "avg_survival": round(avg_survival, 2),
        "arrow_death_pct": round(100.0 * death_causes.get("arrow", 0) / n_results, 2),
        "danger_prediction_error": round(
            float(np.mean([r.get("danger_prediction_error", 0.0) for r in results])) if results else 0.0,
            3,
        ),
        "defensive_action_rate": round(
            float(np.mean([r.get("defensive_action_rate", 0.0) for r in results])) if results else 0.0,
            3,
        ),
        "death_causes": dict(death_causes),
    }


def _setup_mlflow(
    *,
    enabled: bool,
    tracking_uri: str,
    experiment_name: str,
    run_name: str,
    params: dict,
):
    if not enabled:
        return None

    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    run = mlflow.start_run(run_name=run_name)
    clean_params = {
        key: (int(value) if isinstance(value, np.integer) else value)
        for key, value in params.items()
    }
    mlflow.log_params(clean_params)
    return mlflow


def _log_mode_run(
    mlflow_module,
    *,
    mode: str,
    summary: dict,
    params: dict,
) -> None:
    if mlflow_module is None:
        return

    with mlflow_module.start_run(run_name=f"perception_{mode}", nested=True):
        mlflow_module.log_params({"perception_mode": mode, **params})
        for key, value in summary.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    mlflow_module.log_metric(f"{key}_{subkey}", float(subvalue))
            elif isinstance(value, (int, float)):
                mlflow_module.log_metric(key, float(value))


def _log_parent_run(mlflow_module, out: dict) -> None:
    if mlflow_module is None:
        return

    comparison = out.get("comparison", {})
    for key, value in comparison.items():
        if isinstance(value, (int, float)):
            mlflow_module.log_metric(key, float(value))
    mlflow_module.log_artifact(str(OUT_PATH))


def _build_runtime(seed: int, device: torch.device):
    from snks.agent.crafter_textbook import CrafterTextbook
    from snks.agent.post_mortem import PostMortemLearner
    from snks.agent.vector_bootstrap import load_from_textbook
    from snks.agent.vector_world_model import VectorWorldModel
    from snks.encoder.tile_segmenter import load_tile_segmenter

    checkpoint_path = ROOT / "demos" / "checkpoints" / "exp136" / "segmenter_9x9.pt"
    textbook_path = ROOT / "configs" / "crafter_textbook.yaml"

    model = VectorWorldModel(dim=16384, n_locations=50000, seed=seed, device=device)
    load_from_textbook(model, textbook_path)
    segmenter = load_tile_segmenter(str(checkpoint_path), device=device)
    textbook = CrafterTextbook(str(textbook_path))
    stimuli = PostMortemLearner().build_stimuli(["health", "food", "drink", "energy"])
    return model, segmenter, textbook, stimuli


def _run_mode(
    mode: str,
    n_episodes: int,
    max_steps: int,
    start_seed: int,
    device: torch.device,
    mlflow_module=None,
) -> dict:
    from snks.agent.crafter_pixel_env import CrafterPixelEnv
    from snks.agent.perception import HomeostaticTracker
    from snks.agent.vector_mpc_agent import run_vector_mpc_episode

    model, segmenter, textbook, stimuli = _build_runtime(start_seed, device)
    results: list[dict] = []
    for ep in range(n_episodes):
        seed = start_seed + ep
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
            perception_mode=mode,
        )
        results.append(metrics)
        print(
            f"{mode:8s} ep{ep:02d} seed={seed} "
            f"len={metrics.get('episode_steps', 0):4.0f} "
            f"death={metrics.get('death_cause', '?'):10s} "
            f"def={metrics.get('defensive_action_rate', 0.0):.2f}"
        )
    out = {
        "perception_mode": mode,
        "summary": _summarize(results),
        "episodes": [{k: v for k, v in r.items() if k != "damage_log"} for r in results],
        "oracle_symbolic": mode == "symbolic",
        "diagnostic_only": True,
        "valid_for_concept_success_claims": False,
    }
    _log_mode_run(
        mlflow_module,
        mode=mode,
        summary=out["summary"],
        params={
            "n_episodes": n_episodes,
            "max_steps": max_steps,
            "start_seed": start_seed,
        },
    )
    return out


def main() -> None:
    import argparse
    from snks.encoder.tile_segmenter import pick_device

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-episodes", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--start-seed", type=int, default=42)
    parser.add_argument("--use-mlflow", action="store_true")
    parser.add_argument("--mlflow-uri", type=str, default="http://127.0.0.1:5000")
    parser.add_argument("--mlflow-experiment", type=str, default="agi_stage89_symbolic")
    parser.add_argument("--mlflow-run-name", type=str, default="stage89_symbolic_ablation")
    args = parser.parse_args()

    device = torch.device(pick_device())
    mlflow_module = _setup_mlflow(
        enabled=args.use_mlflow,
        tracking_uri=args.mlflow_uri,
        experiment_name=args.mlflow_experiment,
        run_name=args.mlflow_run_name,
        params={
            "n_episodes": args.n_episodes,
            "max_steps": args.max_steps,
            "start_seed": args.start_seed,
            "oracle_symbolic": True,
            "diagnostic_only": True,
        },
    )
    print(
        f"stage89_symbolic_ablation: device={device} episodes={args.n_episodes} "
        f"max_steps={args.max_steps} start_seed={args.start_seed}"
    )

    pixel = _run_mode(
        "pixel", args.n_episodes, args.max_steps, args.start_seed, device, mlflow_module
    )
    symbolic = _run_mode(
        "symbolic", args.n_episodes, args.max_steps, args.start_seed, device, mlflow_module
    )
    out = {
        "pixel": pixel,
        "symbolic": symbolic,
        "comparison": {
            "survival_delta_symbolic_minus_pixel": round(
                symbolic["summary"]["avg_survival"] - pixel["summary"]["avg_survival"], 2
            ),
            "arrow_death_pct_delta_symbolic_minus_pixel": round(
                symbolic["summary"]["arrow_death_pct"] - pixel["summary"]["arrow_death_pct"], 2
            ),
            "defensive_action_rate_delta_symbolic_minus_pixel": round(
                symbolic["summary"]["defensive_action_rate"] - pixel["summary"]["defensive_action_rate"], 3
            ),
            "danger_prediction_error_delta_symbolic_minus_pixel": round(
                symbolic["summary"]["danger_prediction_error"] - pixel["summary"]["danger_prediction_error"], 3
            ),
        },
        "oracle_symbolic": True,
        "diagnostic_only": True,
        "valid_for_concept_success_claims": False,
        "n_episodes": args.n_episodes,
        "max_steps": args.max_steps,
        "start_seed": args.start_seed,
    }
    OUT_PATH.write_text(json.dumps(out, indent=2, default=_json_default))
    _log_parent_run(mlflow_module, out)
    if mlflow_module is not None:
        mlflow_module.end_run()
    print(f"Saved symbolic ablation to {OUT_PATH}")
    print(json.dumps(out["comparison"], indent=2))


if __name__ == "__main__":
    main()
