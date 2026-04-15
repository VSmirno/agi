"""Stage 86: Post-Mortem Learning — eval gates on Crafter.

Run on minipc ONLY:
  ssh minipc "cd /opt/agi && git pull origin main"
  ssh minipc "tmux new-session -d -s stage86 'cd /opt/agi && source venv/bin/activate && \
    HSA_OVERRIDE_GFX_VERSION=11.0.0 PYTHONPATH=src python -u experiments/stage86_eval.py 2>&1 | \
    tee _docs/stage86_eval.log'"

Gates:
  1. zombie_deaths_decrease: zombie_deaths(ep14-20) < zombie_deaths(ep1-7)  [within with_pm run]
  2. starvation_decrease:    starvation_deaths(with_pm) < starvation_deaths(without_pm)
  3. survival_holds:         avg_survival(with_pm) >= 155

Results saved to _docs/stage86_eval.json.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

torch.backends.cudnn.enabled = False


def _build_model_and_segmenter(model_dim: int, n_locations: int, seed: int, device):
    from snks.agent.vector_world_model import VectorWorldModel
    from snks.agent.vector_bootstrap import load_from_textbook
    from snks.encoder.tile_segmenter import load_tile_segmenter

    checkpoint_path = (
        Path(__file__).parent.parent / "demos" / "checkpoints" / "exp135" / "segmenter_9x9.pt"
    )
    segmenter = load_tile_segmenter(str(checkpoint_path), device=device)

    model = VectorWorldModel(dim=model_dim, n_locations=n_locations, seed=seed, device=device)
    textbook_path = Path(__file__).parent.parent / "configs" / "crafter_textbook.yaml"
    stats = load_from_textbook(model, textbook_path)
    print(f"Segmenter: {checkpoint_path.name}  Textbook seeded: {stats}")
    return model, segmenter, textbook_path


def run_episodes(
    label: str,
    n_episodes: int,
    max_steps: int,
    model_dim: int,
    n_locations: int,
    seed: int,
    device,
    with_post_mortem: bool,
) -> list[dict]:
    """Run n_episodes, optionally with PostMortemLearner updating stimuli."""
    from snks.agent.vector_mpc_agent import run_vector_mpc_episode
    from snks.agent.stimuli import StimuliLayer, SurvivalAversion, HomeostasisStimulus
    from snks.agent.perception import HomeostaticTracker
    from snks.agent.crafter_textbook import CrafterTextbook
    from snks.agent.crafter_pixel_env import CrafterPixelEnv
    from snks.agent.post_mortem import PostMortemAnalyzer, PostMortemLearner

    model, segmenter, textbook_path = _build_model_and_segmenter(
        model_dim, n_locations, seed, device
    )
    tb = CrafterTextbook(str(textbook_path))
    vitals = ["health", "food", "drink", "energy"]

    if with_post_mortem:
        learner = PostMortemLearner()
        analyzer = PostMortemAnalyzer()
        stimuli = learner.build_stimuli(vitals)
    else:
        stimuli = StimuliLayer([SurvivalAversion(), HomeostasisStimulus(vital_vars=vitals)])

    print(f"\n=== {label} ({n_episodes} ep, post_mortem={with_post_mortem}) ===")
    results = []
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
        )
        results.append(metrics)

        elapsed = time.time() - t0
        eta = elapsed / (ep + 1) * (n_episodes - ep - 1)
        print(
            f"  ep{ep:2d}: len={metrics.get('avg_len', 0):4.0f} "
            f"death_cause={metrics.get('death_cause', '?'):12s} "
            f"cause={metrics.get('cause', '?'):6s} "
            f"[{elapsed:.0f}s eta={eta:.0f}s]"
        )

        if with_post_mortem:
            attribution = PostMortemAnalyzer().attribute(
                metrics.get("damage_log", []), metrics.get("episode_steps", 0)
            )
            learner.update(attribution)
            stimuli = learner.build_stimuli(vitals)
            if attribution:
                print(f"    attribution: {
                    {k: round(v, 3) for k, v in sorted(attribution.items(), key=lambda x: -x[1])}
                }")
                print(f"    params: food_thr={learner.food_threshold:.2f} "
                      f"drink_thr={learner.drink_threshold:.2f} "
                      f"health_w={learner.health_weight:.2f}")

    return results


def compute_gates(with_pm: list[dict], without_pm: list[dict]) -> dict:
    def death_cause_count(results: list[dict], cause: str) -> int:
        return sum(1 for r in results if r.get("death_cause") == cause)

    # Gate 1: zombie_deaths decrease within with_pm run
    early = with_pm[:7]
    late = with_pm[13:]
    zombie_early = death_cause_count(early, "zombie")
    zombie_late = death_cause_count(late, "zombie")

    # Gate 2: starvation deaths lower with post-mortem
    starv_with = death_cause_count(with_pm, "starvation")
    starv_without = death_cause_count(without_pm, "starvation")

    # Gate 3: survival not degraded
    avg_survival_with = float(np.mean([r.get("avg_len", 0) for r in with_pm]))

    gates = {
        "zombie_deaths_decrease": zombie_late < zombie_early,
        "starvation_decrease": starv_with < starv_without,
        "survival_holds": avg_survival_with >= 155.0,
    }

    print(f"\n=== Gate Results ===")
    print(f"  zombie_deaths early(ep1-7)={zombie_early} late(ep14-20)={zombie_late} "
          f"→ {'✓' if gates['zombie_deaths_decrease'] else '✗'}")
    print(f"  starvation with_pm={starv_with} without_pm={starv_without} "
          f"→ {'✓' if gates['starvation_decrease'] else '✗'}")
    print(f"  avg_survival(with_pm)={avg_survival_with:.1f} (≥155) "
          f"→ {'✓' if gates['survival_holds'] else '✗'}")

    return {
        "gates": gates,
        "zombie_early": zombie_early,
        "zombie_late": zombie_late,
        "starvation_with_pm": starv_with,
        "starvation_without_pm": starv_without,
        "avg_survival_with_pm": avg_survival_with,
        "avg_survival_without_pm": float(np.mean([r.get("avg_len", 0) for r in without_pm])),
    }


def main():
    from snks.encoder.tile_segmenter import pick_device

    device = torch.device(pick_device())
    model_dim = 16384
    n_locations = 50000
    seed = 42
    n_episodes = 20
    max_steps = 1000

    print(f"device={device}, dim={model_dim}, locs={n_locations}")

    with_pm = run_episodes(
        "with_post_mortem", n_episodes, max_steps, model_dim, n_locations,
        seed=seed, device=device, with_post_mortem=True,
    )
    without_pm = run_episodes(
        "without_post_mortem", n_episodes, max_steps, model_dim, n_locations,
        seed=seed, device=device, with_post_mortem=False,
    )

    gate_results = compute_gates(with_pm, without_pm)
    gates = gate_results["gates"]
    passed = sum(gates.values())
    total = len(gates)

    print(f"\nGates: {passed}/{total} {'✓ PASS' if passed == total else '✗ FAIL'}")

    out = {
        **gate_results,
        "gates_passed": passed,
        "gates_total": total,
        "n_episodes": n_episodes,
        "with_pm_episodes": [
            {k: v for k, v in r.items() if k != "damage_log"} for r in with_pm
        ],
        "without_pm_episodes": [
            {k: v for k, v in r.items() if k != "damage_log"} for r in without_pm
        ],
    }

    out_path = Path(__file__).parent.parent / "_docs" / "stage86_eval.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
