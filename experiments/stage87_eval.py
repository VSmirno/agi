"""Stage 87: Curiosity About Death — eval gates on Crafter.

Run on minipc ONLY:
  ssh minipc "cd /opt/agi && git pull origin main"
  ssh minipc "tmux new-session -d -s stage87 'cd /opt/agi && source venv/bin/activate && \
    HSA_OVERRIDE_GFX_VERSION=11.0.0 PYTHONPATH=src python -u experiments/stage87_eval.py 2>&1 | \
    tee _docs/stage87_eval.log'"

Gates:
  1. hypothesis_formed:          tracker.n_verifiable() >= 1 after 20 episodes
  2. curiosity_active_episodes:  >= 5 episodes where active_hypothesis != None
  3. survival_holds:             avg_survival >= 155

Results saved to _docs/stage87_eval.json.
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
) -> dict:
    """Run n_episodes with HypothesisTracker + CuriosityStimulus.

    Returns dict with per-episode results and hypothesis tracking data.
    """
    from snks.agent.vector_mpc_agent import run_vector_mpc_episode
    from snks.agent.perception import HomeostaticTracker
    from snks.agent.crafter_textbook import CrafterTextbook
    from snks.agent.crafter_pixel_env import CrafterPixelEnv
    from snks.agent.post_mortem import PostMortemAnalyzer, PostMortemLearner
    from snks.agent.death_hypothesis import HypothesisTracker

    model, segmenter, textbook_path = _build_model_and_segmenter(
        model_dim, n_locations, seed, device
    )
    tb = CrafterTextbook(str(textbook_path))
    vitals = ["health", "food", "drink", "energy"]

    learner = PostMortemLearner()
    analyzer = PostMortemAnalyzer()
    tracker = HypothesisTracker()

    stimuli = learner.build_stimuli(vitals, hypothesis=None)

    print(f"\n=== {label} ({n_episodes} ep) ===")
    results = []
    t0 = time.time()
    curiosity_active_count = 0

    for ep in range(n_episodes):
        ep_seed = seed + ep
        env = CrafterPixelEnv(seed=ep_seed)
        hom_tracker = HomeostaticTracker()
        hom_tracker.init_from_textbook(tb.body_block)

        metrics = run_vector_mpc_episode(
            env=env,
            segmenter=segmenter,
            model=model,
            tracker=hom_tracker,
            max_steps=max_steps,
            stimuli=stimuli,
            textbook=tb,
            verbose=True,
        )
        results.append(metrics)

        elapsed = time.time() - t0
        eta = elapsed / (ep + 1) * (n_episodes - ep - 1)
        active_h = tracker.active_hypothesis()
        curiosity_status = f"H={active_h.cause}+{active_h.vital}" if active_h else "H=none"
        print(
            f"  ep{ep:2d}: len={metrics.get('episode_steps', 0):4.0f} "
            f"death={metrics.get('death_cause', '?'):12s} "
            f"{curiosity_status:20s} "
            f"[{elapsed:.0f}s eta={eta:.0f}s]"
        )

        # --- Post-episode learning ---
        damage_log = metrics.get("damage_log", [])
        attribution = analyzer.attribute(damage_log, metrics.get("episode_steps", 0))
        vitals_at_death = damage_log[-1].vitals if damage_log else {}

        learner.update(attribution)
        tracker.record(attribution, vitals_at_death)

        if attribution:
            attr_str = {k: round(v, 3) for k, v in sorted(attribution.items(), key=lambda x: -x[1])}
            print(f"    attribution: {attr_str}")
            print(f"    params: food_thr={learner.food_threshold:.2f} "
                  f"drink_thr={learner.drink_threshold:.2f} "
                  f"health_w={learner.health_weight:.2f}")

        # Rebuild stimuli with updated hypothesis for next episode
        new_hypothesis = tracker.active_hypothesis()
        stimuli = learner.build_stimuli(vitals, hypothesis=new_hypothesis)

        if new_hypothesis is not None:
            curiosity_active_count += 1
            print(f"    {new_hypothesis}")

        n_verifiable = tracker.n_verifiable()
        if n_verifiable > 0:
            print(f"    verifiable hypotheses: {n_verifiable}")

    # --- Final hypothesis report ---
    print(f"\n=== Hypothesis Summary ===")
    for h in tracker.all_hypotheses():
        if h.n_observed > 0:
            status = "VERIFIABLE" if h.is_verifiable else "forming"
            print(f"  [{status}] {h}")

    return {
        "results": results,
        "n_verifiable": tracker.n_verifiable(),
        "curiosity_active_episodes": curiosity_active_count,
        "all_hypotheses": [
            {
                "cause": h.cause,
                "vital": h.vital,
                "threshold": h.threshold,
                "n_supporting": h.n_supporting,
                "n_observed": h.n_observed,
                "support_rate": h.support_rate,
                "is_verifiable": h.is_verifiable,
            }
            for h in tracker.all_hypotheses()
            if h.n_observed > 0
        ],
    }


def compute_gates(run_data: dict) -> dict:
    results = run_data["results"]
    n_verifiable = run_data["n_verifiable"]
    curiosity_active = run_data["curiosity_active_episodes"]

    avg_survival = float(np.mean([r.get("episode_steps", 0) for r in results]))

    gates = {
        "hypothesis_formed": n_verifiable >= 1,
        "curiosity_active_episodes": curiosity_active >= 5,
        "survival_holds": avg_survival >= 155.0,
    }

    print(f"\n=== Gate Results ===")
    print(f"  hypothesis_formed: n_verifiable={n_verifiable} (>=1) "
          f"→ {'✓' if gates['hypothesis_formed'] else '✗'}")
    print(f"  curiosity_active_episodes: {curiosity_active} (>=5) "
          f"→ {'✓' if gates['curiosity_active_episodes'] else '✗'}")
    print(f"  survival_holds: avg={avg_survival:.1f} (>=155) "
          f"→ {'✓' if gates['survival_holds'] else '✗'}")

    return {
        "gates": gates,
        "n_verifiable": n_verifiable,
        "curiosity_active_episodes": curiosity_active,
        "avg_survival": avg_survival,
        "all_hypotheses": run_data["all_hypotheses"],
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

    run_data = run_episodes(
        "stage87", n_episodes, max_steps, model_dim, n_locations,
        seed=seed, device=device,
    )

    gate_results = compute_gates(run_data)
    gates = gate_results["gates"]
    passed = sum(gates.values())
    total = len(gates)

    print(f"\nGates: {passed}/{total} {'✓ PASS' if passed == total else '✗ FAIL'}")

    out = {
        **gate_results,
        "gates_passed": passed,
        "gates_total": total,
        "n_episodes": n_episodes,
        "episodes": [
            {k: v for k, v in r.items() if k != "damage_log"}
            for r in run_data["results"]
        ],
    }

    out_path = Path(__file__).parent.parent / "_docs" / "stage87_eval.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
