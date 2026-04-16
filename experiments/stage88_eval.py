"""Stage 88: Knowledge Flow — Textbook Promotion, 5-generation eval.

Run on minipc ONLY:
  ssh minipc "cd /opt/agi && git pull origin main"
  ssh minipc "tmux new-session -d -s stage88 'cd /opt/agi && source venv/bin/activate && \
    HSA_OVERRIDE_GFX_VERSION=11.0.0 PYTHONPATH=src python -u experiments/stage88_eval.py 2>&1 | \
    tee _docs/stage88_eval.log'"

Gates:
  1. primary:   gen5_avg_survival / gen1_avg_survival >= 1.20  OR  gen5_avg_survival >= 210
  2. secondary: n_promoted_cumulative >= 2 after gen2

Results saved to _docs/stage88_eval.json.
Also writes/updates configs/promoted_hypotheses.yaml between generations.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from statistics import mean

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


def run_generation(
    gen: int,
    n_episodes: int,
    max_steps: int,
    model_dim: int,
    n_locations: int,
    seed: int,
    device,
    promoted_path: Path,
):
    """Run one generation of n_episodes, using promoted hypotheses from previous gens.

    Returns dict with per-episode results, hypothesis data, and learner params.
    """
    from snks.agent.vector_mpc_agent import run_vector_mpc_episode
    from snks.agent.perception import HomeostaticTracker
    from snks.agent.crafter_textbook import CrafterTextbook
    from snks.agent.crafter_pixel_env import CrafterPixelEnv
    from snks.agent.post_mortem import PostMortemAnalyzer, PostMortemLearner
    from snks.agent.death_hypothesis import HypothesisTracker
    from snks.agent.textbook_promoter import TextbookPromoter

    promoter = TextbookPromoter()
    promoted = promoter.load(promoted_path)

    print(f"\n{'='*60}")
    print(f"  GEN {gen}  |  promoted_in={len(promoted)}  |  seed={seed}")
    if promoted:
        for ph in promoted:
            print(f"    prior: {ph}")
    print(f"{'='*60}")

    model, segmenter, textbook_path = _build_model_and_segmenter(
        model_dim, n_locations, seed, device
    )
    tb = CrafterTextbook(str(textbook_path))
    vitals = ["health", "food", "drink", "energy"]

    learner = PostMortemLearner.from_promoted(promoted)
    analyzer = PostMortemAnalyzer()
    tracker = HypothesisTracker(initial=promoted)

    if promoted:
        print(f"  Learner init: food_thr={learner.food_threshold:.2f} "
              f"drink_thr={learner.drink_threshold:.2f} "
              f"health_w={learner.health_weight:.2f}")

    stimuli = learner.build_stimuli(vitals, hypothesis=tracker.active_hypothesis())

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
            f"  gen{gen} ep{ep:2d}: len={metrics.get('episode_steps', 0):4.0f} "
            f"death={metrics.get('death_cause', '?'):12s} "
            f"{curiosity_status:22s} "
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
            print(f"    attr: {attr_str}  "
                  f"food_thr={learner.food_threshold:.2f} "
                  f"drink_thr={learner.drink_threshold:.2f} "
                  f"health_w={learner.health_weight:.2f}")

        new_hypothesis = tracker.active_hypothesis()
        stimuli = learner.build_stimuli(vitals, hypothesis=new_hypothesis)
        if new_hypothesis is not None:
            curiosity_active_count += 1

    # --- End of generation: hypothesis report ---
    print(f"\n  === Gen {gen} Hypothesis Summary ===")
    for h in tracker.all_hypotheses():
        if h.n_observed > 0:
            status = "VERIFIABLE" if h.is_verifiable else "forming  "
            promote_mark = " [PROMOTABLE]" if promoter.should_promote(h) else ""
            print(f"    [{status}] {h}{promote_mark}")

    # --- Promote and save ---
    to_promote = [h for h in tracker.all_hypotheses() if promoter.should_promote(h)]
    promoter.save(to_promote, promoted_path)
    print(f"  Promoted {len(to_promote)} hypothesis(es) → {promoted_path}")

    avg_survival = float(np.mean([r.get("episode_steps", 0) for r in results]))
    print(f"  Gen {gen} avg_survival={avg_survival:.1f}")

    return {
        "gen": gen,
        "avg_survival": avg_survival,
        "promoted_in": len(promoted),
        "promoted_out": len(to_promote),
        "n_verifiable": tracker.n_verifiable(),
        "curiosity_active_episodes": curiosity_active_count,
        "all_hypotheses": [
            {
                "cause": h.cause,
                "vital": h.vital,
                "threshold": h.threshold,
                "n_supporting": h.n_supporting,
                "n_observed": h.n_observed,
                "support_rate": round(h.support_rate, 3),
                "is_verifiable": h.is_verifiable,
                "promotable": promoter.should_promote(h),
            }
            for h in tracker.all_hypotheses()
            if h.n_observed > 0
        ],
        "episodes": [
            {k: v for k, v in r.items() if k != "damage_log"}
            for r in results
        ],
    }


def compute_gates(gen_results: list[dict]) -> dict:
    gen1 = gen_results[0]["avg_survival"]
    gen5 = gen_results[-1]["avg_survival"]
    ratio = gen5 / gen1 if gen1 > 0 else 0.0

    # Secondary gate: n_promoted after gen2
    n_promoted_after_gen2 = gen_results[1]["promoted_in"] if len(gen_results) >= 2 else 0

    primary_pass = (ratio >= 1.20) or (gen5 >= 210.0)
    secondary_pass = n_promoted_after_gen2 >= 2

    gates = {
        "primary": primary_pass,
        "secondary": secondary_pass,
    }

    print(f"\n{'='*60}")
    print(f"  GATE RESULTS")
    print(f"{'='*60}")
    print(f"  gen1_avg_survival = {gen1:.1f}")
    print(f"  gen5_avg_survival = {gen5:.1f}")
    print(f"  ratio = {ratio:.3f} (>= 1.20? {'✓' if ratio >= 1.20 else '✗'})")
    print(f"  gen5 >= 210? {'✓' if gen5 >= 210 else '✗'}")
    print(f"  primary (ratio>=1.20 OR gen5>=210): {'✓ PASS' if primary_pass else '✗ FAIL'}")
    print(f"  secondary (n_promoted_after_gen2={n_promoted_after_gen2} >= 2): "
          f"{'✓ PASS' if secondary_pass else '✗ FAIL'}")

    return {
        "gates": gates,
        "gen1_avg_survival": gen1,
        "gen5_avg_survival": gen5,
        "ratio": round(ratio, 3),
        "n_promoted_after_gen2": n_promoted_after_gen2,
    }


def main():
    from snks.encoder.tile_segmenter import pick_device

    device = torch.device(pick_device())
    model_dim = 16384
    n_locations = 50000
    seed = 42
    n_episodes = 20
    max_steps = 1000
    n_gens = 5

    promoted_path = Path(__file__).parent.parent / "configs" / "promoted_hypotheses.yaml"

    # Reset promoted_hypotheses.yaml to empty at start of experiment
    promoted_path.write_text("# Auto-generated by TextbookPromoter. Do not edit manually.\nhypotheses: []\n")
    print(f"Reset {promoted_path}")
    print(f"device={device}, dim={model_dim}, locs={n_locations}, n_gens={n_gens}, n_ep={n_episodes}")

    all_gen_results = []
    for gen in range(1, n_gens + 1):
        gen_data = run_generation(
            gen=gen,
            n_episodes=n_episodes,
            max_steps=max_steps,
            model_dim=model_dim,
            n_locations=n_locations,
            seed=seed,
            device=device,
            promoted_path=promoted_path,
        )
        all_gen_results.append(gen_data)

        # Summary line
        print(f"\n  >>> Gen {gen} summary: avg={gen_data['avg_survival']:.1f} "
              f"promoted_out={gen_data['promoted_out']}")

    gate_results = compute_gates(all_gen_results)
    gates = gate_results["gates"]
    passed = sum(gates.values())
    total = len(gates)

    print(f"\nGates: {passed}/{total} {'✓ PASS' if passed == total else '✗ FAIL'}")

    out = {
        **gate_results,
        "gates_passed": passed,
        "gates_total": total,
        "n_gens": n_gens,
        "n_episodes_per_gen": n_episodes,
        "generations": all_gen_results,
    }

    out_path = Path(__file__).parent.parent / "_docs" / "stage88_eval.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
