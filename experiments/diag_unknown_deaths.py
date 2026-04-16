"""Diagnostic: inspect DamageEvent.nearby_cids for 'unknown' death episodes.

Runs N episodes, logs full damage_log for unknown-cause deaths.
Goal: understand why _detect_sources returns ["unknown"].

Run on minipc:
  HSA_OVERRIDE_GFX_VERSION=11.0.0 PYTHONPATH=src python -u experiments/diag_unknown_deaths.py
"""

from __future__ import annotations

import json
from pathlib import Path
from collections import Counter

import torch
torch.backends.cudnn.enabled = False


def main():
    from snks.agent.vector_mpc_agent import run_vector_mpc_episode
    from snks.agent.perception import HomeostaticTracker
    from snks.agent.crafter_textbook import CrafterTextbook
    from snks.agent.crafter_pixel_env import CrafterPixelEnv
    from snks.agent.post_mortem import PostMortemAnalyzer, PostMortemLearner, DamageEvent
    from snks.agent.death_hypothesis import HypothesisTracker
    from snks.agent.stimuli import StimuliLayer, SurvivalAversion, HomeostasisStimulus
    from snks.agent.vector_world_model import VectorWorldModel
    from snks.agent.vector_bootstrap import load_from_textbook
    from snks.encoder.tile_segmenter import load_tile_segmenter, pick_device

    device = torch.device(pick_device())
    root = Path(__file__).parent.parent

    checkpoint_path = root / "demos" / "checkpoints" / "exp136" / "segmenter_9x9.pt"
    textbook_path = root / "configs" / "crafter_textbook.yaml"

    segmenter = load_tile_segmenter(str(checkpoint_path), device=device)
    model = VectorWorldModel(dim=16384, n_locations=50000, seed=42, device=device)
    load_from_textbook(model, textbook_path)
    tb = CrafterTextbook(str(textbook_path))
    vitals = ["health", "food", "drink", "energy"]

    learner = PostMortemLearner()
    stimuli = learner.build_stimuli(vitals)

    N_EPISODES = 30
    unknown_episodes = []
    all_causes = Counter()

    for ep in range(N_EPISODES):
        env = CrafterPixelEnv(seed=42 + ep)
        hom_tracker = HomeostaticTracker()
        hom_tracker.init_from_textbook(tb.body_block)

        metrics = run_vector_mpc_episode(
            env=env,
            segmenter=segmenter,
            model=model,
            tracker=hom_tracker,
            max_steps=1000,
            stimuli=stimuli,
            textbook=tb,
            verbose=False,
        )

        damage_log: list[DamageEvent] = metrics.get("damage_log", [])
        analyzer = PostMortemAnalyzer()
        attribution = analyzer.attribute(damage_log, metrics["episode_steps"])
        death_cause = max(attribution, key=attribution.__getitem__) if attribution else "alive"
        env_cause = metrics.get("cause", "?")
        all_causes[death_cause] += 1

        print(f"ep{ep:2d}: steps={metrics['episode_steps']:4d} death={death_cause:12s} env={env_cause}")

        if death_cause == "unknown":
            print(f"  damage_log ({len(damage_log)} events):")
            for ev in damage_log:
                # Show all entities with distances
                entities_str = ", ".join(
                    f"{cid}@{dist}" for cid, dist in sorted(ev.nearby_cids, key=lambda x: x[1])
                ) or "NO_ENTITIES"
                print(f"    step={ev.step:4d} delta={ev.health_delta:+.2f} "
                      f"food={ev.vitals.get('food',9):.1f} drink={ev.vitals.get('drink',9):.1f} "
                      f"energy={ev.vitals.get('energy',9):.1f} | {entities_str}")

        elif death_cause == "alive" and env_cause == "health":
            print(f"  *** ALIVE+HEALTH BUG: damage_log has {len(damage_log)} events, "
                  f"last 2 steps = {[e.step for e in damage_log[-2:]]}, "
                  f"episode_steps={metrics['episode_steps']}")

    print(f"\n=== Death cause distribution ({N_EPISODES} ep) ===")
    for cause, count in sorted(all_causes.items(), key=lambda x: -x[1]):
        print(f"  {cause:15s}: {count:3d} ({100*count/N_EPISODES:.0f}%)")


if __name__ == "__main__":
    main()
