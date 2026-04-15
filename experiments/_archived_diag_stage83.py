"""Stage 83 diagnostic — 1 episode, full trace every step.

Answers:
1. What concepts does the agent see?
2. What does predict() return for each concept×action?
3. What plans are generated and how do they score?
4. Why does surprise=0?
5. Why does the agent never gather?

Run on minipc:
  HSA_OVERRIDE_GFX_VERSION=11.0.0 PYTHONPATH=src python experiments/diag_stage83.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.backends.cudnn.enabled = False

from snks.agent.vector_world_model import VectorWorldModel, hamming_similarity
from snks.agent.vector_bootstrap import load_from_textbook
from snks.agent.vector_sim import (
    VectorState, VectorPlan, VectorPlanStep,
    simulate_forward, score_trajectory,
)
from snks.agent.vector_mpc_agent import (
    generate_candidate_plans, expand_to_primitive,
    _update_spatial_map, DynamicEntityTracker,
)
from snks.agent.perception import HomeostaticTracker, perceive_tile_field
from snks.agent.crafter_spatial_map import CrafterSpatialMap
from snks.agent.crafter_textbook import CrafterTextbook
from snks.encoder.tile_segmenter import load_tile_segmenter, pick_device
from snks.agent.crafter_pixel_env import CrafterPixelEnv, ACTION_TO_IDX


def main():
    device = torch.device(pick_device())
    print(f"Device: {device}")

    # --- Setup ---
    checkpoint = Path("demos/checkpoints/exp135/segmenter_9x9.pt")
    segmenter = load_tile_segmenter(str(checkpoint), device=device)
    print("Segmenter loaded")

    model = VectorWorldModel(dim=16384, n_locations=50000, seed=42, device=device)
    textbook_path = Path("configs/crafter_textbook.yaml")
    stats = load_from_textbook(model, textbook_path)
    print(f"Bootstrap: {stats}")
    print(f"Concepts: {list(model.concepts.keys())}")
    print(f"Actions: {list(model.actions.keys())}")
    print(f"Roles: {list(model.roles.keys())}")
    print(f"SDM writes: {model.memory.n_writes}, radius: {model.memory.activation_radius}")

    # --- Verify bootstrap predictions ---
    print("\n=== BOOTSTRAP PREDICTION CHECK ===")
    target_actions = ["do", "make", "place"]
    for concept in ["tree", "stone", "cow", "water", "zombie", "empty", "table"]:
        for action in target_actions:
            eff, conf = model.predict(concept, action)
            if conf > 0.01:
                decoded = model.decode_effect(eff)
                print(f"  predict({concept}, {action}) → conf={conf:.3f}, decoded={decoded}")

    # --- Init env ---
    tb = CrafterTextbook(str(textbook_path))
    tracker = HomeostaticTracker()
    tracker.init_from_textbook(tb.body_block)

    env = CrafterPixelEnv(seed=42)
    rng = np.random.RandomState(42)

    entity_tracker = DynamicEntityTracker()
    for cid in ("zombie", "skeleton", "cow"):
        entity_tracker.register_dynamic_concept(cid)

    pixels, info = env.reset()
    spatial_map = CrafterSpatialMap()
    vital_vars = ["health", "food", "drink", "energy"]

    prev_inv = None
    prev_body = None
    prev_action = None
    prev_player_pos = None

    print("\n=== EPISODE START ===")
    for step in range(200):
        inv = dict(info.get("inventory", {}))
        body = {v: float(info.get(v, 9.0)) for v in vital_vars}
        player_pos = tuple(info.get("player_pos", (32, 32)))

        # Perception
        vf = perceive_tile_field(pixels, segmenter)
        _update_spatial_map(spatial_map, vf, player_pos)
        entity_tracker.update(vf, player_pos)

        if prev_inv is not None:
            tracker.update(prev_inv, inv, vf.visible_concepts())

        # Diagnostic: what does the agent see?
        visible = vf.visible_concepts()
        known = set(visible) | set(spatial_map.known_objects.keys())

        if step < 10 or step % 20 == 0:
            print(f"\n--- Step {step} ---")
            print(f"  pos={player_pos}, near={vf.near_concept}, visible={visible}")
            print(f"  known={known}, spatial_map_objects={spatial_map.known_objects}")
            print(f"  body={body}")
            print(f"  inv={inv}")

        # Build state
        state = VectorState(
            inventory=inv, body=body, player_pos=player_pos,
            step=step, last_action=prev_action, spatial_map=spatial_map,
        )

        # Generate candidates
        candidates = generate_candidate_plans(
            model, state, spatial_map, visible,
            beam_width=5, max_depth=3,
        )

        # Score
        scored = []
        for plan in candidates:
            traj = simulate_forward(model, plan, state, horizon=10, vital_vars=vital_vars)
            score = score_trajectory(traj, vital_vars)
            scored.append((score, plan, traj))
        scored.sort(key=lambda x: x[0], reverse=True)

        if step < 10 or step % 20 == 0:
            print(f"  Candidates: {len(candidates)}")
            for i, (sc, pl, _) in enumerate(scored[:5]):
                print(f"    #{i}: score={sc}, plan={pl.origin}, steps={[f'{s.target}:{s.action}' for s in pl.steps]}")

        best_score, best_plan, best_traj = scored[0]

        # Expand to primitive
        if best_plan.steps:
            primitive = expand_to_primitive(
                best_plan.steps[0], player_pos, spatial_map, model, rng,
                last_action=prev_action,
            )
        else:
            move_actions = [a for a in model.actions if a.startswith("move_")]
            primitive = str(rng.choice(move_actions)) if move_actions else "move_right"

        if step < 10 or step % 20 == 0:
            print(f"  → primitive={primitive} (from plan={best_plan.origin})")

        # Surprise check
        if prev_inv is not None and prev_body is not None and prev_action is not None:
            inv_deltas = {k: inv.get(k, 0) - prev_inv.get(k, 0)
                         for k in set(inv) | set(prev_inv) if inv.get(k, 0) != prev_inv.get(k, 0)}
            body_deltas = {k: int(round(body.get(k, 0) - prev_body.get(k, 0)))
                          for k in vital_vars if abs(body.get(k, 0) - prev_body.get(k, 0)) > 0.01}
            all_deltas = {**inv_deltas, **body_deltas}
            if all_deltas and step < 10:
                print(f"  Deltas: {all_deltas}, prev_action={prev_action}")
                target = vf.near_concept if prev_action in ("do", "place", "make") else None
                print(f"  Surprise target: {target} (filter: prev_action in do/place/make = {prev_action in ('do', 'place', 'make')})")

        # Step env
        prev_inv = dict(inv)
        prev_body = dict(body)
        prev_action = primitive
        prev_player_pos = player_pos

        pixels, _reward, done, info = env.step(primitive)

        if done:
            final_body = {v: float(info.get(v, 0)) for v in vital_vars}
            print(f"\n=== DIED at step {step+1}, body={final_body}, inv={info.get('inventory', {})} ===")
            break

    print(f"\nTotal steps: {step+1}")


if __name__ == "__main__":
    main()
