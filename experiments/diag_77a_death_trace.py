"""Diagnostic: trace MPC agent actions during early-death episodes.

Runs exp137-style MPC loop but logs every step of episodes that die
quickly (< 100 steps). Reveals what action sequence leads to premature
death in warmup-safe mode (where random walk survives 337 steps).

Goal: find the specific action pattern that causes MPC to die faster
than random walk in the safe environment.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch

from snks.agent.concept_store import ConceptStore
from snks.agent.crafter_pixel_env import CrafterPixelEnv
from snks.agent.crafter_textbook import CrafterTextbook
from snks.agent.crafter_spatial_map import CrafterSpatialMap
from snks.agent.mpc_agent import (
    DynamicEntityTracker,
    build_sim_state,
    generate_candidate_plans,
    outcome_to_verify,
    score_trajectory,
    update_spatial_map_from_viewport,
)
from snks.agent.concept_store import _expand_to_primitive as expand_to_primitive
from snks.agent.perception import HomeostaticTracker, verify_outcome
from snks.encoder.cnn_encoder import disable_rocm_conv
from snks.encoder.tile_segmenter import load_tile_segmenter, pick_device


STAGE75_CHECKPOINT = Path("demos/checkpoints/exp135/segmenter_9x9.pt")


def run_trace_episode(
    env,
    segmenter,
    store,
    tracker,
    rng,
    max_steps: int = 100,
    horizon: int = 40,
) -> dict:
    """Run one episode and log every step. Returns trace."""
    from snks.agent.continuous_agent import perceive_tile_field

    entity_tracker = DynamicEntityTracker()
    for rule in store.passive_rules:
        if rule.kind == "passive_movement" and rule.concept:
            entity_tracker.register_dynamic_concept(rule.concept)

    pixels, info = env.reset()
    spatial_map = CrafterSpatialMap()
    prev_inv = None
    prev_action = None
    trace = []

    for step in range(max_steps):
        inv = dict(info.get("inventory", {}))
        player_pos = tuple(info.get("player_pos", (32, 32)))
        vf = perceive_tile_field(pixels, segmenter)
        update_spatial_map_from_viewport(spatial_map, vf, player_pos)
        entity_tracker.update(vf, player_pos)
        if prev_inv is not None:
            tracker.update(prev_inv, inv, vf.visible_concepts())

        state = build_sim_state(
            inventory=inv,
            player_pos=player_pos,
            spatial_map=spatial_map,
            entity_tracker=entity_tracker,
            tracker=tracker,
            last_action=prev_action,
            step=step,
        )
        candidates = generate_candidate_plans(state, store, tracker, horizon=horizon)
        scored = []
        for plan in candidates:
            traj = store.simulate_forward(plan, state, tracker, horizon=horizon)
            score = score_trajectory(traj, tracker)
            scored.append((score, plan, traj))
        scored.sort(key=lambda x: x[0], reverse=True)
        _, best_plan, best_traj = scored[0]

        primitive = (
            expand_to_primitive(best_plan.steps[0], state, store)
            if best_plan.steps else "move_right"
        )

        # Log this step BEFORE executing
        trace.append({
            "step": step,
            "player_pos": player_pos,
            "near": vf.near_concept,
            "visible": list(vf.visible_concepts()),
            "entities": [(e.concept_id, e.pos) for e in entity_tracker.current()],
            "body": {k: inv.get(k, 0) for k in ["health", "food", "drink", "energy"]},
            "inventory_resources": {k: inv.get(k, 0) for k in ["wood", "wood_sword", "wood_pickaxe", "stone_item"]},
            "plan": best_plan.origin,
            "plan_len": len(best_plan.steps),
            "plan_first": f"{best_plan.steps[0].action} {best_plan.steps[0].target}" if best_plan.steps else "none",
            "primitive": primitive,
            "num_candidates": len(candidates),
        })

        pixels, _, done, info = env.step(primitive)
        new_inv = dict(info.get("inventory", {}))
        outcome = outcome_to_verify(primitive, inv, new_inv)
        if outcome:
            verify_outcome(vf.near_concept, primitive, outcome, store)
        prev_inv = inv
        prev_action = primitive
        if done:
            # Log final state
            trace.append({
                "step": step + 1,
                "player_pos": tuple(info.get("player_pos", (32, 32))),
                "body": {k: new_inv.get(k, 0) for k in ["health", "food", "drink", "energy"]},
                "inventory_resources": {k: new_inv.get(k, 0) for k in ["wood", "wood_sword", "wood_pickaxe", "stone_item"]},
                "primitive": "<DONE>",
                "plan": "<DONE>",
                "near": "<DONE>",
                "visible": [],
                "entities": [],
                "plan_len": 0,
                "plan_first": "",
                "num_candidates": 0,
            })
            break
    return {"length": step + 1, "trace": trace, "final_inv": new_inv if done else dict(info.get("inventory", {}))}


def main():
    disable_rocm_conv()
    device = pick_device()
    segmenter = load_tile_segmenter(str(STAGE75_CHECKPOINT), device=device)
    store = ConceptStore()
    tb = CrafterTextbook("configs/crafter_textbook.yaml")
    tb.load_into(store)
    tracker = HomeostaticTracker()
    tracker.init_from_textbook(tb.body_block, store.passive_rules)

    # Run 5 episodes. For each, print the trace if length < 150
    for ep in range(5):
        env = CrafterPixelEnv(seed=ep * 11 + 300)
        try:
            env._env._balance_chunk = lambda *a, **kw: None
        except Exception:
            pass
        rng = np.random.RandomState(ep + 300)
        result = run_trace_episode(env, segmenter, store, tracker, rng, max_steps=150, horizon=40)
        trace = result["trace"]
        length = result["length"]
        print(f"\n{'='*80}\nEpisode {ep}: length={length}, final_inv={result['final_inv']}\n{'='*80}")

        if length < 150:
            # Dying episode — print every step
            for t in trace:
                body = t["body"]
                print(
                    f"s{t['step']:3d} pos={t['player_pos']} "
                    f"H{body['health']}F{body['food']}D{body['drink']}E{body['energy']} "
                    f"near={t['near']:<10} ent={t['entities']} "
                    f"plan={t['plan']}({t['plan_len']}) {t['plan_first']:<20} → {t['primitive']}"
                )
        else:
            # Long episode — just first/last 10
            print("First 10 steps:")
            for t in trace[:10]:
                body = t["body"]
                print(f"  s{t['step']:3d} H{body['health']}F{body['food']}D{body['drink']}E{body['energy']} plan={t['plan']} {t['plan_first']} → {t['primitive']}")
            print("Last 10 steps:")
            for t in trace[-10:]:
                body = t["body"]
                print(f"  s{t['step']:3d} H{body['health']}F{body['food']}D{body['drink']}E{body['energy']} plan={t['plan']} {t['plan_first']} → {t['primitive']}")


if __name__ == "__main__":
    main()
