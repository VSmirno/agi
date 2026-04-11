"""Stage 80 diagnostic — print ALL candidate plan scores at the FIRST tick.

Purpose: see why gather plans never get chosen in Crafter, even though
the diagnostic showed they're generated and the toy test confirmed
they fire `do tree` in their rollouts within ~3 ticks.

Hypothesis options:
  1. Gather plans don't complete in horizon=20 in Crafter (tree is
     too far away in spatial_map).
  2. The score_trajectory has_gain check returns 0 for some reason.
  3. Gather plan rollouts are dead (body decay over the chain).
  4. Something else.

Output: per-candidate print with origin, plan steps, score tuple,
trajectory ticks, terminated flag.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from snks.agent.concept_store import ConceptStore
from snks.agent.crafter_pixel_env import CrafterPixelEnv
from snks.agent.crafter_textbook import CrafterTextbook
from snks.agent.mpc_agent import (
    DynamicEntityTracker,
    build_sim_state,
    generate_candidate_plans,
    score_trajectory,
    update_spatial_map_from_viewport,
)
from snks.agent.crafter_spatial_map import CrafterSpatialMap
from snks.agent.perception import HomeostaticTracker, perceive_tile_field
from snks.encoder.cnn_encoder import disable_rocm_conv
from snks.encoder.tile_segmenter import load_tile_segmenter


STAGE75_CHECKPOINT = Path("demos/checkpoints/exp135/segmenter_9x9.pt")


def main_full() -> bool:
    return main()


def main() -> bool:
    disable_rocm_conv()
    from snks.encoder.tile_segmenter import pick_device as _seg_dev
    seg_device = _seg_dev()
    segmenter = load_tile_segmenter(str(STAGE75_CHECKPOINT), device=seg_device)
    print(f"segmenter device: {seg_device}")

    store = ConceptStore()
    tb = CrafterTextbook("configs/crafter_textbook.yaml")
    tb.load_into(store)
    tracker = HomeostaticTracker()
    tracker.init_from_textbook(tb.body_block, store.passive_rules)

    # Use the same seed_offset as Stage 78c/79 eval_run0 ep0
    env = CrafterPixelEnv(seed=0 * 11 + 1000)

    pixels, info = env.reset()
    spatial_map = CrafterSpatialMap()
    entity_tracker = DynamicEntityTracker()
    for rule in store.passive_rules:
        if rule.kind == "passive_movement" and rule.concept:
            entity_tracker.register_dynamic_concept(rule.concept)

    inv = dict(info.get("inventory", {}))
    player_pos = tuple(info.get("player_pos", (32, 32)))
    print(f"\nInitial player_pos: {player_pos}")
    print(f"Initial inv: {inv}")

    # Perceive + update maps
    vf = perceive_tile_field(pixels, segmenter)
    update_spatial_map_from_viewport(spatial_map, vf, player_pos)
    entity_tracker.update(vf, player_pos)

    print(f"\nVisible concepts: {sorted(vf.visible_concepts())}")
    print(f"near_concept: {vf.near_concept}")
    print(f"Spatial map contents (resources within ~7 tiles):")
    for (y, x), concept in spatial_map._map.items():
        if concept != "empty":
            dist = abs(y - player_pos[0]) + abs(x - player_pos[1])
            print(f"  {(y, x)} = {concept} (manhattan {dist})")

    # Build state
    state = build_sim_state(
        inventory=inv,
        player_pos=player_pos,
        spatial_map=spatial_map,
        entity_tracker=entity_tracker,
        tracker=tracker,
        last_action=None,
        step=0,
    )

    print(f"\nstate.body: {state.body}")

    # Generate candidates
    candidates = generate_candidate_plans(state, store, tracker, horizon=20)
    print(f"\nGenerated {len(candidates)} candidate plans")
    print()

    # Score each, print in detail
    scored: list = []
    for plan in candidates:
        traj = store.simulate_forward(
            plan, state, tracker, horizon=20,
            visible_concepts=vf.visible_concepts(),
        )
        sc = score_trajectory(traj, tracker)
        scored.append((sc, plan, traj))

    scored.sort(key=lambda x: x[0], reverse=True)

    print("=== RANKED CANDIDATES (best first) ===")
    for i, (sc, plan, traj) in enumerate(scored):
        steps_summary = ", ".join(
            f"{s.action}({s.target or '-'})" for s in plan.steps[:6]
        )
        if len(plan.steps) > 6:
            steps_summary += f"... +{len(plan.steps)-6}"
        # has_gain — count inv_gain events
        gain_events = [e for e in traj.events if e.kind == "inv_gain" and e.amount > 0]
        gain_str = ",".join(f"{e.var}+{e.amount}" for e in gain_events[:5])
        print(
            f"{i:2d} {plan.origin:8s} score={tuple(round(x, 3) if isinstance(x, float) else x for x in sc)}"
            f" ticks={traj.tick_count():2d} term={traj.terminated} reason={traj.terminated_reason} prog={traj.plan_progress}"
        )
        print(f"     plan: {steps_summary}")
        if gain_events:
            print(f"     inv_gain: {gain_str}")
        else:
            print(f"     inv_gain: NONE")
        print()

    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
