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
    prev_action: str | None = None

    print(f"\nInitial player_pos: {player_pos}")
    print(f"Initial inv: {inv}")

    # Run 10 steps. At each step, print ALL candidate scores.
    for step in range(10):
        print("\n" + "=" * 80)
        print(f"STEP {step}")
        print("=" * 80)

        # Perceive + update maps
        vf = perceive_tile_field(pixels, segmenter)
        update_spatial_map_from_viewport(spatial_map, vf, player_pos)
        entity_tracker.update(vf, player_pos)

        print(f"player_pos: {player_pos}  near: {vf.near_concept}")
        print(f"visible: {sorted(vf.visible_concepts())}")
        print(f"body: H{inv.get('health', 0)}F{inv.get('food', 0)}D{inv.get('drink', 0)}E{inv.get('energy', 0)}  W{inv.get('wood', 0)}")

        # Build state
        state = build_sim_state(
            inventory=inv,
            player_pos=player_pos,
            spatial_map=spatial_map,
            entity_tracker=entity_tracker,
            tracker=tracker,
            last_action=prev_action,
            step=step,
        )

        candidates = generate_candidate_plans(state, store, tracker, horizon=20)
        scored: list = []
        for plan in candidates:
            traj = store.simulate_forward(
                plan, state, tracker, horizon=20,
                visible_concepts=vf.visible_concepts(),
            )
            sc = score_trajectory(traj, tracker)
            scored.append((sc, plan, traj))

        scored.sort(key=lambda x: x[0], reverse=True)

        print(f"  {len(candidates)} candidates:")
        for i, (sc, plan, traj) in enumerate(scored[:12]):
            gain_events = [e for e in traj.events if e.kind == "inv_gain" and e.amount > 0]
            gain_str = ",".join(f"{e.var}+{e.amount}" for e in gain_events[:3]) if gain_events else "-"
            steps_summary = "+".join(
                f"{s.action}({s.target or '-'})" for s in plan.steps[:5]
            )
            if len(plan.steps) > 5:
                steps_summary += f"...+{len(plan.steps)-5}"
            score_str = "(" + ",".join(f"{round(x,2) if isinstance(x,float) else x}" for x in sc) + ")"
            print(f"  [{i:2d}] {plan.origin:8s} {score_str} term={traj.terminated} prog={traj.plan_progress} gain={gain_str}  {steps_summary[:60]}")

        # Pick best, execute
        sc, plan, traj = scored[0]
        if plan.steps:
            from snks.agent.concept_store import _expand_to_primitive as expand_to_primitive
            primitive = expand_to_primitive(plan.steps[0], state, store)
        else:
            primitive = "move_right"

        print(f"  → CHOSEN: {plan.origin} primitive={primitive} (plan steps={len(plan.steps)})")

        pixels, _, done, info = env.step(primitive)
        inv = dict(info.get("inventory", {}))
        player_pos = tuple(info.get("player_pos", (32, 32)))
        prev_action = primitive
        if done:
            print(f"  EPISODE DONE at step {step}")
            break

    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
