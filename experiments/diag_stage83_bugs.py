"""Stage 83 — Bug A & Bug B diagnostic.

Bug A (save/load SDM addressing):
  Creates model1, learns "tree"+"do"→wood, saves. Creates model2 (different seed),
  loads gen1. Measures predict confidence before vs after load and prints
  Hamming distance between write address and read address in model2.

Bug B (entropy collapse):
  Runs 1 episode, dumps action_counts every 25 steps. When last-20-step
  entropy < 0.5 triggers, prints find_nearest result for the dominant plan
  target, spatial_map state, and plan details.

Run on minipc:
  HSA_OVERRIDE_GFX_VERSION=11.0.0 PYTHONPATH=src python experiments/diag_stage83_bugs.py
"""

from __future__ import annotations

import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch

torch.backends.cudnn.enabled = False

from snks.agent.vector_world_model import (
    VectorWorldModel, hamming_similarity, bind, bundle,
    random_bitvector,
)
from snks.agent.vector_bootstrap import load_from_textbook


# ---------------------------------------------------------------------------
# Part 1 — Bug A: synthetic save/load test
# ---------------------------------------------------------------------------

def diag_bug_a():
    print("\n" + "=" * 60)
    print("BUG A — synthetic save/load addressing test")
    print("=" * 60)

    # --- Model 1: learn a simple association ---
    print("\n[Model1] seed=42, learning tree+do→wood")
    m1 = VectorWorldModel(dim=4096, n_locations=10000, seed=42)
    m1._ensure_concept("tree")
    m1._ensure_action("do")
    m1.learn("tree", "do", {"wood": 1})

    eff1, conf1 = m1.predict("tree", "do")
    print(f"  predict(tree, do): conf={conf1:.4f}  (should be > 0.01)")

    addr1 = bind(m1.concepts["tree"], m1.actions["do"])
    print(f"  write address bit-1 fraction: {addr1.mean().item():.3f}")
    print(f"  SDM n_writes: {m1.memory.n_writes}")

    # --- Save ---
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        save_path = f.name
    m1.save(save_path)
    print(f"  Saved to {save_path}")

    # --- Model 2: different seed, load gen1 experience ---
    print("\n[Model2] seed=1042 (DIFFERENT seed), loading gen1")
    m2 = VectorWorldModel(dim=4096, n_locations=10000, seed=1042)
    m2._ensure_concept("tree")
    m2._ensure_action("do")

    # Record state before load
    v_tree_before = m2.concepts["tree"].clone()
    v_do_before = m2.actions["do"].clone()
    addr_before = bind(v_tree_before, v_do_before)

    print(f"  [before load] predict(tree, do): ", end="")
    _, conf_before = m2.predict("tree", "do")
    print(f"conf={conf_before:.4f}")

    m2.load(save_path)

    # Record state after load
    v_tree_after = m2.concepts["tree"]
    v_do_after = m2.actions["do"]
    addr_after = bind(v_tree_after, v_do_after)

    print(f"\n  [after load]")
    print(f"  tree vec changed: {not torch.equal(v_tree_before, v_tree_after)}")
    print(f"  do   vec changed: {not torch.equal(v_do_before,   v_do_after)}")

    sim_tree = hamming_similarity(v_tree_before, v_tree_after)
    sim_do   = hamming_similarity(v_do_before,   v_do_after)
    print(f"  tree similarity (m2_before vs m2_after): {sim_tree:.4f}")
    print(f"  do   similarity (m2_before vs m2_after): {sim_do:.4f}")

    # The write address (from model1) vs the new read address (model2 after load)
    sim_addr = hamming_similarity(addr1, addr_after)
    print(f"\n  Hamming sim(write_addr_m1, read_addr_m2): {sim_addr:.4f}")
    print(f"    (0.5 = random noise, 1.0 = identical)")

    eff2, conf2 = m2.predict("tree", "do")
    print(f"\n  [after load] predict(tree, do): conf={conf2:.4f}")

    # Verdict
    print("\n  === VERDICT ===")
    if conf1 > 0.01 and conf2 < 0.01:
        print("  BUG A CONFIRMED: confidence drops to ~0 after load")
        print("  Root cause: concept vec changed by bundle-merge, action vec unchanged")
        print(f"    addr similarity={sim_addr:.4f} — query misses the written locations")
    elif conf2 > 0.01:
        print(f"  BUG A NOT reproduced at this scale (conf2={conf2:.4f})")
        print("  Try with full dim=16384 or check if tree concept was seeded same way")
    else:
        print(f"  conf1={conf1:.4f} also low — learning itself may not work at dim=4096")

    Path(save_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Part 2 — Bug B: entropy collapse trace (1 episode, full debug)
# ---------------------------------------------------------------------------

def diag_bug_b():
    print("\n" + "=" * 60)
    print("BUG B — entropy collapse trace (1 episode)")
    print("=" * 60)

    import numpy as np

    from snks.agent.vector_world_model import VectorWorldModel
    from snks.agent.vector_bootstrap import load_from_textbook
    from snks.agent.vector_sim import (
        VectorState, simulate_forward, score_trajectory,
    )
    from snks.agent.vector_mpc_agent import (
        generate_candidate_plans, expand_to_primitive,
        _update_spatial_map, DynamicEntityTracker,
        build_prediction_cache,
    )
    from snks.agent.perception import HomeostaticTracker, perceive_tile_field
    from snks.agent.crafter_spatial_map import CrafterSpatialMap
    from snks.agent.crafter_textbook import CrafterTextbook
    from snks.encoder.tile_segmenter import load_tile_segmenter, pick_device
    from snks.agent.crafter_pixel_env import CrafterPixelEnv

    device = torch.device(pick_device())
    print(f"device={device}")

    checkpoint = Path("demos/checkpoints/exp135/segmenter_9x9.pt")
    segmenter = load_tile_segmenter(str(checkpoint), device=device)

    model = VectorWorldModel(dim=16384, n_locations=50000, seed=42, device=device)
    textbook_path = Path("configs/crafter_textbook.yaml")
    stats = load_from_textbook(model, textbook_path)
    print(f"Bootstrap: {stats}")

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
    prev_move = None
    prev_plan_target = None
    prev_player_pos = None

    action_counts: Counter = Counter()
    recent_actions: list[str] = []   # last 20 primitives for sliding entropy
    collapse_reported = False

    def sliding_entropy(actions: list[str]) -> float:
        if not actions:
            return 0.0
        cnt = Counter(actions)
        total = len(actions)
        H = 0.0
        for c in cnt.values():
            p = c / total
            H -= p * np.log2(p)
        return H

    print("\n=== EPISODE START ===\n")

    for step in range(1000):
        inv = dict(info.get("inventory", {}))
        body = {v: float(info.get(v, 9.0)) for v in vital_vars}
        player_pos = tuple(info.get("player_pos", (32, 32)))

        # Blocked detection
        if (prev_action and prev_action.startswith("move_")
                and prev_player_pos is not None
                and prev_player_pos == player_pos):
            dx, dy = 0, 0
            if prev_action == "move_right":  dx = 1
            elif prev_action == "move_left": dx = -1
            elif prev_action == "move_down": dy = 1
            elif prev_action == "move_up":   dy = -1
            spatial_map.mark_blocked((player_pos[0] + dx, player_pos[1] + dy))

        # Perception
        vf = perceive_tile_field(pixels, segmenter)
        _update_spatial_map(spatial_map, vf, player_pos)
        entity_tracker.update(vf, player_pos)

        if prev_inv is not None:
            tracker.update(prev_inv, inv, vf.visible_concepts())

        # Surprise-driven learning
        if prev_inv is not None and prev_body is not None and prev_action is not None:
            inv_d = {k: inv.get(k, 0) - prev_inv.get(k, 0)
                     for k in set(inv) | set(prev_inv)
                     if inv.get(k, 0) != prev_inv.get(k, 0)}
            body_d = {k: int(round(body.get(k, 0) - prev_body.get(k, 0)))
                      for k in vital_vars
                      if abs(body.get(k, 0) - prev_body.get(k, 0)) > 0.01}
            all_d = {**inv_d, **body_d}
            if all_d and prev_action in ("do", "place", "make") and prev_plan_target:
                if prev_plan_target not in ("empty", "self"):
                    surprise = model.learn(prev_plan_target, prev_action, all_d)

        # State + planning
        state = VectorState(
            inventory=inv, body=body, player_pos=player_pos,
            step=step, last_action=prev_action, spatial_map=spatial_map,
        )
        known_step = set(vf.visible_concepts()) | set(spatial_map.known_objects.keys())
        target_acts = [a for a in model.actions if a in ("do", "make", "place")]
        step_cache = build_prediction_cache(model, known_step, target_acts)

        candidates = generate_candidate_plans(
            model, state, spatial_map, vf.visible_concepts(),
            beam_width=5, max_depth=3, cache=step_cache,
        )

        def _plan_dist(plan):
            if not plan.steps:
                return 9999
            first = plan.steps[0]
            if first.target == "self":
                return 0
            pos = spatial_map.find_nearest(first.target, player_pos)
            if pos is None:
                return 9999
            return abs(pos[0] - player_pos[0]) + abs(pos[1] - player_pos[1])

        candidates.sort(key=_plan_dist)

        scored = []
        for plan in candidates:
            traj = simulate_forward(model, plan, state, 10, vital_vars, cache=step_cache)
            sc = score_trajectory(traj, vital_vars)
            scored.append((sc, plan, traj))
        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best_plan, best_traj = scored[0]

        # Primitive
        if best_plan.steps:
            primitive = expand_to_primitive(
                best_plan.steps[0], player_pos, spatial_map, model, rng,
                last_action=prev_move,
            )
        else:
            moves = [a for a in model.actions if a.startswith("move_")]
            primitive = str(rng.choice(moves)) if moves else "move_right"

        action_counts[primitive] += 1
        recent_actions.append(primitive)
        if len(recent_actions) > 20:
            recent_actions.pop(0)

        H_recent = sliding_entropy(recent_actions)

        # --- Per-25-step summary ---
        if step % 25 == 0:
            total = sum(action_counts.values())
            H_full = 0.0
            for c in action_counts.values():
                p = c / total
                H_full -= p * np.log2(p)
            print(
                f"s{step:4d}  H{body.get('health',0):.0f}"
                f" F{body.get('food',0):.0f}"
                f" D{body.get('drink',0):.0f}"
                f"  near={vf.near_concept:10s}"
                f"  plan={best_plan.origin[:30]:30s}"
                f"  prim={primitive:14s}"
                f"  H_full={H_full:.2f}  H_20={H_recent:.2f}"
                f"  inv={dict(inv)}"
            )

        # --- Collapse detection ---
        if H_recent < 0.5 and len(recent_actions) >= 20 and not collapse_reported:
            collapse_reported = True
            print(f"\n!!! COLLAPSE DETECTED at step {step} (H_20={H_recent:.3f}) !!!")
            print(f"  recent actions: {dict(Counter(recent_actions))}")
            print(f"  player_pos: {player_pos}")
            print(f"  near_concept: {vf.near_concept}")
            print(f"  best_plan: {best_plan.origin}")
            if best_plan.steps:
                first_step = best_plan.steps[0]
                target = first_step.target
                action = first_step.action
                print(f"  first step: {target}:{action}")
                nearest_pos = spatial_map.find_nearest(target, player_pos)
                print(f"  find_nearest({target}, {player_pos}) = {nearest_pos}")
                if nearest_pos:
                    dist = abs(nearest_pos[0] - player_pos[0]) + abs(nearest_pos[1] - player_pos[1])
                    print(f"  manhattan dist to target: {dist}")
                    blocked = spatial_map.is_blocked(nearest_pos)
                    print(f"  target pos blocked: {blocked}")
                # Show all known objects in spatial map
                known = spatial_map.known_objects
                print(f"  spatial_map.known_objects: {known}")
            print(f"  top-5 plans:")
            for i, (sc, pl, _) in enumerate(scored[:5]):
                steps_str = [f"{s.target}:{s.action}" for s in pl.steps]
                print(f"    #{i}: score={sc} plan={pl.origin} steps={steps_str}")

        # State update
        prev_inv = dict(inv)
        prev_body = dict(body)
        prev_action = primitive
        if primitive.startswith("move_"):
            prev_move = primitive
        prev_plan_target = best_plan.steps[0].target if best_plan.steps else None
        prev_player_pos = player_pos

        pixels, _r, done, info = env.step(primitive)

        # Bug 6: clear facing tile on gather
        new_inv = dict(info.get("inventory", {}))
        inv_changed_diag = False
        for item_key in model.roles:
            if item_key.startswith("__"):
                continue
            if new_inv.get(item_key, 0) > inv.get(item_key, 0) and primitive == "do":
                dx, dy = 0, 0
                if prev_move == "move_right":  dx = 1
                elif prev_move == "move_left": dx = -1
                elif prev_move == "move_down": dy = 1
                elif prev_move == "move_up":   dy = -1
                spatial_map.update((player_pos[0] + dx, player_pos[1] + dy), "empty")
                inv_changed_diag = True
                break

        # Bug 6b: frustrated do — clear stale resource entries
        if primitive == "do" and not inv_changed_diag:
            dx, dy = 0, 0
            if prev_move == "move_right":  dx = 1
            elif prev_move == "move_left": dx = -1
            elif prev_move == "move_down": dy = 1
            elif prev_move == "move_up":   dy = -1
            facing_tile = (player_pos[0] + dx, player_pos[1] + dy)
            if facing_tile != player_pos:
                spatial_map.update(facing_tile, "empty", 1.0)

        if done:
            final = {v: float(info.get(v, 0)) for v in vital_vars}
            print(f"\n=== DIED step={step+1}, body={final}, inv={info.get('inventory',{})} ===")
            break

    # --- Final summary ---
    total = sum(action_counts.values())
    H_final = 0.0
    for c in action_counts.values():
        p = c / total
        H_final -= p * np.log2(p)
    print(f"\n=== ACTION COUNTS (total={total}, entropy={H_final:.3f}) ===")
    for act, cnt in sorted(action_counts.items(), key=lambda x: -x[1]):
        print(f"  {act:20s}: {cnt:5d}  ({100*cnt/total:.1f}%)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Stage 83 Bug Diagnostic")
    print("=" * 60)

    # Part 1: Bug A (no GPU needed, uses CPU)
    diag_bug_a()

    # Part 2: Bug B (needs segmenter + Crafter)
    print("\n\nProceeding to Bug B trace (needs segmenter)...")
    try:
        diag_bug_b()
    except FileNotFoundError as e:
        print(f"SKIP Bug B: {e}")
    except ImportError as e:
        print(f"SKIP Bug B: {e}")
