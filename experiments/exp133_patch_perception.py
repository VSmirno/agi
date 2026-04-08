"""exp133: Patch-based perception + optimized babble.

No CNN cosine matching. 7×7 pixel templates from experience.
Babble only when blocked (collision detection).
Every step teaches the agent something.

Pipeline: sandbox (no enemies) → real world (with enemies)
"""

from __future__ import annotations

import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

CHECKPOINT_DIR = Path("demos/checkpoints/exp133")

from snks.agent.concept_store import ConceptStore
from snks.agent.crafter_textbook import CrafterTextbook
from snks.agent.crafter_pixel_env import CrafterPixelEnv
from snks.agent.crafter_spatial_map import CrafterSpatialMap, _step_toward, MOVE_ACTIONS
from snks.agent.outcome_labeler import OutcomeLabeler
from snks.agent.patch_perception import (
    PatchStore, extract_facing_patch, detect_collision, FACING_TO_PATCH,
)
from snks.agent.perception import (
    HomeostaticTracker, select_goal, compute_curiosity,
    verify_outcome, outcome_to_verify,
)

_DIRECTIONS = ["move_up", "move_down", "move_left", "move_right"]
_STAT_GAIN_TO_NEAR = {"food": "cow", "drink": "water"}


def run_episode(
    store: ConceptStore,
    tracker: HomeostaticTracker,
    patch_store: PatchStore,
    labeler: OutcomeLabeler,
    seed: int,
    max_steps: int = 1500,
    enemies: bool = True,
    verbose: bool = False,
) -> dict:
    """Run one episode with patch-based perception."""
    env = CrafterPixelEnv(seed=seed)
    if not enemies:
        try:
            env._env._world._balance_chunk = lambda *a, **kw: None
        except Exception:
            pass

    pixels, info = env.reset()
    rng = np.random.RandomState(seed)
    spatial_map = CrafterSpatialMap()

    resources = Counter()
    grounding_events = []
    steps = 0
    current_goal = ""
    current_plan = []
    plan_step_idx = 0
    nav_steps = 0
    replan_counter = 0
    prev_inv = {}
    last_direction = "move_right"

    for step in range(max_steps):
        steps = step + 1
        inv = dict(info.get("inventory", {}))
        player_pos = info.get("player_pos", (32, 32))

        # 1. PERCEIVE — match ALL visible tiles against patch templates
        detections = patch_store.match_all_visible(pixels)
        near_label = None
        for label, row, col, diff in detections:
            # Center tiles (row 3, col 4 ± 1) = adjacent to player
            if row == 3 and col in (3, 4, 5):
                near_label = label
            elif col == 4 and row in (2, 3, 4):
                near_label = label

        near_str = near_label or "empty"

        # Update spatial map from detections
        # player_pos = [x, y]. spatial_map expects (pos[0], pos[1]) consistently.
        spatial_map.update(player_pos, near_str)
        for label, row, col, diff in detections:
            # tile grid: row=vertical(screen Y), col=horizontal(screen X)
            # player_pos[0]=world_x, player_pos[1]=world_y
            wx = int(player_pos[0]) + (col - 4)   # horizontal offset
            wy = int(player_pos[1]) + (row - 3)   # vertical offset
            spatial_map.update((wx, wy), label)

        # 1b. HOMEOSTATIC TRACKING
        if prev_inv:
            tracker.update(prev_inv, inv, {d[0] for d in detections})
            # Zombie grounding from damage
            h_before = prev_inv.get("health", 9)
            h_after = inv.get("health", 9)
            if h_after < h_before and inv.get("food", 0) > 0 and inv.get("drink", 0) > 0:
                # Damage from entity — extract facing patch as zombie
                patch = extract_facing_patch(pixels, last_direction)
                if patch is not None:
                    patch_store.add("zombie", patch)
                    grounding_events.append("damage→zombie")
                    if verbose:
                        print(f"    [{step}] DAMAGE→zombie")
        prev_inv = dict(inv)

        # 2. GOAL SELECTION
        replan_counter += 1
        needs_replan = (
            not current_plan
            or plan_step_idx >= len(current_plan)
            or (current_goal == "explore" and replan_counter >= 20)
        )
        if needs_replan:
            replan_counter = 0
            current_goal, current_plan = select_goal(
                inv, store, tracker=tracker, spatial_map=spatial_map)
            plan_step_idx = 0
            nav_steps = 0

            if verbose and step < 200 and step % 50 == 0:
                print(f"    [{step}] GOAL: {current_goal} ({len(current_plan)} steps)")

        # 3. CHOOSE ACTION
        if current_goal == "explore" or not current_plan or plan_step_idx >= len(current_plan):
            # Explore: random walk
            direction = _DIRECTIONS[rng.randint(0, 4)]
        else:
            plan_step = current_plan[plan_step_idx]

            if near_str == plan_step.target:
                # AT TARGET — execute action
                if plan_step.action == "do":
                    old_inv = dict(info.get("inventory", {}))
                    # Directional probe
                    success = False
                    for d in _DIRECTIONS:
                        pixels, _, done, info = env.step(d)
                        if done:
                            break
                        oi = dict(info.get("inventory", {}))
                        # Capture patch BEFORE "do" — object still exists
                        patch_before_do = extract_facing_patch(pixels, d)
                        pixels, _, done, info = env.step("do")
                        if done:
                            break
                        ni = dict(info.get("inventory", {}))
                        outcome = labeler.label("do", oi, ni)
                        if outcome == plan_step.target:
                            success = True
                            for k, v in ni.items():
                                delta = v - oi.get(k, 0)
                                if delta > 0 and k not in ("health","food","drink","energy"):
                                    resources[k] += delta
                            # Ground patch from BEFORE "do"
                            if patch_before_do is not None:
                                patch_store.add(plan_step.target, patch_before_do)
                            break
                        # Stat gains (food/drink from cow/water)
                        for stat, near in _STAT_GAIN_TO_NEAR.items():
                            if ni.get(stat, 0) > oi.get(stat, 0):
                                if patch_before_do is not None:
                                    patch_store.add(near, patch_before_do)
                                break
                    if done:
                        break
                    new_inv = dict(info.get("inventory", {}))
                    if success:
                        do_outcome = outcome_to_verify("do", old_inv, new_inv)
                        verify_outcome(near_str, "do", do_outcome, store)
                        # Check prereqs for next step
                        next_idx = plan_step_idx + 1
                        can_advance = True
                        if next_idx < len(current_plan):
                            nr = current_plan[next_idx].requires
                            if nr:
                                can_advance = all(new_inv.get(r,0)>=n for r,n in nr.items())
                        if can_advance:
                            plan_step_idx += 1
                        spatial_map.update(player_pos, "empty")
                    else:
                        nav_steps += 8
                        if nav_steps > 200:
                            plan_step_idx += 1
                            nav_steps = 0
                    continue

                elif plan_step.action in ("make", "place"):
                    crafter_action = f"{plan_step.action}_{plan_step.expected_gain}"
                    old_inv = dict(info.get("inventory", {}))
                    pixels, _, done, info = env.step(crafter_action)
                    if done:
                        break
                    new_inv = dict(info.get("inventory", {}))
                    craft_out = outcome_to_verify(crafter_action, old_inv, new_inv)
                    verify_outcome(near_str, plan_step.action, craft_out, store)
                    if craft_out is not None:
                        if plan_step.action == "place":
                            # Ground placed object from current view
                            patch = extract_facing_patch(pixels, last_direction)
                            if patch is not None:
                                patch_store.add(plan_step.expected_gain, patch)
                                grounding_events.append(f"place→{plan_step.expected_gain}")
                            # Immediate next craft if same location
                            if plan_step_idx + 1 < len(current_plan):
                                ns = current_plan[plan_step_idx + 1]
                                if ns.action == "make" and ns.target == plan_step.expected_gain:
                                    c2 = f"{ns.action}_{ns.expected_gain}"
                                    o2 = dict(info.get("inventory", {}))
                                    pixels, _, done, info = env.step(c2)
                                    if not done:
                                        n2 = dict(info.get("inventory", {}))
                                        co2 = outcome_to_verify(c2, o2, n2)
                                        if co2:
                                            if verbose:
                                                print(f"    [{step}] IMMEDIATE→{ns.expected_gain}")
                                            plan_step_idx += 1
                        plan_step_idx += 1
                    else:
                        nav_steps += 1
                        if nav_steps > 15:
                            plan_step_idx += 1
                            nav_steps = 0
                    continue
                else:
                    plan_step_idx += 1
                    continue
            else:
                # Navigate to target — or explore if target unknown
                known_pos = spatial_map.find_nearest(plan_step.target, player_pos)
                if known_pos:
                    direction = _step_toward(player_pos, known_pos, rng)
                else:
                    # Target not in map — explore to find it
                    # Random walk + collision learning will discover objects
                    unvisited = spatial_map.unvisited_neighbors(player_pos, radius=5)
                    if unvisited:
                        target = unvisited[rng.randint(len(unvisited))]
                        direction = _step_toward(player_pos, target, rng)
                    else:
                        direction = _DIRECTIONS[rng.randint(0, 4)]

        # 4. EXECUTE MOVE + COLLISION LEARNING
        pos_before = info.get("player_pos", (32, 32))
        pixels, _, done, info = env.step(direction)
        if done:
            break
        pos_after = info.get("player_pos", (32, 32))
        last_direction = direction

        if detect_collision(pos_before, pos_after):
            # BLOCKED — extract patch to identify what's ahead
            patch_ahead = extract_facing_patch(pixels, direction)

            # Only "do" if patch matches a known RESOURCE or is unknown
            # Don't waste "do" on grass/water/known non-interactable
            should_do = True
            if patch_ahead is not None:
                match = patch_store.match(patch_ahead)
                if match in ("water", "unknown_obstacle", "zombie"):
                    should_do = False  # can't collect, don't waste step
                elif match is not None and match not in ("tree", "stone", "coal", "iron", "cow"):
                    should_do = False  # known non-resource

            if should_do:
                old_inv = dict(info.get("inventory", {}))
                pixels, _, done, info = env.step("do")
                if done:
                    break
                new_inv = dict(info.get("inventory", {}))

                outcome = labeler.label("do", old_inv, new_inv)
                if outcome is not None and patch_ahead is not None:
                    patch_store.add(outcome, patch_ahead)
                    grounding_events.append(f"collision→{outcome}")
                    if verbose:
                        print(f"    [{step}] COLLISION→{outcome}")
                    spatial_map.update(pos_before, outcome)
                    for k, v in new_inv.items():
                        delta = v - old_inv.get(k, 0)
                        if delta > 0 and k not in ("health", "food", "drink", "energy"):
                            resources[k] += delta
                    verify_outcome(outcome, "do",
                                   outcome_to_verify("do", old_inv, new_inv), store)
                else:
                    for stat, near in _STAT_GAIN_TO_NEAR.items():
                        if new_inv.get(stat, 0) > old_inv.get(stat, 0):
                            if patch_ahead is not None:
                                patch_store.add(near, patch_ahead)
                                grounding_events.append(f"collision→{near}")
                            break
                    else:
                        if patch_ahead is not None:
                            # Tried "do", nothing happened — record as obstacle
                            patch_store.add("unknown_obstacle", patch_ahead)
            else:
                # Known non-resource — just record in map, don't "do"
                if patch_ahead is not None:
                    match = patch_store.match(patch_ahead)
                    if match:
                        wx, wy = int(pos_before[0]), int(pos_before[1])
                        ddx, ddy = {"move_up":(0,-1),"move_down":(0,1),"move_left":(-1,0),"move_right":(1,0)}[direction]
                        spatial_map.update((wx+ddx, wy+ddy), match)

        nav_steps += 1
        if nav_steps > 200:
            plan_step_idx += 1
            nav_steps = 0

    # Death cause
    final = dict(info.get("inventory", {}))
    health = final.get("health", 0)
    food = final.get("food", 0)
    drink = final.get("drink", 0)
    death_cause = "timeout"
    if steps < max_steps:
        if food == 0 or drink == 0:
            death_cause = "starvation"
        else:
            death_cause = "zombie"

    return {
        "length": steps,
        "resources": dict(resources),
        "grounding_events": grounding_events,
        "death_cause": death_cause,
        "templates": list(patch_store.templates.keys()),
        "map_visited": spatial_map.n_visited,
    }


def main():
    print("=" * 60)
    print("exp133: Patch Perception + Collision Babble")
    print("=" * 60)
    t_start = time.time()

    store = ConceptStore()
    tb = CrafterTextbook("configs/crafter_textbook.yaml")
    tb.load_into(store)
    tracker = HomeostaticTracker()
    tracker.init_from_body_rules(tb.body_rules)
    patch_store = PatchStore()
    labeler = OutcomeLabeler()

    # SANDBOX: learn the world
    print("\nSANDBOX: 200 episodes, no enemies...")
    for i in range(200):
        result = run_episode(
            store, tracker, patch_store, labeler,
            40000 + i * 7, max_steps=2000, enemies=False,
            verbose=(i < 3))
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/200] templates={list(patch_store.templates.keys())} "
                  f"wood={result['resources'].get('wood',0)} "
                  f"length={result['length']}")

    print(f"  Templates learned: {list(patch_store.templates.keys())}")
    print(f"  Template counts: {dict((k, t.count) for k, t in patch_store.templates.items())}")

    # REAL WORLD: survival
    print(f"\n{'='*60}")
    print("REAL WORLD: 500 episodes, enemies ON")
    print(f"{'='*60}")
    lengths = []
    death_causes = Counter()
    sword_count = 0
    for i in range(500):
        result = run_episode(
            store, tracker, patch_store, labeler,
            90000 + i * 7, max_steps=1500, enemies=True,
            verbose=(i < 5))
        lengths.append(result["length"])
        death_causes[result["death_cause"]] += 1
        if any("sword" in e for e in result.get("grounding_events", [])):
            sword_count += 1
        if (i + 1) % 50 == 0:
            last50 = lengths[-50:]
            print(f"  [{i+1}/500] mean={np.mean(lengths):.0f} last50={np.mean(last50):.0f} "
                  f"deaths={{z:{death_causes.get('zombie',0)},s:{death_causes.get('starvation',0)}}} "
                  f"sword={sword_count}/{i+1} templates={len(patch_store.templates)}")

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"exp133 SUMMARY ({elapsed:.0f}s)")
    print(f"{'='*60}")
    print(f"  Survival: {np.mean(lengths):.0f} steps (gate: ≥200)")
    print(f"  Templates: {list(patch_store.templates.keys())}")
    print(f"  Deaths: {dict(death_causes)}")
    print(f"  Sword: {sword_count}/500")
    print(f"  Gate: {'PASS' if np.mean(lengths) >= 200 else 'FAIL'}")


if __name__ == "__main__":
    main()
