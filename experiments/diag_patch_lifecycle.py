"""Lifecycle diagnostic for patch perception: where does time go?"""

import numpy as np
from collections import Counter

from snks.agent.concept_store import ConceptStore
from snks.agent.crafter_textbook import CrafterTextbook
from snks.agent.crafter_pixel_env import CrafterPixelEnv
from snks.agent.crafter_spatial_map import CrafterSpatialMap, _step_toward, MOVE_ACTIONS
from snks.agent.outcome_labeler import OutcomeLabeler
from snks.agent.patch_perception import (
    PatchStore, extract_facing_patch, detect_collision,
)
from snks.agent.perception import (
    HomeostaticTracker, select_goal,
)

_DIRECTIONS = ["move_up", "move_down", "move_left", "move_right"]
_STAT_GAIN = {"food": "cow", "drink": "water"}


def trace_episode(store, tracker, patch_store, labeler, seed, max_steps=500):
    env = CrafterPixelEnv(seed=seed)
    pixels, info = env.reset()
    rng = np.random.RandomState(seed)
    spatial_map = CrafterSpatialMap()

    milestones = {}
    action_log = Counter()
    wood = 0
    prev_inv = {}
    current_goal = ""
    current_plan = []
    plan_step_idx = 0
    replan_counter = 0

    for step in range(max_steps):
        inv = dict(info.get("inventory", {}))
        pp = info.get("player_pos", (32, 32))

        # Perceive
        detections = patch_store.match_all_visible(pixels)
        near_str = "empty"
        for label, row, col, diff in detections:
            if (row == 3 and col in (3, 4, 5)) or (col == 4 and row in (2, 3, 4)):
                near_str = label
        spatial_map.update(pp, near_str)
        for label, row, col, diff in detections:
            py, px = int(pp[0]), int(pp[1])
            spatial_map.update((py + (col - 4), px + (row - 3)), label)

        if prev_inv:
            tracker.update(prev_inv, inv, {d[0] for d in detections})
            # Damage
            if inv.get("health", 9) < prev_inv.get("health", 9):
                if inv.get("food", 0) > 0 and inv.get("drink", 0) > 0:
                    if "first_damage" not in milestones:
                        milestones["first_damage"] = step

        # Wood tracking
        w = inv.get("wood", 0)
        wp = prev_inv.get("wood", 0)
        if w > wp:
            wood += (w - wp)
            milestones[f"wood_{wood}"] = step

        if inv.get("wood_sword", 0) > prev_inv.get("wood_sword", 0):
            milestones["sword"] = step
        prev_inv = dict(inv)

        # Goal
        replan_counter += 1
        if not current_plan or plan_step_idx >= len(current_plan) or \
           (current_goal == "explore" and replan_counter >= 20):
            replan_counter = 0
            current_goal, current_plan = select_goal(
                inv, store, tracker=tracker, spatial_map=spatial_map)
            plan_step_idx = 0

        # What is agent DOING?
        if current_goal == "explore":
            action_type = "explore"
        elif current_goal == "kill_zombie":
            action_type = "craft_sword"
        elif current_goal.startswith("restore_"):
            action_type = "survival"
        else:
            action_type = "other"

        # Execute
        if current_goal == "explore" or not current_plan or plan_step_idx >= len(current_plan):
            d = _DIRECTIONS[rng.randint(0, 4)]
            pos_b = info.get("player_pos", (32, 32))
            pixels, _, done, info = env.step(d)
            if done: break
            pos_a = info.get("player_pos", (32, 32))

            if detect_collision(pos_b, pos_a):
                patch = extract_facing_patch(pixels, d)
                if patch is not None:
                    match = patch_store.match(patch)
                    if match in ("tree",) or match is None:
                        oi = dict(info.get("inventory", {}))
                        pixels, _, done, info = env.step("do")
                        if done: break
                        ni = dict(info.get("inventory", {}))
                        outcome = labeler.label("do", oi, ni)
                        if outcome and patch is not None:
                            patch_store.add(outcome, patch)
                        action_log["collision_do"] += 2
                    else:
                        action_log["collision_skip"] += 1
                else:
                    action_log["collision_skip"] += 1
            else:
                action_log["move"] += 1
        else:
            ps = current_plan[plan_step_idx]
            if near_str == ps.target:
                if ps.action == "do":
                    oi = dict(info.get("inventory", {}))
                    success = False
                    for d in _DIRECTIONS:
                        pixels, _, done, info = env.step(d)
                        if done: break
                        oi2 = dict(info.get("inventory", {}))
                        pixels, _, done, info = env.step("do")
                        if done: break
                        ni2 = dict(info.get("inventory", {}))
                        if labeler.label("do", oi2, ni2) == ps.target:
                            success = True
                            patch = extract_facing_patch(pixels, d)
                            if patch is not None:
                                patch_store.add(ps.target, patch)
                            break
                    if done: break
                    ni = dict(info.get("inventory", {}))
                    if success:
                        nxt = plan_step_idx + 1
                        ok = True
                        if nxt < len(current_plan) and current_plan[nxt].requires:
                            ok = all(ni.get(r,0)>=n for r,n in current_plan[nxt].requires.items())
                        if ok:
                            plan_step_idx += 1
                    action_log["plan_probe"] += 8
                elif ps.action in ("make", "place"):
                    ca = f"{ps.action}_{ps.expected_gain}"
                    pixels, _, done, info = env.step(ca)
                    if done: break
                    action_log["plan_craft"] += 1
                    plan_step_idx += 1
                else:
                    plan_step_idx += 1
            else:
                known = spatial_map.find_nearest(ps.target, pp)
                if known:
                    d = _step_toward(pp, known, rng)
                else:
                    d = _DIRECTIONS[rng.randint(0, 4)]
                pixels, _, done, info = env.step(d)
                if done: break
                action_log["plan_nav"] += 1

    death = "starvation" if inv.get("food",0)==0 or inv.get("drink",0)==0 else "zombie"
    if step+1 >= max_steps: death = "timeout"
    milestones["death"] = (death, step+1)
    return {"milestones": milestones, "actions": dict(action_log), "steps": step+1}


def main():
    print("PATCH PERCEPTION LIFECYCLE DIAGNOSTIC")
    print("=" * 60)

    store = ConceptStore()
    tb = CrafterTextbook("configs/crafter_textbook.yaml")
    tb.load_into(store)
    tracker = HomeostaticTracker()
    tracker.init_from_body_rules(tb.body_rules)
    patch_store = PatchStore()
    labeler = OutcomeLabeler()

    # Bootstrap
    print("Bootstrap 20 episodes (no enemies)...")
    for i in range(20):
        env = CrafterPixelEnv(seed=40000+i*7)
        try:
            env._env._world._balance_chunk = lambda *a,**kw: None
        except: pass
        pixels, info = env.reset()
        rng = np.random.RandomState(40000+i*7)
        for s in range(300):
            d = _DIRECTIONS[rng.randint(0,4)]
            pb = info['player_pos'].copy()
            pixels,_,done,info = env.step(d)
            if done: break
            pa = info['player_pos']
            if detect_collision(pb, pa):
                patch = extract_facing_patch(pixels, d)
                oi = dict(info.get('inventory',{}))
                pixels,_,done,info = env.step('do')
                if done: break
                ni = dict(info.get('inventory',{}))
                outcome = labeler.label('do', oi, ni)
                if outcome and patch is not None:
                    patch_store.add(outcome, patch)
                for stat, near in _STAT_GAIN.items():
                    if ni.get(stat,0) > oi.get(stat,0) and patch is not None:
                        patch_store.add(near, patch)
    print(f"Templates: {list(patch_store.templates.keys())}")

    # Trace 50 episodes with enemies
    print(f"\nTracing 50 episodes (enemies ON)...")
    all_actions = Counter()
    milestone_times = {}
    deaths = Counter()

    for i in range(50):
        r = trace_episode(store, tracker, patch_store, labeler, 90000+i*7, max_steps=500)
        for k, v in r["actions"].items():
            all_actions[k] += v
        for k, v in r["milestones"].items():
            if k == "death":
                deaths[v[0]] += 1
                continue
            if k not in milestone_times:
                milestone_times[k] = []
            milestone_times[k].append(v)

    total = sum(all_actions.values())
    print(f"\nTIME ALLOCATION ({total} total steps):")
    for k, v in sorted(all_actions.items(), key=lambda x:-x[1]):
        print(f"  {k}: {v} ({v/total*100:.1f}%)")

    print(f"\nMILESTONES:")
    for k in sorted(milestone_times.keys()):
        t = milestone_times[k]
        print(f"  {k}: median={np.median(t):.0f} mean={np.mean(t):.0f} "
              f"hit={len(t)}/50 ({len(t)/50*100:.0f}%)")

    print(f"\nDEATHS: {dict(deaths)}")
    print(f"Mean lifetime: {np.mean([v for d,v in [r['milestones']['death'] for r in [trace_episode(store,tracker,patch_store,labeler,90000+i*7) for i in range(5)]]]):.0f}")


if __name__ == "__main__":
    main()
