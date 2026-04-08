"""Diagnostic: trace agent lifecycle step by step.

For each episode, track:
- When was each concept first grounded?
- When did agent first collect wood? 2nd? 3rd?
- When did agent place table?
- When did agent craft sword?
- When did agent first take damage?
- When did agent die? From what?
- What was agent DOING each step? (explore/navigate/babble/plan)

This reveals WHERE time is spent.
"""

import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from snks.encoder.cnn_encoder import CNNEncoder, disable_rocm_conv
from snks.agent.concept_store import ConceptStore
from snks.agent.crafter_textbook import CrafterTextbook
from snks.agent.crafter_pixel_env import CrafterPixelEnv
from snks.agent.crafter_spatial_map import CrafterSpatialMap, _step_toward, MOVE_ACTIONS
from snks.agent.outcome_labeler import OutcomeLabeler
from snks.agent.perception import (
    perceive_field, HomeostaticTracker, VisualField,
    on_action_outcome, select_goal, explore_action,
    ground_empty_on_start, ground_zombie_on_damage,
    verify_outcome, outcome_to_verify,
)

_DIRECTIONS = ["move_up", "move_down", "move_left", "move_right"]


def trace_episode(encoder, store, labeler, tracker, seed, max_steps=1500):
    """Run one episode and trace every milestone."""
    env = CrafterPixelEnv(seed=seed)
    pixels, info = env.reset()
    rng = np.random.RandomState(seed)
    spatial_map = CrafterSpatialMap()
    device = next(encoder.parameters()).device

    pix_t = torch.from_numpy(pixels).float()
    if device.type != "cpu":
        pix_t = pix_t.to(device)
    ground_empty_on_start(pix_t, encoder, store)

    milestones = {}
    action_counts = Counter()  # what agent spent time on
    wood_collected = 0
    prev_inv = {}
    current_goal = ""
    current_plan = []
    plan_step_idx = 0
    nav_steps = 0
    replan_counter = 0

    for step in range(max_steps):
        inv = dict(info.get("inventory", {}))

        # Perceive
        pix_tensor = torch.from_numpy(pixels).float()
        if device.type != "cpu":
            pix_tensor = pix_tensor.to(device)
        vf = perceive_field(pix_tensor, encoder, store)
        near_str = vf.near_concept
        player_pos = info.get("player_pos", (32, 32))
        spatial_map.update(player_pos, near_str)

        # Track damage
        if prev_inv:
            ground_zombie_on_damage(prev_inv, inv, vf, store)
            tracker.update(prev_inv, inv, vf.visible_concepts())

            h_before = prev_inv.get("health", 9)
            h_after = inv.get("health", 9)
            if h_after < h_before and "first_damage" not in milestones:
                milestones["first_damage"] = step

        # Track wood
        wood_now = inv.get("wood", 0)
        wood_prev = prev_inv.get("wood", 0)
        if wood_now > wood_prev:
            wood_collected += (wood_now - wood_prev)
            if f"wood_{wood_collected}" not in milestones:
                milestones[f"wood_{wood_collected}"] = step

        # Track crafts
        if inv.get("wood_sword", 0) > prev_inv.get("wood_sword", 0):
            milestones["sword_crafted"] = step
        if inv.get("wood_pickaxe", 0) > prev_inv.get("wood_pickaxe", 0):
            milestones["pickaxe_crafted"] = step

        prev_inv = dict(inv)

        # Goal selection
        replan_counter += 1
        needs_replan = (
            not current_plan
            or plan_step_idx >= len(current_plan)
            or (current_goal == "explore" and replan_counter >= 20)
        )
        if needs_replan:
            replan_counter = 0
            current_goal, current_plan = select_goal(
                inv, store, tracker=tracker, visual_field=vf, spatial_map=spatial_map)
            plan_step_idx = 0
            nav_steps = 0

        # Track what agent is doing
        if current_goal == "explore":
            action_type = "explore"
        elif current_goal == "kill_zombie":
            action_type = "craft_sword"
        elif current_goal.startswith("restore_"):
            action_type = "survival"
        else:
            action_type = "other"

        # Execute (simplified — just step to measure time allocation)
        if not current_plan or plan_step_idx >= len(current_plan):
            # Explore
            act = explore_action(rng, store, inventory=inv)
            if act == "babble_do":
                direction = _DIRECTIONS[rng.randint(0, 4)]
                pixels, _, done, info = env.step(direction)
                if done:
                    break
                cf_b = None
                if encoder:
                    pt = torch.from_numpy(pixels).float()
                    if device.type != "cpu":
                        pt = pt.to(device)
                    vf_b = perceive_field(pt, encoder, store)
                    cf_b = vf_b.center_feature
                old_b = dict(info.get("inventory", {}))
                pixels, _, done, info = env.step("do")
                if done:
                    break
                new_b = dict(info.get("inventory", {}))
                if cf_b is not None:
                    on_action_outcome("do", old_b, new_b, cf_b, store, labeler)
                action_counts["babble"] += 2
            elif act.startswith("babble_"):
                craft = act.replace("babble_", "")
                old_b = dict(info.get("inventory", {}))
                pixels, _, done, info = env.step(craft)
                if done:
                    break
                action_counts["craft_babble"] += 1
            else:
                pixels, _, done, info = env.step(act)
                if done:
                    break
                action_counts["move"] += 1
        else:
            plan_step = current_plan[plan_step_idx]

            if near_str == plan_step.target:
                if plan_step.action == "do":
                    # Probe
                    old_inv = dict(info.get("inventory", {}))
                    success = False
                    for d in _DIRECTIONS:
                        pixels, _, done, info = env.step(d)
                        if done:
                            break
                        oi = dict(info.get("inventory", {}))
                        pixels, _, done, info = env.step("do")
                        if done:
                            break
                        ni = dict(info.get("inventory", {}))
                        label = labeler.label("do", oi, ni)
                        if label == plan_step.target:
                            success = True
                            break
                    if done:
                        break
                    new_inv = dict(info.get("inventory", {}))
                    if success:
                        next_idx = plan_step_idx + 1
                        can_advance = True
                        if next_idx < len(current_plan):
                            nr = current_plan[next_idx].requires
                            if nr:
                                can_advance = all(new_inv.get(r, 0) >= n for r, n in nr.items())
                        if can_advance:
                            plan_step_idx += 1
                        verify_outcome(near_str, "do", plan_step.expected_gain, store)
                    action_counts["probe"] += 8
                elif plan_step.action in ("make", "place"):
                    craft_act = f"{plan_step.action}_{plan_step.expected_gain}"
                    old_inv = dict(info.get("inventory", {}))
                    pixels, _, done, info = env.step(craft_act)
                    if done:
                        break
                    new_inv = dict(info.get("inventory", {}))
                    craft_out = outcome_to_verify(craft_act, old_inv, new_inv)
                    if craft_out is not None:
                        if plan_step.action == "place":
                            # Ground result + immediate next craft
                            pt = torch.from_numpy(pixels).float()
                            if device.type != "cpu":
                                pt = pt.to(device)
                            _, z_a = perceive_field(pt, encoder, store), None
                            vf_a = perceive_field(pt, encoder, store)
                            rc = store.query_text(plan_step.expected_gain)
                            if rc and rc.visual is None and vf_a.center_feature is not None:
                                from torch.nn.functional import normalize
                                store.ground_visual(plan_step.expected_gain,
                                    normalize(vf_a.center_feature.unsqueeze(0), dim=1).squeeze(0))
                            # Immediate next craft
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
                                            plan_step_idx += 1
                        plan_step_idx += 1
                    action_counts["craft"] += 1
                else:
                    plan_step_idx += 1
            else:
                # Navigate
                known = spatial_map.find_nearest(plan_step.target, player_pos)
                if known:
                    act = _step_toward(player_pos, known, rng)
                else:
                    act = explore_action(rng, store, inventory=inv)
                    if act.startswith("babble"):
                        direction = _DIRECTIONS[rng.randint(0, 4)]
                        pixels, _, done, info = env.step(direction)
                        if not done:
                            pixels, _, done, info = env.step("do")
                        if done:
                            break
                        action_counts["babble"] += 2
                        nav_steps += 2
                        continue
                pixels, _, done, info = env.step(act)
                if done:
                    break
                action_counts["navigate"] += 1
                nav_steps += 1

        if nav_steps > 200:
            plan_step_idx += 1
            nav_steps = 0

    # Death cause
    final = dict(info.get("inventory", {}))
    health = final.get("health", 0)
    food = final.get("food", 0)
    drink = final.get("drink", 0)
    if step + 1 < max_steps:
        if food == 0 or drink == 0:
            milestones["death"] = ("starvation", step + 1)
        else:
            milestones["death"] = ("zombie", step + 1)
    else:
        milestones["death"] = ("timeout", step + 1)

    return {
        "milestones": milestones,
        "action_counts": dict(action_counts),
        "total_steps": step + 1,
    }


def main():
    disable_rocm_conv()
    print("=" * 60)
    print("LIFECYCLE DIAGNOSTIC")
    print("=" * 60)

    # Load best encoder
    for ckpt, ch in [(Path("demos/checkpoints/exp132"), 512),
                     (Path("demos/checkpoints/exp128"), 256)]:
        for tag in ["final", "phase0", "phase3"]:
            path = ckpt / tag / "encoder.pt"
            if path.exists():
                encoder = CNNEncoder(feature_channels=ch)
                try:
                    encoder.load_state_dict(torch.load(path, weights_only=True))
                except RuntimeError:
                    continue
                encoder.eval()
                if torch.cuda.is_available():
                    encoder = encoder.cuda()
                print(f"Loaded {ch}ch encoder from {path}")
                break
        else:
            continue
        break

    store = ConceptStore()
    tb = CrafterTextbook("configs/crafter_textbook.yaml")
    tb.load_into(store)
    tracker = HomeostaticTracker()
    tracker.init_from_body_rules(tb.body_rules)
    labeler = OutcomeLabeler()

    # Bootstrap: 10 episodes
    print("\nBootstrap (10 episodes)...")
    for i in range(10):
        trace_episode(encoder, store, labeler, tracker, 60000 + i * 7)
    grounded = [c.id for c in store.concepts.values() if c.visual is not None]
    print(f"Grounded: {grounded}")

    # Trace 50 episodes with detailed output
    print(f"\nTracing 50 episodes...")
    all_milestones = []
    all_actions = Counter()

    for i in range(50):
        result = trace_episode(encoder, store, labeler, tracker, 90000 + i * 7)
        all_milestones.append(result["milestones"])
        for k, v in result["action_counts"].items():
            all_actions[k] += v

    # Analyze
    print(f"\n{'='*60}")
    print("TIME ALLOCATION (total steps across 50 episodes):")
    total = sum(all_actions.values())
    for action, count in sorted(all_actions.items(), key=lambda x: -x[1]):
        print(f"  {action}: {count} ({count/total*100:.1f}%)")

    print(f"\nMILESTONE TIMING (median step when event happens):")
    milestone_times: dict[str, list[int]] = {}
    for ms in all_milestones:
        for k, v in ms.items():
            if k == "death":
                continue
            if k not in milestone_times:
                milestone_times[k] = []
            milestone_times[k].append(v)

    for k in sorted(milestone_times.keys()):
        times = milestone_times[k]
        hit_rate = len(times) / 50
        print(f"  {k}: median={np.median(times):.0f} mean={np.mean(times):.0f} "
              f"hit_rate={hit_rate:.0%} (min={min(times)} max={max(times)})")

    # Death analysis
    death_causes = Counter()
    death_steps = []
    for ms in all_milestones:
        cause, step = ms.get("death", ("unknown", 0))
        death_causes[cause] += 1
        death_steps.append(step)

    print(f"\nDEATH ANALYSIS:")
    print(f"  Mean lifetime: {np.mean(death_steps):.0f}")
    print(f"  Causes: {dict(death_causes)}")

    # Key question: how many episodes reach wood_3 before first_damage?
    wood3_before_damage = 0
    sword_before_death = 0
    for ms in all_milestones:
        w3 = ms.get("wood_3", float("inf"))
        fd = ms.get("first_damage", float("inf"))
        if w3 < fd:
            wood3_before_damage += 1
        if "sword_crafted" in ms:
            sword_before_death += 1

    print(f"\n  Wood×3 before first damage: {wood3_before_damage}/50 = {wood3_before_damage/50:.0%}")
    print(f"  Sword crafted: {sword_before_death}/50 = {sword_before_death/50:.0%}")


if __name__ == "__main__":
    main()
