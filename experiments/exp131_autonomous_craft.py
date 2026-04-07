"""Stage 73: Autonomous Craft — Plan-Driven Crafting Through Experience.

Extends exp130 with:
- Empty grounding from first frame (figure/ground separation)
- Zombie grounding through damage
- Craft babbling (place/make actions)
- Universal verification on every outcome
- Enemies ON for all phases (no scaffolding)

Gates:
  - Tree nav ≥50% (already proven in exp130)
  - Stone nav ≥20% (with craft chain)
  - Concepts grounded ≥5 from experience
  - Survival with enemies ≥200 steps
  - Verification: ≥3 rules with confidence >0.5

Design: docs/superpowers/specs/2026-04-07-stage73-autonomous-craft-design.md
"""

from __future__ import annotations

import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

CHECKPOINT_DIR = Path("demos/checkpoints/exp131")
EXP128_CHECKPOINT = Path("demos/checkpoints/exp128")

from snks.encoder.cnn_encoder import CNNEncoder, disable_rocm_conv
from snks.agent.concept_store import ConceptStore, PlannedStep
from snks.agent.crafter_textbook import CrafterTextbook
from snks.agent.crafter_pixel_env import CrafterPixelEnv
from snks.agent.crafter_spatial_map import CrafterSpatialMap, _step_toward, MOVE_ACTIONS
from snks.agent.outcome_labeler import OutcomeLabeler
from snks.agent.reactive_check import ReactiveCheck
from snks.agent.perception import (
    perceive,
    perceive_field,
    VisualField,
    HomeostaticTracker,
    on_action_outcome,
    select_goal,
    get_drive_strengths,
    explore_action,
    ground_empty_on_start,
    ground_zombie_on_damage,
    verify_outcome,
    outcome_to_verify,
)
from snks.agent.crafter_textbook import CrafterTextbook

_DIRECTIONS = ["move_up", "move_down", "move_left", "move_right"]


# ---------------------------------------------------------------------------
# Phase 0: Load frozen encoder from exp128
# ---------------------------------------------------------------------------

def phase0_load_encoder() -> CNNEncoder:
    print("Phase 0: Loading frozen encoder from exp128...")
    t0 = time.time()
    for tag in ["final", "phase3", "phase1"]:
        path = EXP128_CHECKPOINT / tag / "encoder.pt"
        if path.exists():
            encoder = CNNEncoder()
            encoder.load_state_dict(torch.load(path, weights_only=True))
            encoder.eval()
            if torch.cuda.is_available():
                encoder = encoder.cuda()
            print(f"  Loaded encoder from {path} ({time.time()-t0:.1f}s)")
            return encoder
    raise FileNotFoundError(f"No encoder in {EXP128_CHECKPOINT}")


def phase1_init_store() -> tuple[ConceptStore, HomeostaticTracker]:
    print("Phase 1: Init ConceptStore + HomeostaticTracker from textbook...")
    store = ConceptStore()
    tb = CrafterTextbook("configs/crafter_textbook.yaml")
    n_rules = tb.load_into(store)

    tracker = HomeostaticTracker()
    tracker.init_from_body_rules(tb.body_rules)
    print(f"  Loaded {n_rules} rules, {len(store.concepts)} concepts")
    print(f"  Body rules: {len(tb.body_rules)} innate rates")
    return store, tracker


# ---------------------------------------------------------------------------
# Autonomous episode runner (Stage 73)
# ---------------------------------------------------------------------------

def run_autonomous_episode(
    encoder: CNNEncoder,
    store: ConceptStore,
    labeler: OutcomeLabeler,
    tracker: HomeostaticTracker,
    seed: int,
    max_steps: int = 1500,
    enemies: bool = True,
    verbose: bool = False,
) -> dict:
    """Run one autonomous episode with Stage 73 agent loop."""
    env = CrafterPixelEnv(seed=seed)
    if not enemies:
        try:
            env._env._world._balance_chunk = lambda *a, **kw: None
        except Exception:
            pass

    pixels, info = env.reset()
    rng = np.random.RandomState(seed)
    spatial_map = CrafterSpatialMap()
    reactive = ReactiveCheck(store)
    device = next(encoder.parameters()).device

    # --- Bootstrap: ground "empty" from first frame ---
    pix_t = torch.from_numpy(pixels).float()
    if device.type != "cpu":
        pix_t = pix_t.to(device)
    if ground_empty_on_start(pix_t, encoder, store):
        if verbose:
            print(f"    [0] BOOTSTRAP: grounded 'empty'")

    # Metrics
    resources = Counter()
    grounding_events: list[str] = []
    nav_successes = 0
    steps = 0
    current_goal = ""
    current_plan: list = []
    plan_step_idx = 0
    nav_steps = 0
    replan_counter = 0
    prev_inv: dict[str, int] = {}

    for step in range(max_steps):
        steps = step + 1
        inv = dict(info.get("inventory", {}))

        # 1. PERCEIVE (spatial visual field)
        pix_tensor = torch.from_numpy(pixels).float()
        if device.type != "cpu":
            pix_tensor = pix_tensor.to(device)

        vf = perceive_field(pix_tensor, encoder, store)
        near_str = vf.near_concept

        player_pos = info.get("player_pos", (32, 32))
        spatial_map.update(player_pos, near_str)
        # Fill map from peripheral detections
        for cid, _sim, gy, gx in vf.detections:
            if (gy, gx) not in [(1,1),(1,2),(2,1),(2,2)]:
                py, px = int(player_pos[0]), int(player_pos[1])
                dy, dx = gy - 2, gx - 2
                spatial_map.update((py + dy * 2, px + dx * 2), cid)

        # 1b. ZOMBIE GROUNDING + HOMEOSTATIC TRACKING
        if prev_inv:
            if ground_zombie_on_damage(prev_inv, inv, vf, store):
                grounding_events.append("damage→zombie")
                if verbose:
                    print(f"    [{step}] DAMAGE→zombie grounded")
            # Track body rates — how each variable changes and what's visible
            tracker.update(prev_inv, inv, vf.visible_concepts())
        prev_inv = dict(inv)

        # 2b. REACTIVE CHECK
        # Near danger: react immediately (center positions)
        # Far danger: inform drives (select_goal will plan response)
        danger = reactive.check(near_str, inv)
        if danger == "flee":
            for _ in range(4):
                d = _DIRECTIONS[rng.randint(0, 4)]
                pixels, _, done, info = env.step(d)
                if done:
                    break
            if done:
                break
            continue
        if danger == "do":
            # Attack — face the direction where zombie was detected
            pixels, _, done, info = env.step("do")
            if done:
                break
            verify_outcome("zombie", "do", "kill_zombie", store)
            continue

        # 3. GOAL SELECTION
        # Only replan when: no plan, plan completed, OR periodic check when exploring
        replan_counter += 1
        needs_replan = (
            not current_plan
            or plan_step_idx >= len(current_plan)
            or (current_goal == "explore" and replan_counter >= 20)
        )
        if needs_replan:
            replan_counter = 0
            old_goal = current_goal
            current_goal, current_plan = select_goal(
                inv, store, tracker=tracker, visual_field=vf, spatial_map=spatial_map)
            plan_step_idx = 0
            if verbose and current_goal != old_goal:
                print(f"    [{step}] GOAL: {current_goal} ({len(current_plan)} steps)")
            nav_steps = 0

            if current_goal == "restore_energy":
                for _ in range(3):
                    pixels, _, done, info = env.step("sleep")
                    if done:
                        break
                if done:
                    break
                current_plan = []
                continue

        # 4. EXECUTE PLAN
        if not current_plan or plan_step_idx >= len(current_plan):
            # Curiosity-driven exploration with craft babbling
            action = explore_action(rng, store, inventory=inv)

            if action == "babble_do":
                direction = _DIRECTIONS[rng.randint(0, 4)]
                pixels, _, done, info = env.step(direction)
                if done:
                    break
                pix_t = torch.from_numpy(pixels).float()
                if device.type != "cpu":
                    pix_t = pix_t.to(device)
                vf_b = perceive_field(pix_t, encoder, store)
                z_before = vf_b.center_feature
                old_inv_b = dict(info.get("inventory", {}))
                pixels, _, done, info = env.step("do")
                if done:
                    break
                new_inv_b = dict(info.get("inventory", {}))
                grounded = on_action_outcome(
                    "do", old_inv_b, new_inv_b, z_before, store, labeler)
                if grounded:
                    grounding_events.append(f"babble→{grounded}")
                    if verbose:
                        print(f"    [{step}] BABBLE→{grounded}")
                    spatial_map.update(
                        info.get("player_pos", player_pos), grounded)
                    for k, v in new_inv_b.items():
                        delta = v - old_inv_b.get(k, 0)
                        if delta > 0 and k not in ("health", "food", "drink", "energy"):
                            resources[k] += delta
                # Universal verification
                outcome = outcome_to_verify("do", old_inv_b, new_inv_b)
                verify_outcome(grounded or near_str, "do", outcome, store)
                continue

            elif action.startswith("babble_"):
                # Craft babbling: place_table, make_wood_pickaxe
                craft_action = action.replace("babble_", "")
                old_inv_b = dict(info.get("inventory", {}))
                pixels, _, done, info = env.step(craft_action)
                if done:
                    break
                new_inv_b = dict(info.get("inventory", {}))
                grounded = on_action_outcome(
                    craft_action, old_inv_b, new_inv_b,
                    vf.center_feature if vf.center_feature is not None else torch.zeros(256),
                    store, labeler)
                if grounded:
                    grounding_events.append(f"craft→{grounded}")
                    if verbose:
                        print(f"    [{step}] CRAFT→{grounded}")
                    spatial_map.update(
                        info.get("player_pos", player_pos), grounded)
                craft_outcome = outcome_to_verify(craft_action, old_inv_b, new_inv_b)
                verify_outcome(near_str, craft_action.split("_")[0], craft_outcome, store)
                continue
            else:
                unvisited = spatial_map.unvisited_neighbors(player_pos, radius=5)
                if unvisited:
                    target = unvisited[rng.randint(len(unvisited))]
                    action = _step_toward(player_pos, target, rng)
                pixels, _, done, info = env.step(action)
                if done:
                    break
                continue

        plan_step = current_plan[plan_step_idx]

        if verbose and step % 20 == 0:
            print(f"    [{step}] plan[{plan_step_idx}/{len(current_plan)}]: "
                  f"{plan_step.action} {plan_step.target} (near={near_str})")

        # Skip plan steps targeting dangerous concepts — let reactive handle
        # Agent doesn't navigate TO zombie, zombie comes to agent
        target_concept = store.query_text(plan_step.target) if store else None
        if target_concept and target_concept.attributes.get("dangerous"):
            plan_step_idx += 1
            if plan_step_idx >= len(current_plan):
                current_plan = []
            continue

        # Check if at target
        if near_str == plan_step.target:
            if plan_step.action == "do":
                old_inv = dict(info.get("inventory", {}))
                prediction = store.predict_before_action(near_str, "do", inv)
                # Directional probe
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
                        for k, v in ni.items():
                            delta = v - oi.get(k, 0)
                            if delta > 0 and k not in ("health", "food", "drink", "energy"):
                                resources[k] += delta
                        break
                if done:
                    break
                new_inv = dict(info.get("inventory", {}))
                # Verify ONCE with probe result, not per-direction
                if success:
                    verify_outcome(near_str, "do",
                                   plan_step.expected_gain, store)
                if vf.center_feature is not None:
                    grounded = on_action_outcome("do", old_inv, new_inv, vf.center_feature, store, labeler)
                    if grounded:
                        grounding_events.append(f"plan→{grounded}")
                if success:
                    nav_successes += 1
                    plan_step_idx += 1
                    nav_steps = 0
                    spatial_map.update(player_pos, "empty")
                else:
                    nav_steps += 8
                    if nav_steps > 200:
                        plan_step_idx += 1
                        nav_steps = 0

            elif plan_step.action in ("make", "place"):
                crafter_action = f"{plan_step.action}_{plan_step.expected_gain}"
                old_inv = dict(info.get("inventory", {}))
                pixels, _, done, info = env.step(crafter_action)
                if done:
                    break
                new_inv = dict(info.get("inventory", {}))
                craft_out = outcome_to_verify(crafter_action, old_inv, new_inv)
                verify_outcome(near_str, plan_step.action, craft_out, store)
                if vf.center_feature is not None:
                    grounded = on_action_outcome(
                        crafter_action, old_inv, new_inv, vf.center_feature, store, labeler)
                    if grounded:
                        grounding_events.append(f"craft→{grounded}")
                        if verbose:
                            print(f"    [{step}] PLAN-CRAFT→{grounded}")
                        spatial_map.update(player_pos, grounded)
                if craft_out is not None:
                    # Ground the RESULT visually — perceive after placing/crafting
                    # e.g. after place_table, the tile now shows a table
                    if plan_step.action == "place":
                        pix_after = torch.from_numpy(pixels).float()
                        if device.type != "cpu":
                            pix_after = pix_after.to(device)
                        _, z_after = perceive(pix_after, encoder, store)
                        result_concept = store.query_text(plan_step.expected_gain)
                        if result_concept is not None and result_concept.visual is None:
                            z_n = F.normalize(z_after.unsqueeze(0), dim=1).squeeze(0)
                            store.ground_visual(plan_step.expected_gain, z_n)
                            grounding_events.append(f"place→{plan_step.expected_gain}")
                            if verbose:
                                print(f"    [{step}] PLACE-GROUND→{plan_step.expected_gain}")
                            spatial_map.update(player_pos, plan_step.expected_gain)
                    plan_step_idx += 1
                    nav_steps = 0
                else:
                    nav_steps += 1
                    if nav_steps > 15:
                        plan_step_idx += 1
                        nav_steps = 0
            else:
                plan_step_idx += 1

        else:
            # NAVIGATE toward target
            known_pos = spatial_map.find_nearest(plan_step.target, player_pos)
            if known_pos is not None:
                action = _step_toward(player_pos, known_pos, rng)
            else:
                action = explore_action(rng, store, inventory=inv)

            if action == "babble_do":
                direction = _DIRECTIONS[rng.randint(0, 4)]
                pixels, _, done, info = env.step(direction)
                if done:
                    break
                pix_t = torch.from_numpy(pixels).float()
                if device.type != "cpu":
                    pix_t = pix_t.to(device)
                vf_b = perceive_field(pix_t, encoder, store)
                z_before = vf_b.center_feature
                old_inv_b = dict(info.get("inventory", {}))
                pixels, _, done, info = env.step("do")
                if done:
                    break
                new_inv_b = dict(info.get("inventory", {}))
                grounded = on_action_outcome(
                    "do", old_inv_b, new_inv_b, z_before, store, labeler)
                if grounded:
                    grounding_events.append(f"nav→{grounded}")
                    if verbose:
                        print(f"    [{step}] NAV-BABBLE→{grounded}")
                    spatial_map.update(
                        info.get("player_pos", player_pos), grounded)
                    for k, v in new_inv_b.items():
                        delta = v - old_inv_b.get(k, 0)
                        if delta > 0 and k not in ("health", "food", "drink", "energy"):
                            resources[k] += delta
                outcome = outcome_to_verify("do", old_inv_b, new_inv_b)
                verify_outcome(grounded or near_str, "do", outcome, store)
                nav_steps += 2
            elif action.startswith("babble_"):
                craft_action = action.replace("babble_", "")
                old_inv_b = dict(info.get("inventory", {}))
                pixels, _, done, info = env.step(craft_action)
                if done:
                    break
                new_inv_b = dict(info.get("inventory", {}))
                grounded = on_action_outcome(
                    craft_action, old_inv_b, new_inv_b,
                    vf.center_feature if vf.center_feature is not None else torch.zeros(256),
                    store, labeler)
                if grounded:
                    grounding_events.append(f"nav-craft→{grounded}")
                    spatial_map.update(
                        info.get("player_pos", player_pos), grounded)
                craft_outcome = outcome_to_verify(craft_action, old_inv_b, new_inv_b)
                verify_outcome(near_str, craft_action.split("_")[0], craft_outcome, store)
                nav_steps += 1
            else:
                if action not in _DIRECTIONS:
                    unvisited = spatial_map.unvisited_neighbors(player_pos, radius=8)
                    if unvisited:
                        target = unvisited[rng.randint(len(unvisited))]
                        action = _step_toward(player_pos, target, rng)
                    else:
                        action = str(rng.choice(MOVE_ACTIONS))
                pixels, _, done, info = env.step(action)
                nav_steps += 1
                if done:
                    break

            if nav_steps > 200:
                plan_step_idx += 1
                nav_steps = 0

    # Death cause analysis
    final_inv = dict(info.get("inventory", {}))
    health = final_inv.get("health", 0)
    food = final_inv.get("food", 0)
    drink = final_inv.get("drink", 0)
    energy = final_inv.get("energy", 0)
    death_cause = "timeout"
    if steps < max_steps:
        if health <= 0:
            if food == 0 or drink == 0:
                death_cause = "starvation"
            else:
                death_cause = "zombie"
        else:
            death_cause = "unknown"
    if verbose:
        print(f"    death={death_cause} h={health} f={food} d={drink} e={energy} @step={steps}")

    return {
        "length": steps,
        "resources": dict(resources),
        "grounding_events": grounding_events,
        "nav_successes": nav_successes,
        "map_visited": spatial_map.n_visited,
        "map_objects": spatial_map.known_objects,
        "death_cause": death_cause,
        "final_stats": {"health": health, "food": food, "drink": drink, "energy": energy},
    }


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------

def phase2_tree_nav(encoder, store, tracker, n=200, max_steps=1500):
    print(f"Phase 2: Tree navigation ({n} episodes, enemies ON)...")
    t0 = time.time()
    labeler = OutcomeLabeler()
    successes, total_wood = 0, 0
    for i in range(n):
        result = run_autonomous_episode(
            encoder, store, labeler, tracker, 60000 + i * 7,
            max_steps=max_steps, enemies=True, verbose=(i < 3))
        wood = result["resources"].get("wood", 0)
        total_wood += wood
        if wood > 0:
            successes += 1
        if (i + 1) % 50 == 0 or i < 3:
            grounded = [c.id for c in store.concepts.values() if c.visual is not None]
            print(f"  [{i+1}/{n}] success={successes}/{i+1} wood={total_wood} "
                  f"grounded={grounded} map={result['map_visited']}")
    rate = successes / n
    grounded = [c.id for c in store.concepts.values() if c.visual is not None]
    print(f"  Tree nav: {successes}/{n} = {rate:.1%} (gate: ≥50%)")
    print(f"  Grounded: {grounded}")
    print(f"  Gate: {'PASS' if rate >= 0.50 else 'FAIL'} ({time.time()-t0:.0f}s)\n")
    return {"success_rate": rate, "gate_pass": rate >= 0.50, "grounded": grounded}


def phase3_stone_nav(encoder, store, tracker, n=200, max_steps=2000):
    print(f"Phase 3: Stone navigation ({n} episodes, enemies ON)...")
    t0 = time.time()
    labeler = OutcomeLabeler()
    successes, total_stone = 0, 0
    for i in range(n):
        result = run_autonomous_episode(
            encoder, store, labeler, tracker, 70000 + i * 7,
            max_steps=max_steps, enemies=True)
        stone = result["resources"].get("stone_item", 0) + result["resources"].get("stone", 0)
        total_stone += stone
        if stone > 0:
            successes += 1
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{n}] success={successes}/{i+1} stone={total_stone}")
    rate = successes / n
    print(f"  Stone nav: {successes}/{n} = {rate:.1%} (gate: ≥20%)")
    print(f"  Gate: {'PASS' if rate >= 0.20 else 'FAIL'} ({time.time()-t0:.0f}s)\n")
    return {"success_rate": rate, "gate_pass": rate >= 0.20}


def phase4_grounding_count(store):
    grounded = [c.id for c in store.concepts.values() if c.visual is not None]
    print(f"Phase 4: Grounding count...")
    print(f"  Grounded concepts: {grounded} ({len(grounded)})")
    gate = len(grounded) >= 5
    print(f"  Gate: {'PASS' if gate else 'FAIL'} (≥5)\n")
    return {"grounded": grounded, "count": len(grounded), "gate_pass": gate}


def phase5_survival(encoder, store, tracker, n=200, max_steps=1500):
    print(f"Phase 5: Survival with enemies ({n} episodes)...")
    t0 = time.time()
    labeler = OutcomeLabeler()
    lengths = []
    death_causes = Counter()
    sword_episodes = 0
    for i in range(n):
        result = run_autonomous_episode(
            encoder, store, labeler, tracker, 90000 + i * 7,
            max_steps=max_steps, enemies=True, verbose=(i < 5))
        lengths.append(result["length"])
        death_causes[result.get("death_cause", "unknown")] += 1
        if "wood_sword" in result.get("resources", {}):
            sword_episodes += 1
        # Check if sword was crafted (grounding events)
        if any("sword" in e for e in result.get("grounding_events", [])):
            sword_episodes += 1
        if (i + 1) % 50 == 0:
            last50 = lengths[-50:]
            print(f"  [{i+1}/{n}] mean_length={np.mean(lengths):.0f} "
                  f"last50={np.mean(last50):.0f} deaths={dict(death_causes)} "
                  f"sword={sword_episodes}/{i+1}")
    mean_len = np.mean(lengths)
    print(f"  Mean length: {mean_len:.0f} (gate: ≥200)")
    print(f"  Death causes: {dict(death_causes)}")
    print(f"  Gate: {'PASS' if mean_len >= 200 else 'FAIL'} ({time.time()-t0:.0f}s)\n")
    return {"mean_length": mean_len, "gate_pass": mean_len >= 200, "death_causes": dict(death_causes)}


def phase6_verification(store):
    print("Phase 6: Verification...")
    verified = []
    for cid, concept in store.concepts.items():
        for link in concept.causal_links:
            if link.confidence > 0.5:
                verified.append(f"{cid}.{link.action}→{link.result}: {link.confidence:.2f}")
    print(f"  Rules with confidence >0.5: {len(verified)}")
    for v in verified[:10]:
        print(f"    {v}")
    gate = len(verified) >= 3
    print(f"  Gate: {'PASS' if gate else 'FAIL'} (≥3)\n")
    return {"count": len(verified), "verified": verified, "gate_pass": gate}


# ---------------------------------------------------------------------------
# Checkpoint + Main
# ---------------------------------------------------------------------------

def save_checkpoint(encoder, store, tag="final"):
    d = CHECKPOINT_DIR / tag
    d.mkdir(parents=True, exist_ok=True)
    torch.save(encoder.state_dict(), d / "encoder.pt")
    store.save(str(d / "concept_store"))
    print(f"  Checkpoint → {d}")


def main_diagnostic():
    """Diagnostic: survival only, with death cause logging."""
    disable_rocm_conv()
    print("=" * 60)
    print("exp131 DIAGNOSTIC: Survival death causes")
    print("=" * 60)
    encoder = phase0_load_encoder()
    store, tracker = phase1_init_store()
    # Quick bootstrap: 20 episodes to ground concepts
    labeler = OutcomeLabeler()
    for i in range(20):
        run_autonomous_episode(encoder, store, labeler, tracker, 60000 + i * 7,
                               max_steps=500, enemies=True, verbose=(i < 3))
    grounded = [c.id for c in store.concepts.values() if c.visual is not None]
    print(f"\nGrounded after bootstrap: {grounded}")
    print(f"Tracker rates: {tracker.rates}")
    crs = {k: f'{v:.3f}' for k, v in tracker.conditional_rates.items() if abs(v) > 0.001}
    print(f"Conditional rates: {crs}\n")
    # Survival diagnostic
    phase5_survival(encoder, store, tracker, n=50, max_steps=1500)


def main():
    disable_rocm_conv()
    print("=" * 60)
    print("exp131: Stage 74 — Homeostatic Agent")
    print("=" * 60)
    t_start = time.time()

    encoder = phase0_load_encoder()
    store, tracker = phase1_init_store()
    save_checkpoint(encoder, store, "phase1")

    tree = phase2_tree_nav(encoder, store, tracker)
    save_checkpoint(encoder, store, "phase2")

    stone = phase3_stone_nav(encoder, store, tracker)
    save_checkpoint(encoder, store, "phase3")

    grounding = phase4_grounding_count(store)

    survival = phase5_survival(encoder, store, tracker)
    save_checkpoint(encoder, store, "final")

    verify = phase6_verification(store)

    elapsed = time.time() - t_start
    print("=" * 60)
    print(f"exp131 SUMMARY ({elapsed:.0f}s)")
    print("=" * 60)

    gates = {
        "tree_nav_50%": tree["gate_pass"],
        "stone_nav_20%": stone["gate_pass"],
        "grounding_5": grounding["gate_pass"],
        "survival_200": survival["gate_pass"],
        "verification_3": verify["gate_pass"],
    }

    print(f"  Tree nav:   {tree['success_rate']:.1%} (≥50%)")
    print(f"  Stone nav:  {stone['success_rate']:.1%} (≥20%)")
    print(f"  Grounded:   {grounding['count']} concepts (≥5)")
    print(f"  Survival:   {survival['mean_length']:.0f} steps (≥200)")
    print(f"  Verified:   {verify['count']} rules (≥3)")
    print()
    for g, p in gates.items():
        print(f"  Gate {g}: {'PASS' if p else 'FAIL'}")
    print(f"\n  Overall: {'ALL PASS' if all(gates.values()) else 'SOME FAILED'}")
    print(f"  Grounded concepts: {grounding['grounded']}")


if __name__ == "__main__":
    main()
