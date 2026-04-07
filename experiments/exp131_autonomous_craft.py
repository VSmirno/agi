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
    on_action_outcome,
    select_goal,
    get_drive_strengths,
    explore_action,
    ground_empty_on_start,
    ground_zombie_on_damage,
    verify_outcome,
    outcome_to_verify,
)

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


def phase1_init_store() -> ConceptStore:
    print("Phase 1: Init ConceptStore from textbook (no visual grounding)...")
    store = ConceptStore()
    tb = CrafterTextbook("configs/crafter_textbook.yaml")
    n_rules = tb.load_into(store)
    print(f"  Loaded {n_rules} rules, {len(store.concepts)} concepts")
    return store


# ---------------------------------------------------------------------------
# Autonomous episode runner (Stage 73)
# ---------------------------------------------------------------------------

def run_autonomous_episode(
    encoder: CNNEncoder,
    store: ConceptStore,
    labeler: OutcomeLabeler,
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

        # 1. PERCEIVE
        pix_tensor = torch.from_numpy(pixels).float()
        if device.type != "cpu":
            pix_tensor = pix_tensor.to(device)

        near_concept, z_real = perceive(pix_tensor, encoder, store)
        near_str = near_concept.id if near_concept is not None else "empty"

        player_pos = info.get("player_pos", (32, 32))
        spatial_map.update(player_pos, near_str)

        # 1b. ZOMBIE GROUNDING (damage detection)
        if prev_inv:
            if ground_zombie_on_damage(prev_inv, inv, z_real, store):
                grounding_events.append("damage→zombie")
                if verbose:
                    print(f"    [{step}] DAMAGE→zombie grounded")
        prev_inv = dict(inv)

        # 2. REACTIVE CHECK
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
            old_inv = dict(info.get("inventory", {}))
            pixels, _, done, info = env.step("do")
            if done:
                break
            # Verify combat
            new_inv = dict(info.get("inventory", {}))
            verify_outcome(near_str, "do", "kill_zombie", store)
            continue

        # 3. GOAL SELECTION
        replan_counter += 1
        if not current_plan or replan_counter >= 20:
            replan_counter = 0
            current_goal, current_plan = select_goal(inv, store)
            plan_step_idx = 0
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
                _, z_before = perceive(pix_t, encoder, store)
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
                    craft_action, old_inv_b, new_inv_b, z_real, store, labeler)
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
                actual_near = labeler.label("do", old_inv, new_inv)
                store.verify_after_action(prediction, "do", actual_near, near=near_str)
                # Universal verify with actual outcome (gained item)
                do_outcome = outcome_to_verify("do", old_inv, new_inv)
                verify_outcome(near_str, "do", do_outcome, store)
                if z_real is not None:
                    grounded = on_action_outcome("do", old_inv, new_inv, z_real, store, labeler)
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
                if z_real is not None:
                    grounded = on_action_outcome(
                        crafter_action, old_inv, new_inv, z_real, store, labeler)
                    if grounded:
                        grounding_events.append(f"craft→{grounded}")
                        if verbose:
                            print(f"    [{step}] PLAN-CRAFT→{grounded}")
                        spatial_map.update(player_pos, grounded)
                if craft_out is not None:
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
                _, z_before = perceive(pix_t, encoder, store)
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
                    craft_action, old_inv_b, new_inv_b, z_real, store, labeler)
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

    return {
        "length": steps,
        "resources": dict(resources),
        "grounding_events": grounding_events,
        "nav_successes": nav_successes,
        "map_visited": spatial_map.n_visited,
        "map_objects": spatial_map.known_objects,
    }


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------

def phase2_tree_nav(encoder, store, n=50, max_steps=1500):
    print(f"Phase 2: Tree navigation ({n} episodes, enemies ON)...")
    t0 = time.time()
    labeler = OutcomeLabeler()
    successes, total_wood = 0, 0
    for i in range(n):
        result = run_autonomous_episode(
            encoder, store, labeler, 60000 + i * 7,
            max_steps=max_steps, enemies=True, verbose=(i < 3))
        wood = result["resources"].get("wood", 0)
        total_wood += wood
        if wood > 0:
            successes += 1
        if (i + 1) % 10 == 0 or i < 3:
            grounded = [c.id for c in store.concepts.values() if c.visual is not None]
            print(f"  [{i+1}/{n}] success={successes}/{i+1} wood={total_wood} "
                  f"grounded={grounded} map={result['map_visited']}")
    rate = successes / n
    grounded = [c.id for c in store.concepts.values() if c.visual is not None]
    print(f"  Tree nav: {successes}/{n} = {rate:.1%} (gate: ≥50%)")
    print(f"  Grounded: {grounded}")
    print(f"  Gate: {'PASS' if rate >= 0.50 else 'FAIL'} ({time.time()-t0:.0f}s)\n")
    return {"success_rate": rate, "gate_pass": rate >= 0.50, "grounded": grounded}


def phase3_stone_nav(encoder, store, n=50, max_steps=2000):
    print(f"Phase 3: Stone navigation ({n} episodes, enemies ON)...")
    t0 = time.time()
    labeler = OutcomeLabeler()
    successes, total_stone = 0, 0
    for i in range(n):
        result = run_autonomous_episode(
            encoder, store, labeler, 70000 + i * 7,
            max_steps=max_steps, enemies=True)
        stone = result["resources"].get("stone_item", 0) + result["resources"].get("stone", 0)
        total_stone += stone
        if stone > 0:
            successes += 1
        if (i + 1) % 10 == 0:
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


def phase5_survival(encoder, store, n=50, max_steps=1500):
    print(f"Phase 5: Survival with enemies ({n} episodes)...")
    t0 = time.time()
    labeler = OutcomeLabeler()
    lengths = []
    for i in range(n):
        result = run_autonomous_episode(
            encoder, store, labeler, 90000 + i * 7,
            max_steps=max_steps, enemies=True)
        lengths.append(result["length"])
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{n}] mean_length={np.mean(lengths):.0f}")
    mean_len = np.mean(lengths)
    print(f"  Mean length: {mean_len:.0f} (gate: ≥200)")
    print(f"  Gate: {'PASS' if mean_len >= 200 else 'FAIL'} ({time.time()-t0:.0f}s)\n")
    return {"mean_length": mean_len, "gate_pass": mean_len >= 200}


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


def main():
    disable_rocm_conv()
    print("=" * 60)
    print("exp131: Stage 73 — Autonomous Craft")
    print("=" * 60)
    t_start = time.time()

    encoder = phase0_load_encoder()
    store = phase1_init_store()
    save_checkpoint(encoder, store, "phase1")

    tree = phase2_tree_nav(encoder, store)
    save_checkpoint(encoder, store, "phase2")

    stone = phase3_stone_nav(encoder, store)
    save_checkpoint(encoder, store, "phase3")

    grounding = phase4_grounding_count(store)

    survival = phase5_survival(encoder, store)
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
