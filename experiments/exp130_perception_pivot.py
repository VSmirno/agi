"""Stage 72: Perception Pivot — Self-Organized Perception + Continuous Learning.

Replaces supervised NearDetector with ConceptStore.query_visual_scored() (cosine sim).
Removes GT semantic navigation. Autonomous perceive→decide→act→learn loop.

Phases:
  0. Load frozen CNN encoder from exp128 checkpoint
  1. Init ConceptStore from textbook (NO controlled env grounding)
  2. Phase 1 curriculum: tree nav ≥50% via spatial map only
  3. Phase 2 curriculum: stone nav ≥20%
  4. Phase 3 curriculum: coal grounded from experience
  5. Phase 4 curriculum: survival with enemies, episode ≥200 steps
  6. Verification: confidence growth on confirmed rules
  7. Summary + gates

Gates (from spec):
  - Tree navigation success ≥50% without semantic GT
  - Stone navigation success ≥20% without semantic GT
  - Agent visually grounds new concepts from experience (one-shot)
  - Survival episode length ≥200 with enemies
  - Demo: agent learns, not scripted robot

Design: docs/superpowers/specs/2026-04-07-stage72-perception-pivot-design.md
"""

from __future__ import annotations

import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

CHECKPOINT_DIR = Path("demos/checkpoints/exp130")
EXP128_CHECKPOINT = Path("demos/checkpoints/exp128")

from snks.encoder.cnn_encoder import CNNEncoder, disable_rocm_conv
from snks.encoder.near_detector import NearDetector
from snks.agent.concept_store import ConceptStore
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
)

from exp122_pixels import _detect_near_from_info


# ---------------------------------------------------------------------------
# Phase 0: Load frozen encoder from exp128
# ---------------------------------------------------------------------------

def phase0_load_encoder() -> CNNEncoder:
    """Load frozen CNN encoder from exp128 checkpoint."""
    print("Phase 0: Loading frozen encoder from exp128...")
    t0 = time.time()

    # Try final, then phase3, then phase1
    for tag in ["final", "phase3", "phase1"]:
        path = EXP128_CHECKPOINT / tag / "encoder.pt"
        if path.exists():
            encoder = CNNEncoder()
            encoder.load_state_dict(torch.load(path, weights_only=True))
            encoder.eval()
            # Move to GPU if available
            if torch.cuda.is_available():
                encoder = encoder.cuda()
            print(f"  Loaded encoder from {path} ({time.time()-t0:.1f}s)")
            return encoder

    raise FileNotFoundError(
        f"No encoder checkpoint found in {EXP128_CHECKPOINT}. "
        "Run exp128 first."
    )


# ---------------------------------------------------------------------------
# Phase 1: Init ConceptStore from textbook only (NO visual grounding)
# ---------------------------------------------------------------------------

def phase1_init_store() -> ConceptStore:
    """Load textbook into ConceptStore. NO visual grounding — agent learns from experience."""
    print("Phase 1: Init ConceptStore from textbook (no visual grounding)...")
    store = ConceptStore()
    tb = CrafterTextbook("configs/crafter_textbook.yaml")
    n_rules = tb.load_into(store)
    print(f"  Loaded {n_rules} rules, {len(store.concepts)} concepts")

    # Verify no visual grounding yet
    grounded = [c.id for c in store.concepts.values() if c.visual is not None]
    assert len(grounded) == 0, f"Expected 0 visually grounded, got {grounded}"
    print("  No visual prototypes — agent must learn from experience")
    return store


# ---------------------------------------------------------------------------
# Autonomous agent episode runner (reusable for all phases)
# ---------------------------------------------------------------------------

_DIRECTIONS = ["move_up", "move_down", "move_left", "move_right"]


def run_autonomous_episode(
    encoder: CNNEncoder,
    store: ConceptStore,
    labeler: OutcomeLabeler,
    seed: int,
    max_steps: int = 500,
    enemies: bool = False,
    target_resource: str | None = None,
    verbose: bool = False,
) -> dict:
    """Run one autonomous episode with the Stage 72 agent loop.

    Args:
        encoder: frozen CNN encoder
        store: ConceptStore (mutated — grounding + verification happen in-place)
        labeler: OutcomeLabeler for action outcome detection
        seed: random seed
        max_steps: max steps per episode
        enemies: whether to enable enemies in env
        target_resource: if set, primary goal is navigating to this resource
        verbose: print per-step info

    Returns:
        dict with metrics: length, resources_collected, grounding_events,
        nav_success (if target_resource set), etc.
    """
    env = CrafterPixelEnv(seed=seed)
    if not enemies:
        # Disable enemies by monkeypatching
        try:
            env._env._world._balance_chunk = lambda *a, **kw: None
        except Exception:
            pass

    pixels, info = env.reset()
    rng = np.random.RandomState(seed)
    spatial_map = CrafterSpatialMap()
    reactive = ReactiveCheck(store)

    # Metrics
    resources = Counter()
    grounding_events: list[str] = []
    nav_successes = 0
    nav_attempts = 0
    steps = 0
    current_goal = ""
    current_plan: list = []
    plan_step_idx = 0
    nav_steps = 0
    replan_counter = 0

    device = next(encoder.parameters()).device

    for step in range(max_steps):
        steps = step + 1
        inv = dict(info.get("inventory", {}))

        # 1. PERCEIVE
        pix_tensor = torch.from_numpy(pixels).float()
        if device.type != "cpu":
            pix_tensor = pix_tensor.to(device)

        near_concept, z_real = perceive(pix_tensor, encoder, store)
        near_str = near_concept.id if near_concept is not None else "empty"

        # Update spatial map
        player_pos = info.get("player_pos", (32, 32))
        spatial_map.update(player_pos, near_str)

        # 2. REACTIVE CHECK (danger)
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
            continue

        # 3. GOAL SELECTION
        replan_counter += 1
        if not current_plan or replan_counter >= 20:
            replan_counter = 0

            if target_resource:
                # Forced goal for curriculum phase
                current_goal = target_resource
                current_plan = store.plan(target_resource)
                if not current_plan:
                    # Direct navigation: just go find the resource and "do"
                    from snks.agent.concept_store import PlannedStep
                    current_plan = [PlannedStep(
                        action="do",
                        target=target_resource,
                        near=None,
                        expected_gain=target_resource,
                    )]
            else:
                current_goal, current_plan = select_goal(inv, store)

            plan_step_idx = 0
            nav_steps = 0

            # Handle sleep
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
            # Explore
            unvisited = spatial_map.unvisited_neighbors(player_pos, radius=5)
            if unvisited:
                target = unvisited[rng.randint(len(unvisited))]
                action = _step_toward(player_pos, target, rng)
            else:
                action = str(rng.choice(MOVE_ACTIONS))
            pixels, _, done, info = env.step(action)
            if done:
                break
            continue

        plan_step = current_plan[plan_step_idx]

        # Check if at target
        if near_str == plan_step.target:
            # AT TARGET — execute action
            if plan_step.action == "do":
                nav_attempts += 1
                old_inv = dict(info.get("inventory", {}))

                # Prediction
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
                        # Track resource
                        for k, v in ni.items():
                            delta = v - oi.get(k, 0)
                            if delta > 0 and k not in ("health", "food", "drink", "energy"):
                                resources[k] += delta
                        break

                if done:
                    break

                new_inv = dict(info.get("inventory", {}))
                actual = labeler.label("do", old_inv, new_inv)

                # Verify
                store.verify_after_action(prediction, "do", actual, near=near_str)

                # Experiential grounding
                if z_real is not None:
                    grounded = on_action_outcome("do", old_inv, new_inv, z_real, store, labeler)
                    if grounded:
                        grounding_events.append(grounded)
                        if verbose:
                            print(f"    [{step}] Grounded: {grounded}")

                if success:
                    nav_successes += 1
                    plan_step_idx += 1
                    nav_steps = 0
                else:
                    nav_steps += 8
                    if nav_steps > 200:
                        plan_step_idx += 1
                        nav_steps = 0

            elif plan_step.action in ("make", "place"):
                crafter_action = f"{plan_step.action}_{plan_step.expected_gain}"
                old_inv = dict(info.get("inventory", {}))
                prediction = store.predict_before_action(near_str, plan_step.action, inv)
                pixels, _, done, info = env.step(crafter_action)
                if done:
                    break
                new_inv = dict(info.get("inventory", {}))
                actual = labeler.label(crafter_action, old_inv, new_inv)
                store.verify_after_action(prediction, plan_step.action, actual, near=near_str)
                if actual is not None:
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
        "nav_attempts": nav_attempts,
        "map_visited": spatial_map.n_visited,
        "map_objects": spatial_map.known_objects,
    }


# ---------------------------------------------------------------------------
# Phase 2: Tree navigation curriculum (≥50%)
# ---------------------------------------------------------------------------

def phase2_tree_nav(
    encoder: CNNEncoder,
    store: ConceptStore,
    n_episodes: int = 50,
    max_steps: int = 500,
) -> dict:
    """Phase 1 curriculum: tree nav success ≥50%."""
    print(f"Phase 2: Tree navigation ({n_episodes} episodes, no enemies)...")
    t0 = time.time()
    labeler = OutcomeLabeler()

    successes = 0
    total_wood = 0
    all_grounding: list[str] = []

    for i in range(n_episodes):
        seed = 60000 + i * 7
        result = run_autonomous_episode(
            encoder, store, labeler, seed,
            max_steps=max_steps, enemies=False,
            target_resource="tree",
        )

        wood = result["resources"].get("wood", 0)
        total_wood += wood
        if wood > 0:
            successes += 1
        all_grounding.extend(result["grounding_events"])

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{n_episodes}] success={successes}/{i+1} "
                  f"wood={total_wood} grounded={len(set(all_grounding))}")

    rate = successes / n_episodes
    elapsed = time.time() - t0

    grounded_concepts = [c.id for c in store.concepts.values() if c.visual is not None]

    print(f"  Tree nav: {successes}/{n_episodes} = {rate:.1%}")
    print(f"  Total wood: {total_wood}")
    print(f"  Grounded concepts: {grounded_concepts}")
    print(f"  Gate: {'PASS' if rate >= 0.50 else 'FAIL'} (≥50%)")
    print(f"  ({elapsed:.0f}s)\n")

    return {
        "success_rate": rate,
        "total_wood": total_wood,
        "grounded": grounded_concepts,
        "gate_pass": rate >= 0.50,
    }


# ---------------------------------------------------------------------------
# Phase 3: Stone navigation curriculum (≥20%)
# ---------------------------------------------------------------------------

def phase3_stone_nav(
    encoder: CNNEncoder,
    store: ConceptStore,
    n_episodes: int = 50,
    max_steps: int = 800,
) -> dict:
    """Phase 2 curriculum: stone nav success ≥20%.

    More steps since agent needs: wood → table → pickaxe → find stone → mine.
    """
    print(f"Phase 3: Stone navigation ({n_episodes} episodes, no enemies)...")
    t0 = time.time()
    labeler = OutcomeLabeler()

    successes = 0
    total_stone = 0

    for i in range(n_episodes):
        seed = 70000 + i * 7
        result = run_autonomous_episode(
            encoder, store, labeler, seed,
            max_steps=max_steps, enemies=False,
            target_resource=None,  # let agent plan full chain
        )

        stone = result["resources"].get("stone_item", 0) + result["resources"].get("stone", 0)
        total_stone += stone
        if stone > 0:
            successes += 1

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{n_episodes}] success={successes}/{i+1} stone={total_stone}")

    rate = successes / n_episodes
    elapsed = time.time() - t0

    print(f"  Stone nav: {successes}/{n_episodes} = {rate:.1%}")
    print(f"  Total stone: {total_stone}")
    print(f"  Gate: {'PASS' if rate >= 0.20 else 'FAIL'} (≥20%)")
    print(f"  ({elapsed:.0f}s)\n")

    return {
        "success_rate": rate,
        "total_stone": total_stone,
        "gate_pass": rate >= 0.20,
    }


# ---------------------------------------------------------------------------
# Phase 4: Coal grounded from experience
# ---------------------------------------------------------------------------

def phase4_coal_grounding(
    encoder: CNNEncoder,
    store: ConceptStore,
    n_episodes: int = 30,
    max_steps: int = 1000,
) -> dict:
    """Phase 3 curriculum: coal grounded from experience."""
    print(f"Phase 4: Coal grounding from experience ({n_episodes} episodes)...")
    t0 = time.time()
    labeler = OutcomeLabeler()

    coal_grounded_before = store.query_text("coal") is not None and store.query_text("coal").visual is not None

    for i in range(n_episodes):
        seed = 80000 + i * 7
        result = run_autonomous_episode(
            encoder, store, labeler, seed,
            max_steps=max_steps, enemies=False,
        )

        coal_concept = store.query_text("coal")
        if coal_concept is not None and coal_concept.visual is not None:
            if not coal_grounded_before:
                print(f"  Coal grounded on episode {i+1}!")
                break

        if (i + 1) % 10 == 0:
            grounded = [c.id for c in store.concepts.values() if c.visual is not None]
            print(f"  [{i+1}/{n_episodes}] grounded: {grounded}")

    coal_concept = store.query_text("coal")
    coal_grounded = coal_concept is not None and coal_concept.visual is not None
    elapsed = time.time() - t0

    all_grounded = [c.id for c in store.concepts.values() if c.visual is not None]
    print(f"  All grounded concepts: {all_grounded}")
    print(f"  Coal grounded: {coal_grounded}")
    print(f"  Gate: {'PASS' if coal_grounded else 'FAIL'}")
    print(f"  ({elapsed:.0f}s)\n")

    return {
        "coal_grounded": coal_grounded,
        "all_grounded": all_grounded,
        "gate_pass": coal_grounded,
    }


# ---------------------------------------------------------------------------
# Phase 5: Survival with enemies (episode ≥200)
# ---------------------------------------------------------------------------

def phase5_survival(
    encoder: CNNEncoder,
    store: ConceptStore,
    n_episodes: int = 50,
    max_steps: int = 1000,
) -> dict:
    """Phase 4 curriculum: survival with enemies, mean episode ≥200 steps."""
    print(f"Phase 5: Survival with enemies ({n_episodes} episodes)...")
    t0 = time.time()
    labeler = OutcomeLabeler()

    lengths = []
    total_resources = Counter()

    for i in range(n_episodes):
        seed = 90000 + i * 7
        result = run_autonomous_episode(
            encoder, store, labeler, seed,
            max_steps=max_steps, enemies=True,
        )
        lengths.append(result["length"])
        total_resources.update(result["resources"])

        if (i + 1) % 10 == 0:
            mean_len = np.mean(lengths)
            print(f"  [{i+1}/{n_episodes}] mean_length={mean_len:.0f}")

    mean_length = np.mean(lengths)
    elapsed = time.time() - t0

    print(f"  Mean episode length: {mean_length:.0f} (gate: ≥200)")
    print(f"  Resources: {dict(total_resources)}")
    print(f"  Gate: {'PASS' if mean_length >= 200 else 'FAIL'}")
    print(f"  ({elapsed:.0f}s)\n")

    return {
        "mean_length": mean_length,
        "lengths": lengths,
        "resources": dict(total_resources),
        "gate_pass": mean_length >= 200,
    }


# ---------------------------------------------------------------------------
# Phase 6: Verification — confidence growth
# ---------------------------------------------------------------------------

def phase6_verification(store: ConceptStore) -> dict:
    """Check that verified rules have confidence >0.5 (grew from experience)."""
    print("Phase 6: Verification — confidence check...")

    verified = []
    for cid, concept in store.concepts.items():
        for link in concept.causal_links:
            if link.confidence > 0.5:
                verified.append(f"{cid}.{link.action}→{link.result}: {link.confidence:.2f}")

    print(f"  Rules with confidence >0.5: {len(verified)}")
    for v in verified[:10]:
        print(f"    {v}")

    gate_pass = len(verified) >= 3
    print(f"  Gate: {'PASS' if gate_pass else 'FAIL'} (≥3 rules with confidence >0.5)")
    print()

    return {
        "verified_count": len(verified),
        "verified": verified,
        "gate_pass": gate_pass,
    }


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(
    encoder: CNNEncoder,
    store: ConceptStore,
    tag: str = "final",
) -> None:
    d = CHECKPOINT_DIR / tag
    d.mkdir(parents=True, exist_ok=True)
    torch.save(encoder.state_dict(), d / "encoder.pt")
    store.save(str(d / "concept_store"))
    print(f"  Checkpoint saved → {d}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    disable_rocm_conv()
    print("=" * 60)
    print("exp130: Stage 72 — Perception Pivot")
    print("=" * 60)
    t_start = time.time()

    # Phase 0: Load encoder
    encoder = phase0_load_encoder()

    # Phase 1: Init store (textbook only, no visual grounding)
    store = phase1_init_store()
    save_checkpoint(encoder, store, tag="phase1")

    # Phase 2: Tree nav (≥50%)
    tree_result = phase2_tree_nav(encoder, store)
    save_checkpoint(encoder, store, tag="phase2_tree")

    # Phase 3: Stone nav (≥20%)
    stone_result = phase3_stone_nav(encoder, store)
    save_checkpoint(encoder, store, tag="phase3_stone")

    # Phase 4: Coal grounding from experience
    coal_result = phase4_coal_grounding(encoder, store)
    save_checkpoint(encoder, store, tag="phase4_coal")

    # Phase 5: Survival with enemies (≥200 steps)
    survival_result = phase5_survival(encoder, store)
    save_checkpoint(encoder, store, tag="final")

    # Phase 6: Verification
    verify_result = phase6_verification(store)

    # Summary
    elapsed = time.time() - t_start
    print("=" * 60)
    print(f"exp130 SUMMARY ({elapsed:.0f}s)")
    print("=" * 60)

    gates = {
        "tree_nav_50%": tree_result["gate_pass"],
        "stone_nav_20%": stone_result["gate_pass"],
        "coal_grounding": coal_result["gate_pass"],
        "survival_200": survival_result["gate_pass"],
        "verification": verify_result["gate_pass"],
    }

    print(f"  Tree nav:     {tree_result['success_rate']:.1%} (gate: ≥50%)")
    print(f"  Stone nav:    {stone_result['success_rate']:.1%} (gate: ≥20%)")
    print(f"  Coal grounded: {coal_result['coal_grounded']}")
    print(f"  Survival:     {survival_result['mean_length']:.0f} steps (gate: ≥200)")
    print(f"  Verified:     {verify_result['verified_count']} rules")
    print()

    for gate, passed in gates.items():
        print(f"  Gate {gate}: {'PASS' if passed else 'FAIL'}")

    all_pass = all(gates.values())
    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")

    # Grounded concepts summary
    grounded = [c.id for c in store.concepts.values() if c.visual is not None]
    print(f"\n  Visually grounded concepts: {grounded}")


if __name__ == "__main__":
    main()
