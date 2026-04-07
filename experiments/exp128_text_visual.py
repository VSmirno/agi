"""Stage 71: Text-Visual Integration — Каузальные правила из текста + visual grounding.

Эксперимент проверяет Gates 5 и 7:
- Gate 5: Zombie survival — reactive rules увеличивают episode length
- Gate 7: Regression — smoke ≥60%, QA ≥85% (не хуже Stage 70)

Также собирает данные через ChainGenerator (авто-цепочки из текста) вместо
хардкоженных TREE_CHAIN/IRON_CHAIN и проверяет что результат эквивалентен.

Phases:
  0. Nav encoder (Stage 68 bootstrap, как в exp127)
  1. Load textbook → ConceptStore → visual grounding
  2. ChainGenerator → ScenarioChains → collect data
  3. Train encoder on chain-generated data
  4. Smoke test ≥60%
  5. QA gate ≥85%
  6. Regression (exp123) ≥90%
  7. Zombie survival comparison
  8. Verification loop (prediction error)
"""

from __future__ import annotations

import time
from collections import Counter

import numpy as np
import torch

from snks.encoder.cnn_encoder import CNNEncoder, disable_rocm_conv
from snks.encoder.predictive_trainer import JEPAPredictor, PredictiveTrainer
from snks.encoder.near_detector import NearDetector
from snks.agent.decode_head import NEAR_CLASSES, NEAR_TO_IDX
from snks.agent.crafter_pixel_env import CrafterPixelEnv, CrafterControlledEnv
from snks.agent.crafter_spatial_map import CrafterSpatialMap, find_target_with_map
from snks.agent.outcome_labeler import OutcomeLabeler
from snks.agent.scenario_runner import ScenarioRunner, ScenarioStep

# Stage 71 components
from snks.agent.concept_store import ConceptStore
from snks.agent.crafter_textbook import CrafterTextbook
from snks.agent.chain_generator import ChainGenerator
from snks.agent.grounding_session import GroundingSession
from snks.agent.reactive_check import ReactiveCheck

# Reuse Phase 0 and training from exp127
from exp127_scenario_curriculum import (
    phase0_load_nav_encoder,
    _collect_empty_walk_frames,
    _balance_classes,
    _run_controlled_batch,
    _run_controlled_items_batch,
    _STONE_CONTROLLED,
    _COAL_CONTROLLED,
    _IRON_CONTROLLED,
    _EMPTY_TABLE_CONTROLLED,
)
from exp122_pixels import phase2_train_encoder, _detect_near_from_info
from exp123_pixel_agent import phase3_regression


# ---------------------------------------------------------------------------
# Phase 1: Load textbook + visual grounding
# ---------------------------------------------------------------------------

def phase1_textbook_grounding(
    encoder: CNNEncoder,
) -> tuple[ConceptStore, ChainGenerator]:
    """Load textbook, register concepts, ground visually."""
    print("Phase 1: Loading textbook + visual grounding...")
    t0 = time.time()

    store = ConceptStore()
    tb = CrafterTextbook("configs/crafter_textbook.yaml")
    n_rules = tb.load_into(store)
    print(f"  Loaded {n_rules} rules, {len(store.concepts)} concepts")

    # Visual grounding using controlled env
    # Simple grounding: use CNNEncoder on controlled env frames
    env = CrafterControlledEnv(seed=42)
    encoder.eval()

    visual_targets = [
        c.id for c in store.concepts.values()
        if c.attributes.get("category") in ("resource", "crafted", "terrain", "enemy")
    ]

    for target in visual_targets:
        z_accum = []
        for k in range(5):
            seed = 42 + k * 100
            ctrl_env = CrafterControlledEnv(seed=seed)
            if target == "empty":
                pixels, _ = ctrl_env.reset()
            else:
                try:
                    pixels, _ = ctrl_env.reset_near(target, no_enemies=True)
                except Exception:
                    # Some targets may not be placeable
                    break
            pixels_t = torch.from_numpy(pixels).float().unsqueeze(0)
            with torch.no_grad():
                out = encoder(pixels_t)
            z_accum.append(out.z_real[0])

        if z_accum:
            z_mean = torch.stack(z_accum).mean(dim=0)
            z_norm = torch.nn.functional.normalize(z_mean.unsqueeze(0), dim=1).squeeze(0)
            store.ground_visual(target, z_norm)
            print(f"    Grounded '{target}' ({len(z_accum)} samples)")

    gen = ChainGenerator(store, use_semantic_nav=True)
    print(f"  Available goals: {gen.available_goals()}")
    print(f"  Phase 1 done ({time.time()-t0:.0f}s)\n")
    return store, gen


# ---------------------------------------------------------------------------
# Phase 2: Collect data via ChainGenerator
# ---------------------------------------------------------------------------

def phase2_collect_via_chains(
    detector: NearDetector,
    gen: ChainGenerator,
    store: ConceptStore,
    n_tree: int = 80,
    n_stone_natural: int = 50,
    n_stone_controlled: int = 80,
    n_coal: int = 50,
    n_iron: int = 50,
    n_empty_table: int = 100,
    n_empty_walk: int = 100,
) -> dict:
    """Collect data using ChainGenerator for natural chains + controlled for rare.

    Uses ChainGenerator.generate() for tree/stone chains (replaces hardcoded).
    Still uses controlled env for coal/iron (navigation impossible).
    """
    print("Phase 2: Collecting data via ChainGenerator...")
    t0 = time.time()
    runner = ScenarioRunner()
    labeler = OutcomeLabeler()

    # Auto-generated chains
    tree_chain = gen.generate("wood")
    stone_chain = gen.generate("stone_item")

    print(f"  Generated tree chain: {len(tree_chain)} steps")
    print(f"  Generated stone chain: {len(stone_chain)} steps")

    # Tree (natural, auto-generated chain)
    print(f"  Tree chain ({n_tree} seeds)...")
    tree_labeled = _run_chain_batch_generic(
        runner, detector, labeler, tree_chain, n_tree, 20000, "tree"
    )

    # Stone natural (auto-generated chain)
    print(f"  Stone chain natural ({n_stone_natural} seeds)...")
    stone_natural = _run_chain_batch_generic(
        runner, detector, labeler, stone_chain, n_stone_natural, 23000, "stone"
    )

    # Stone controlled
    print(f"  Stone controlled ({n_stone_controlled} seeds)...")
    stone_controlled = _run_controlled_batch(
        "stone", _STONE_CONTROLLED, {"wood_pickaxe": 1},
        n_stone_controlled, 28000, "stone"
    )
    stone_labeled = stone_natural + stone_controlled

    # Coal controlled
    print(f"  Coal controlled ({n_coal} seeds)...")
    coal_labeled = _run_controlled_batch(
        "coal", _COAL_CONTROLLED, {"wood_pickaxe": 1},
        n_coal, 25000, "coal"
    )

    # Iron controlled
    print(f"  Iron controlled ({n_iron} seeds)...")
    iron_labeled = _run_controlled_batch(
        "iron", _IRON_CONTROLLED, {"stone_pickaxe": 1},
        n_iron, 26000, "iron"
    )

    # Empty/Table controlled
    print(f"  Empty/Table controlled ({n_empty_table} seeds)...")
    empty_table_labeled = _run_controlled_items_batch(
        _EMPTY_TABLE_CONTROLLED, {"wood": 9}, n_empty_table, 27000, "empty"
    )

    # Empty walk
    print(f"  Empty walk ({n_empty_walk} seeds)...")
    t_e = time.time()
    empty_walk = _collect_empty_walk_frames(n_empty_walk, 29000, frames_per_seed=30)
    print(f"    empty (walk): {len(empty_walk)} frames ({time.time()-t_e:.0f}s)")

    all_labeled = (
        tree_labeled + stone_labeled + coal_labeled
        + iron_labeled + empty_table_labeled + empty_walk
    )

    if not all_labeled:
        raise RuntimeError("No labeled frames collected")

    all_labeled = _balance_classes(all_labeled, max_ratio=4.0)

    # Stats
    counter = Counter(NEAR_CLASSES[idx] for _, idx in all_labeled)
    print(f"\n  Balanced class distribution:")
    for cls, cnt in sorted(counter.items()):
        print(f"    {cls}: {cnt}")
    print(f"  Total: {len(all_labeled)} frames")

    # Convert to tensors
    pixels = torch.stack([p for p, _ in all_labeled]).float()
    near_labels = torch.tensor([idx for _, idx in all_labeled], dtype=torch.long)

    elapsed = time.time() - t0
    print(f"  Phase 2 done ({elapsed:.0f}s)\n")

    return {
        "pixels": pixels,
        "near_labels": near_labels,
        "trained_classes": set(counter.keys()),
    }


def _run_chain_batch_generic(
    runner: ScenarioRunner,
    detector: NearDetector,
    labeler: OutcomeLabeler,
    chain: list[ScenarioStep],
    n_seeds: int,
    seed_base: int,
    label: str,
) -> list[tuple[torch.Tensor, int]]:
    """Run auto-generated chain on n_seeds."""
    all_labeled: list[tuple[torch.Tensor, int]] = []
    t0 = time.time()
    n_success = 0

    for seed_idx in range(n_seeds):
        seed = seed_base + seed_idx * 17
        env = CrafterPixelEnv(seed=seed)
        env.reset()
        rng = np.random.RandomState(seed)
        labeled = runner.run_chain(env, detector, labeler, chain, rng)
        all_labeled.extend(labeled)
        seed_classes = set(NEAR_CLASSES[idx] for _, idx in labeled)
        if label in seed_classes:
            n_success += 1

    elapsed = time.time() - t0
    print(f"    {label}: {n_success}/{n_seeds} seeds ({elapsed:.0f}s)")
    return all_labeled


# ---------------------------------------------------------------------------
# Phase 7: Zombie survival comparison
# ---------------------------------------------------------------------------

def phase7_zombie_survival(
    store: ConceptStore,
    detector: NearDetector,
    n_episodes: int = 30,
    max_steps: int = 500,
) -> dict:
    """Compare episode length with vs without reactive zombie handling.

    Gate 5: reactive > baseline × 1.5
    """
    print("Phase 7: Zombie survival comparison...")
    rc = ReactiveCheck(store)

    baseline_lengths = []
    reactive_lengths = []

    for i in range(n_episodes):
        seed = 50000 + i * 7

        # Baseline: no reactive check (ignore zombies)
        env = CrafterPixelEnv(seed=seed)
        pixels, info = env.reset()
        for step in range(max_steps):
            action = int(np.random.RandomState(seed + step).randint(0, 17))
            pixels, _, done, info = env.step(action)
            if done:
                break
        baseline_lengths.append(step + 1)

        # Reactive: flee/attack when zombie detected
        env = CrafterPixelEnv(seed=seed)
        pixels, info = env.reset()
        rng = np.random.RandomState(seed + 99)
        for step in range(max_steps):
            near = detector.detect(torch.from_numpy(pixels).float())
            inv = dict(info.get("inventory", {}))
            override = rc.check(near, inv)

            if override == "flee":
                rc.flee_action(env, rng, steps=4)
                pixels, info = env.observe()
            elif override == "do":
                pixels, _, done, info = env.step("do")
                if done:
                    break
            else:
                action = int(rng.randint(0, 17))
                pixels, _, done, info = env.step(action)
                if done:
                    break
        reactive_lengths.append(step + 1)

    mean_baseline = np.mean(baseline_lengths)
    mean_reactive = np.mean(reactive_lengths)
    ratio = mean_reactive / max(mean_baseline, 1)

    print(f"  Baseline: mean={mean_baseline:.0f} steps")
    print(f"  Reactive: mean={mean_reactive:.0f} steps")
    print(f"  Ratio: {ratio:.2f}x")
    gate5 = ratio >= 1.5
    print(f"  Gate 5 {'PASS' if gate5 else 'FAIL'}: ratio={ratio:.2f} (threshold=1.5x)")
    return {"baseline": mean_baseline, "reactive": mean_reactive, "ratio": ratio, "pass": gate5}


# ---------------------------------------------------------------------------
# Phase 8: Verification loop (prediction error)
# ---------------------------------------------------------------------------

def phase8_verification(store: ConceptStore) -> dict:
    """Run verification loop on a few episodes, check confidence grows."""
    print("Phase 8: Verification loop...")
    labeler = OutcomeLabeler()
    verified_rules = []

    # Simple verification: do actions in controlled env, check confidence
    test_pairs = [
        ("tree", "do", {}, "wood"),
        ("stone", "do", {"wood_pickaxe": 1}, "stone"),
        ("coal", "do", {"wood_pickaxe": 1}, "coal"),
    ]

    for concept_id, action, inv, target in test_pairs:
        initial = None
        for link in store.concepts[concept_id].causal_links:
            if link.action == action:
                initial = link.confidence
                break

        env = CrafterControlledEnv(seed=42)
        if target != concept_id:
            env.reset_near(target, inventory=inv, no_enemies=True)
        else:
            env.reset_near(target, inventory=inv, no_enemies=True)

        # Do action 5 times
        for trial in range(5):
            _, info_before = env.observe()
            inv_before = dict(info_before.get("inventory", {}))
            inv_before.update(inv)  # ensure prereqs present

            pred = store.predict_before_action(concept_id, action, inv_before)
            _, _, _, info_after = env.step("do")
            inv_after = dict(info_after.get("inventory", {}))

            actual = labeler.label(action, inv_before, inv_after)
            store.verify_after_action(pred, action, actual, near=concept_id)

        final = None
        for link in store.concepts[concept_id].causal_links:
            if link.action == action:
                final = link.confidence
                break

        verified_rules.append({
            "concept": concept_id,
            "initial": initial,
            "final": final,
            "grew": final > initial if final and initial else False,
        })
        print(f"  {concept_id}.{action}: {initial:.2f} -> {final:.2f}")

    grew_count = sum(1 for r in verified_rules if r["grew"])
    print(f"  Rules with confidence growth: {grew_count}/{len(verified_rules)}")
    return {"rules": verified_rules, "grew_count": grew_count}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    disable_rocm_conv()
    print("=" * 60)
    print("exp128: Stage 71 — Text-Visual Integration")
    print("=" * 60)
    t_start = time.time()

    # Phase 0: Nav encoder
    nav_encoder, detector = phase0_load_nav_encoder()

    # Phase 1: Textbook + grounding
    store, gen = phase1_textbook_grounding(nav_encoder)

    # Phase 2: Collect data via ChainGenerator
    dataset = phase2_collect_via_chains(detector, gen, store)

    # Phase 3: Train encoder
    print("Phase 3: Training outcome encoder...")
    encoder, predictor, trainer = phase2_train_encoder(dataset, epochs=100)
    encoder.eval().cpu()
    detector_trained = NearDetector(encoder)
    print("Phase 3 done\n")

    # Phase 4: Smoke test
    print("Phase 4: Smoke test...")
    from exp127_scenario_curriculum import phase3_smoke
    smoke_result = phase3_smoke(detector_trained, n_seeds=50)
    print()

    # Phase 5: QA gate
    print("Phase 5: QA gate...")
    from exp127_scenario_curriculum import phase4_qa_gate
    qa_result = phase4_qa_gate(detector_trained, encoder, dataset)
    print()

    # Phase 6: Regression
    print("Phase 6: Regression...")
    regression_result = phase3_regression(encoder)
    print()

    # Phase 7: Zombie survival
    zombie_result = phase7_zombie_survival(store, detector_trained)
    print()

    # Phase 8: Verification loop
    verify_result = phase8_verification(store)
    print()

    # Summary
    elapsed = time.time() - t_start
    print("=" * 60)
    print(f"exp128 SUMMARY ({elapsed:.0f}s)")
    print("=" * 60)

    smoke_acc = smoke_result.get("overall_accuracy", 0)
    qa_pass = qa_result.get("pass_rate", 0)
    regression_acc = regression_result.get("accuracy", 0) if isinstance(regression_result, dict) else 0

    print(f"  Smoke:      {smoke_acc:.1%} (gate: ≥60%)")
    print(f"  QA:         {qa_pass:.1%} (gate: ≥85%)")
    print(f"  Regression: {regression_acc:.1%} (gate: ≥90%)")
    print(f"  Zombie:     {zombie_result['ratio']:.2f}x (gate: ≥1.5x)")
    print(f"  Verified:   {verify_result['grew_count']}/{len(verify_result['rules'])} rules grew")
    print()

    gates = {
        "smoke": smoke_acc >= 0.60,
        "qa": qa_pass >= 0.85,
        "regression": regression_acc >= 0.90,
        "zombie": zombie_result["pass"],
        "verification": verify_result["grew_count"] >= 2,
    }

    for gate, passed in gates.items():
        print(f"  Gate {gate}: {'PASS' if passed else 'FAIL'}")

    all_pass = all(gates.values())
    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")


if __name__ == "__main__":
    main()
