"""Stage 70: Scenario Curriculum — Целенаправленное Обучение.

Проблема exp126 (Stage 69b): outcome encoder обучен только на tree/stone/table/empty.
Уголь (coal) и железо (iron) не покрыты — нет кирки при random walk.

Решение: ScenarioRunner выполняет полную цепочку S1→S7 в одном эпизоде:
  S1: harvest_wood (×3)
  S2: place_table (label: empty)
  S3: make_wood_pickaxe (label: table)
  S4: harvest_stone (×4, для stone_pickaxe)
  S5: make_stone_pickaxe (label: table)
  S6: harvest_coal (requires wood_pickaxe) ← НОВОЕ
  S7: harvest_iron (requires stone_pickaxe) ← НОВОЕ

Фазы:
  0. Nav encoder: Stage 68 pipeline (symbolic labels, используется как bootstrap)
  1. Scenario collection: полная цепочка S1→S7, N seeds
  2. Train outcome encoder на scenario-labeled frames
  3. Smoke: fair accuracy на обученных классах ≥60%
  4. QA gate ≥85%
  5. Regression (exp123) ≥90%

Критерии успеха:
  - Классов в обучении: ≥6 (tree/stone/coal/table/empty + ещё)
  - Smoke: ≥60%
  - QA gate: ≥85%
  - Regression: ≥90%
"""

from __future__ import annotations

import time
from collections import Counter, deque

import numpy as np
import torch

from snks.encoder.cnn_encoder import CNNEncoder, disable_rocm_conv
from snks.encoder.predictive_trainer import JEPAPredictor, PredictiveTrainer
from snks.encoder.near_detector import NearDetector
from snks.agent.decode_head import NEAR_CLASSES, NEAR_TO_IDX
from snks.agent.crafter_pixel_env import CrafterPixelEnv
from snks.agent.crafter_spatial_map import CrafterSpatialMap, find_target_with_map
from snks.agent.outcome_labeler import OutcomeLabeler
from snks.agent.cls_world_model import CLSWorldModel
from snks.agent.crafter_trainer import CRAFTER_RULES
from snks.agent.crafter_pixel_env import CrafterControlledEnv
from snks.agent.scenario_runner import (
    ScenarioRunner, ScenarioStep, TREE_CHAIN, COAL_CHAIN, IRON_CHAIN,
)

# Controlled scenario steps: target placed adjacent by CrafterControlledEnv,
# no navigation needed. do_retries=3 covers all 4 cardinal directions.
_COAL_CONTROLLED: list[ScenarioStep] = [
    ScenarioStep(None, "do", "coal",
                 prerequisite_inv={"wood_pickaxe": 1}, repeat=3, do_retries=3),
]
_IRON_CONTROLLED: list[ScenarioStep] = [
    ScenarioStep(None, "do", "iron",
                 prerequisite_inv={"stone_pickaxe": 1}, repeat=3, do_retries=3),
]

from exp122_pixels import (
    phase1_collect,
    phase2_train_encoder,
    _detect_near_from_info,
)
from exp123_pixel_agent import phase3_regression


# ---------------------------------------------------------------------------
# Phase 0: Navigation encoder (Stage 68 bootstrap)
# ---------------------------------------------------------------------------

def phase0_load_nav_encoder() -> tuple[CNNEncoder, NearDetector]:
    """Train Stage 68 nav encoder for navigation.

    Uses symbolic labels (info["semantic"]) for training — acknowledged limitation.
    Stage 71 will remove this dependency.
    """
    print("Phase 0: Training navigation encoder (Stage 68 bootstrap)...")
    dataset = phase1_collect(n_trajectories=50, steps_per_traj=200)
    nav_encoder, _, _ = phase2_train_encoder(dataset, epochs=100)
    nav_encoder.eval().cpu()
    detector = NearDetector(nav_encoder)
    print("Phase 0 done: navigation encoder ready\n")
    return nav_encoder, detector


# ---------------------------------------------------------------------------
# Phase 1: Scenario-based collection (S1→S7)
# ---------------------------------------------------------------------------

def _run_chain_batch(
    detector: NearDetector,
    chain: list,
    n_seeds: int,
    seed_base: int,
    label: str,
) -> list[tuple[torch.Tensor, int]]:
    """Run a scenario chain on n_seeds, return labeled frames."""
    runner = ScenarioRunner()
    labeler = OutcomeLabeler()
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


def _run_controlled_batch(
    target: str,
    chain: list,
    inventory: dict[str, int],
    n_seeds: int,
    seed_base: int,
    label: str,
) -> list[tuple[torch.Tensor, int]]:
    """Collect labeled frames using CrafterControlledEnv (target placed adjacent).

    Bypasses navigation — the target material is placed at all 4 cardinal
    neighbors of the player start. Guarantees ~100% collection success rate
    for rare objects (coal, iron) that are inaccessible via natural chains.
    """
    runner = ScenarioRunner()
    labeler = OutcomeLabeler()
    all_labeled: list[tuple[torch.Tensor, int]] = []
    n_success = 0
    t0 = time.time()

    for seed_idx in range(n_seeds):
        seed = seed_base + seed_idx * 13
        env = CrafterControlledEnv(seed=seed)
        env.reset_near(target, inventory=inventory)
        rng = np.random.RandomState(seed)

        labeled = runner.run_chain(env, None, labeler, chain, rng)
        seed_classes = {NEAR_CLASSES[idx] for _, idx in labeled}
        if label in seed_classes:
            n_success += 1
        all_labeled.extend(labeled)

    elapsed = time.time() - t0
    print(f"    {label} (controlled): {n_success}/{n_seeds} seeds ({elapsed:.0f}s)")
    return all_labeled


def _balance_classes(
    labeled: list[tuple[torch.Tensor, int]],
    max_ratio: float = 4.0,
) -> list[tuple[torch.Tensor, int]]:
    """Cap majority classes to max_ratio × minority class count."""
    from collections import defaultdict
    import random
    by_class: dict[int, list] = defaultdict(list)
    for item in labeled:
        by_class[item[1]].append(item)

    if not by_class:
        return labeled

    min_count = min(len(v) for v in by_class.values())
    cap = max(min_count, int(min_count * max_ratio))

    result = []
    for idx, items in by_class.items():
        if len(items) > cap:
            items = random.sample(items, cap)
        result.extend(items)
    random.shuffle(result)
    return result


def phase1_collect_scenarios(
    detector: NearDetector,
    n_tree: int = 80,
    n_coal: int = 50,
    n_iron: int = 50,
) -> dict:
    """Phase 1: Collect labeled frames via scenario chains + controlled envs.

    - TREE_CHAIN (natural): tree + stone + empty + table (high success rate)
    - Coal (controlled): CrafterControlledEnv places coal adjacent, no navigation
    - Iron (controlled): CrafterControlledEnv places iron adjacent, no navigation

    Controlled collection bypasses the need to navigate to rare underground
    materials — guarantees ~100% success vs. ~3% with natural chains.

    Class balancing applied after collection (cap at 4× minority).
    Returns dict with 'pixels', 'near_labels', 'trained_classes'.
    """
    print(f"Phase 1: Independent scenario chains...")
    t0 = time.time()

    print(f"  TREE_CHAIN ({n_tree} seeds)...")
    tree_labeled = _run_chain_batch(detector, TREE_CHAIN, n_tree, 20000, "tree")

    print(f"  Coal controlled ({n_coal} seeds)...")
    coal_labeled = _run_controlled_batch("coal", _COAL_CONTROLLED, {"wood_pickaxe": 1},
                                         n_coal, 25000, "coal")

    print(f"  Iron controlled ({n_iron} seeds)...")
    iron_labeled = _run_controlled_batch("iron", _IRON_CONTROLLED, {"stone_pickaxe": 1},
                                         n_iron, 26000, "iron")

    all_labeled = tree_labeled + coal_labeled + iron_labeled

    if not all_labeled:
        raise RuntimeError("No labeled frames — check NearDetector and ScenarioRunner")

    # Balance classes
    all_labeled = _balance_classes(all_labeled, max_ratio=4.0)

    class_counts: Counter = Counter(NEAR_CLASSES[idx] for _, idx in all_labeled)
    trained_classes = [cls for cls, cnt in class_counts.items() if cnt > 0]

    elapsed = time.time() - t0
    print(f"\nPhase 1 done in {elapsed:.0f}s")
    print(f"  Total frames (after balance): {len(all_labeled)}")
    print(f"  Class distribution: {dict(class_counts)}")
    print(f"  Classes covered: {len(trained_classes)} — {trained_classes}")

    pixels_list = [p for p, _ in all_labeled]
    labels_list = [lbl for _, lbl in all_labeled]

    seeds_reached_coal = sum(1 for _, idx in coal_labeled if NEAR_CLASSES[idx] == "coal")
    seeds_reached_iron = sum(1 for _, idx in iron_labeled if NEAR_CLASSES[idx] == "iron")

    return {
        "pixels": torch.stack(pixels_list),
        "near_labels": torch.tensor(labels_list, dtype=torch.long),
        "trained_classes": trained_classes,
        "class_counts": dict(class_counts),
        "seeds_reached_coal": seeds_reached_coal,
        "seeds_reached_iron": seeds_reached_iron,
        "n_seeds": n_tree + n_coal + n_iron,
    }


# ---------------------------------------------------------------------------
# Phase 2: Train outcome-supervised encoder
# ---------------------------------------------------------------------------

def phase2_train_outcome_encoder(
    outcome_data: dict,
    epochs: int = 150,
    batch_size: int = 64,
) -> tuple[CNNEncoder, NearDetector]:
    """Phase 2: Train CNNEncoder on scenario-labeled frames."""
    N = len(outcome_data["pixels"])
    print(f"\nPhase 2: Training outcome encoder ({N} frames, {epochs} epochs)...")

    train_device = "cuda" if torch.cuda.is_available() else "cpu"
    if train_device == "cuda":
        disable_rocm_conv()

    pixels = outcome_data["pixels"]
    near_labels = outcome_data["near_labels"]

    # Build adjacent (t, t+1) pairs for JEPA predictive loss
    pixels_t = pixels[:-1]
    pixels_t1 = pixels[1:]
    actions = torch.zeros(N - 1, dtype=torch.long)
    nl = near_labels[:-1]

    encoder = CNNEncoder(n_near_classes=len(NEAR_CLASSES))
    predictor = JEPAPredictor()
    trainer = PredictiveTrainer(
        encoder, predictor,
        contrastive_weight=0.3,
        near_weight=3.0,   # outcome supervision is the primary signal
        device=train_device,
    )

    history = trainer.train_full(
        pixels_t, pixels_t1, actions,
        situation_labels=nl,
        near_labels=nl,
        epochs=epochs,
        batch_size=min(batch_size, N - 1),
        log_every=30,
    )

    encoder.eval().cpu()
    detector = NearDetector(encoder)

    final = history[-1]
    print(f"Phase 2 done: pred={final['pred_loss']:.4f} near={final['near_loss']:.4f}")
    return encoder, detector


# ---------------------------------------------------------------------------
# Phase 3: Smoke — fair accuracy on trained classes
# ---------------------------------------------------------------------------

def phase3_smoke(
    detector: NearDetector,
    trained_classes: list[str],
    n_frames: int = 1000,
    seed: int = 44444,
    threshold: float = 0.60,
) -> dict:
    """Smoke test: accuracy on frames where GT ∈ trained_classes.

    Fair metric: evaluate only on classes the encoder was trained on.
    Gate: ≥60% (higher than Stage 69b due to expanded training set).
    """
    print(f"\nPhase 3: Smoke on trained classes {trained_classes} ({n_frames} frames)...")

    env = CrafterPixelEnv(seed=seed)
    pixels, info = env.reset()
    rng = np.random.RandomState(seed)

    correct = 0
    total = 0
    class_stats: dict[str, list] = {c: [0, 0] for c in trained_classes}

    for _ in range(n_frames):
        gt_near = _detect_near_from_info(info)
        if gt_near in trained_classes:
            cnn_near = detector.detect(torch.from_numpy(pixels))
            is_correct = int(cnn_near == gt_near)
            correct += is_correct
            total += 1
            class_stats[gt_near][0] += is_correct
            class_stats[gt_near][1] += 1

        action_idx = int(rng.randint(0, 17))
        pixels, _, done, info = env.step(action_idx)
        if done:
            pixels, info = env.reset()

    accuracy = correct / max(total, 1)
    passed = accuracy >= threshold

    print(f"Smoke (fair, {total} frames where GT∈trained):")
    for cls, (c, t) in class_stats.items():
        pct = c / max(t, 1)
        flag = " ✓" if t > 0 else " (no GT samples)"
        print(f"  {cls}: {c}/{t} = {pct:.1%}{flag}")
    print(f"Overall: {correct}/{total} = {accuracy:.1%} "
          f"({'PASS' if passed else 'FAIL'}, threshold {threshold:.0%})")

    return {"accuracy": accuracy, "total": total, "passed": passed,
            "class_stats": class_stats, "threshold": threshold}


# ---------------------------------------------------------------------------
# Phase 4: QA gate
# ---------------------------------------------------------------------------

def phase4_qa_gate(
    encoder: CNNEncoder,
    detector: NearDetector,
    n_seeds: int = 50,
    threshold: float = 0.85,
) -> dict:
    """QA gate with outcome-trained NearDetector. Gate: ≥85%."""
    print(f"\nPhase 4: QA gate (threshold {threshold:.0%})...")

    from snks.agent.crafter_trainer import generate_taught_transitions
    cls = CLSWorldModel(dim=2048, device="cpu")
    cls.train(generate_taught_transitions())

    print(f"  Collecting prototypes ({n_seeds} seeds × {len(CRAFTER_RULES)} rules)...")
    t0 = time.time()
    n_added = 0
    n_skipped = 0

    for rule in CRAFTER_RULES:
        rule_added = 0
        for seed_idx in range(n_seeds):
            seed = 30000 + seed_idx * 41
            env = CrafterPixelEnv(seed=seed)
            env.reset()
            smap = CrafterSpatialMap()
            rng = np.random.RandomState(seed)

            pixels_t, _, found = find_target_with_map(
                env, detector, smap, rule["near"],
                max_steps=300, rng=rng,
            )
            if not found:
                n_skipped += 1
                continue

            with torch.no_grad():
                out = encoder(pixels_t)

            outcome = {"result": rule["result"], "gives": rule.get("gives", "")}
            cls.prototype_memory.add(out.z_real, rule["action"], outcome)
            rule_added += 1
            n_added += 1

        print(f"    {rule['action']} near {rule['near']}: {rule_added}/{n_seeds}")

    print(f"  Prototypes: {n_added} added, {n_skipped} skipped ({time.time()-t0:.0f}s)")

    print("  Running QA test...")
    correct = 0
    total = 0
    results = []

    for rule in CRAFTER_RULES:
        env = CrafterPixelEnv(seed=77777)
        env.reset()
        smap = CrafterSpatialMap()
        rng = np.random.RandomState(77777)

        pixels_t, _, found = find_target_with_map(
            env, detector, smap, rule["near"],
            max_steps=300, rng=rng,
        )
        if not found:
            continue

        with torch.no_grad():
            outcome, conf, source = cls.query_from_pixels(pixels_t, rule["action"], encoder)

        expected = rule["result"]
        got = outcome.get("result", "unknown")
        is_correct = got == expected
        results.append({
            "rule": f"{rule['action']} near {rule['near']}",
            "expected": expected, "got": got, "correct": is_correct,
        })
        if is_correct:
            correct += 1
        total += 1

    accuracy = correct / max(total, 1)
    passed = accuracy >= threshold

    print(f"\n{'='*50}")
    print(f"GATE: {correct}/{total} = {accuracy:.0%}")
    print(f"{'PASS' if passed else 'FAIL'} (threshold: {threshold:.0%})")
    print(f"{'='*50}")
    for r in results:
        mark = "+" if r["correct"] else "-"
        print(f"  [{mark}] {r['rule']}: {r['expected']} → {r['got']}")

    return {"accuracy": accuracy, "correct": correct, "total": total,
            "passed": passed, "threshold": threshold}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Stage 70: Scenario Curriculum — Целенаправленное Обучение")
    print("=" * 60)

    # Phase 0: Navigation encoder (Stage 68 bootstrap)
    nav_encoder, nav_detector = phase0_load_nav_encoder()

    # Phase 1: Independent chains per material class
    outcome_data = phase1_collect_scenarios(
        nav_detector,
        n_tree=60,
        n_coal=150,
        n_iron=200,
    )

    n_classes = len(outcome_data["trained_classes"])
    print(f"\nCoverage check: {n_classes} classes covered "
          f"(target: ≥6)")
    if n_classes < 6:
        print(f"  WARNING: only {n_classes} classes, need ≥6")

    # Phase 2: Train outcome encoder
    new_encoder, new_detector = phase2_train_outcome_encoder(
        outcome_data, epochs=150, batch_size=64,
    )

    # Phase 3: Smoke (fair, trained classes)
    smoke = phase3_smoke(
        new_detector, outcome_data["trained_classes"],
        n_frames=1000, threshold=0.60,
    )

    # Phase 4: QA gate
    qa = phase4_qa_gate(new_encoder, new_detector, n_seeds=50, threshold=0.85)

    # Phase 5: Regression
    print("\nPhase 5: Regression (exp123 pipeline)...")
    reg = phase3_regression(nav_encoder)

    # Summary
    print("\n" + "=" * 60)
    print("STAGE 70 SUMMARY")
    print(f"  Seeds:          {outcome_data['n_seeds']} total (80 tree + 150 coal + 150 iron)")
    print(f"  Coal frames:    {outcome_data['seeds_reached_coal']}")
    print(f"  Iron frames:    {outcome_data['seeds_reached_iron']}")
    print(f"  Classes:        {outcome_data['trained_classes']}")
    print(f"  Frame counts:   {outcome_data['class_counts']}")
    print(f"  Phase 3 smoke:  {smoke['accuracy']:.1%} "
          f"({'PASS' if smoke['passed'] else 'FAIL'}, threshold {smoke['threshold']:.0%})")
    print(f"  Phase 4 QA:     {qa['accuracy']:.0%} "
          f"({'PASS' if qa['passed'] else 'FAIL'}, threshold {qa['threshold']:.0%})")
    print(f"  Phase 5 regr.:  {reg['accuracy']:.0%} "
          f"({'PASS' if reg['passed'] else 'FAIL'}, threshold 90%)")

    all_passed = (
        smoke["passed"]
        and qa["passed"]
        and reg["passed"]
        and len(outcome_data["trained_classes"]) >= 6
    )
    print(f"\n{'STAGE 70 COMPLETE' if all_passed else 'STAGE 70 INCOMPLETE'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
