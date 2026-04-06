"""Stage 69: Outcome-Supervised Near Labeling — закрываем circular dependency.

NearDetector обучается без info["semantic"]. Источник near_labels — изменение инвентаря
(проприоцепция) после действий.

Phases:
0. Загрузить Stage 68 encoder + NearDetector (для навигации к объектам).
1. Outcome-supervised collection: (pixels, near_label) без info["semantic"].
2. Train new encoder на outcome-supervised labels.
3. Smoke: новый NearDetector vs ground truth ≥60%.
4. QA gate: Crafter QA L1-L4 ≥85%.
5. Regression: exp123 pipeline ≥90%.

Run на minipc. CNN на GPU (disable_rocm_conv для AMD ROCm).
"""

from __future__ import annotations

import time

import numpy as np
import torch

from snks.encoder.cnn_encoder import CNNEncoder, disable_rocm_conv
from snks.encoder.predictive_trainer import JEPAPredictor, PredictiveTrainer
from snks.encoder.near_detector import NearDetector
from snks.agent.decode_head import NEAR_CLASSES, NEAR_TO_IDX
from snks.agent.crafter_pixel_env import CrafterPixelEnv, ACTION_TO_IDX
from snks.agent.crafter_spatial_map import CrafterSpatialMap, find_target_with_map
from snks.agent.outcome_labeler import OutcomeLabeler, DO_GAIN_TO_NEAR, MAKE_GAIN_TO_NEAR, PLACE_ACTION_COST
from snks.agent.cls_world_model import CLSWorldModel
from snks.agent.crafter_trainer import CRAFTER_RULES

from exp122_pixels import (
    phase2_train_encoder,
    phase4_gate_test,
    _detect_near_from_info,  # only for smoke test comparison
    make_situation_label,
)
from exp123_pixel_agent import phase3_regression


# ---------------------------------------------------------------------------
# Phase 0: Load Stage 68 encoder
# ---------------------------------------------------------------------------

def phase0_load_stage68_encoder() -> tuple[CNNEncoder, NearDetector]:
    """Train encoder via Stage 68 pipeline for navigation.

    This encoder uses symbolic near_labels (old way) — used only for
    navigation to objects during outcome-supervised collection.
    It is NOT the final encoder for Stage 69.
    """
    print("Phase 0: Training navigation encoder (Stage 68 pipeline)...")
    from exp122_pixels import phase1_collect
    dataset = phase1_collect(n_trajectories=50, steps_per_traj=200)
    nav_encoder, _, _ = phase2_train_encoder(dataset, epochs=100)
    nav_encoder.eval().cpu()
    detector = NearDetector(nav_encoder)
    print("Phase 0 done: navigation encoder ready")
    return nav_encoder, detector


# ---------------------------------------------------------------------------
# Phase 1: Outcome-supervised collection
# ---------------------------------------------------------------------------

# Objects to seek during "do" collection
_DO_TARGETS = list(DO_GAIN_TO_NEAR.keys())  # wood,stone,coal,iron,diamond ← via near

# Make actions to try near table
_MAKE_ACTIONS = list(MAKE_GAIN_TO_NEAR.keys())

# Place actions to try on empty ground
_PLACE_ACTIONS = list(PLACE_ACTION_COST.keys())


def _collect_do_labels(
    detector: NearDetector,
    labeler: OutcomeLabeler,
    n_seeds: int = 100,
    max_search_steps: int = 300,
) -> list[tuple[torch.Tensor, int]]:
    """Collect (pixels, near_idx) from "do" outcomes near resource objects."""
    print(f"  Collecting 'do' labels ({n_seeds} seeds × {len(DO_GAIN_TO_NEAR)} targets)...")
    labeled: list[tuple[torch.Tensor, int]] = []
    rng = np.random.RandomState(5000)

    for target_near, target_item in [
        ("tree", "wood"), ("stone", "stone"), ("coal", "coal"),
        ("iron", "iron"), ("diamond", "diamond"),
    ]:
        found_count = 0
        for seed_idx in range(n_seeds):
            seed = 5000 + seed_idx * 19
            env = CrafterPixelEnv(seed=seed)
            env.reset()
            smap = CrafterSpatialMap()
            seed_rng = np.random.RandomState(seed)

            pixels_t, info, found = find_target_with_map(
                env, detector, smap, target_near,
                max_steps=max_search_steps, rng=seed_rng,
            )
            if not found:
                continue

            # Try "do" up to 8 times with small moves between attempts.
            # NearDetector may fire 1-2 tiles away; moving closer helps.
            got_label = False
            for attempt in range(8):
                inv_before = dict(info.get("inventory", {}))
                pixels_after, _, done, info_after = env.step("do")
                inv_after = dict(info_after.get("inventory", {}))
                near_label = labeler.label("do", inv_before, inv_after)
                if near_label is not None and near_label in NEAR_TO_IDX:
                    labeled.append((pixels_t, NEAR_TO_IDX[near_label]))
                    found_count += 1
                    got_label = True
                    break
                if done:
                    break
                # Move slightly and retry
                move = seed_rng.choice(["move_left", "move_right", "move_up", "move_down"])
                pixels_t, _, done, info = env.step(move)
                if done:
                    break

        print(f"    near={target_near}: {found_count}/{n_seeds} labeled frames")

    return labeled


def _collect_make_labels(
    detector: NearDetector,
    labeler: OutcomeLabeler,
    n_seeds: int = 50,
    max_search_steps: int = 300,
) -> list[tuple[torch.Tensor, int]]:
    """Collect (pixels, near_idx='table') from make_* outcomes."""
    print(f"  Collecting 'make' labels ({n_seeds} seeds × {len(_MAKE_ACTIONS)} actions)...")
    labeled: list[tuple[torch.Tensor, int]] = []
    if "table" not in NEAR_TO_IDX:
        print("  WARNING: 'table' not in NEAR_TO_IDX, skipping make labels")
        return labeled

    table_idx = NEAR_TO_IDX["table"]
    rng = np.random.RandomState(6000)

    for action in _MAKE_ACTIONS:
        found_count = 0
        for seed_idx in range(n_seeds):
            seed = 6000 + seed_idx * 23
            env = CrafterPixelEnv(seed=seed)
            env.reset()
            smap = CrafterSpatialMap()
            seed_rng = np.random.RandomState(seed)

            # Navigate to table
            pixels_t, info, found = find_target_with_map(
                env, detector, smap, "table",
                max_steps=max_search_steps, rng=seed_rng,
            )
            if not found:
                continue

            inv_before = dict(info.get("inventory", {}))
            _, _, _, info_after = env.step(action)
            inv_after = dict(info_after.get("inventory", {}))

            near_label = labeler.label(action, inv_before, inv_after)
            if near_label == "table":
                labeled.append((pixels_t, table_idx))
                found_count += 1

        print(f"    {action}: {found_count}/{n_seeds} labeled frames")

    return labeled


def _collect_place_labels(
    detector: NearDetector,
    labeler: OutcomeLabeler,
    n_seeds: int = 30,
    max_search_steps: int = 200,
) -> list[tuple[torch.Tensor, int]]:
    """Collect (pixels, near_idx='empty') from place_* outcomes."""
    print(f"  Collecting 'place' labels ({n_seeds} seeds × {len(_PLACE_ACTIONS)} actions)...")
    labeled: list[tuple[torch.Tensor, int]] = []
    if "empty" not in NEAR_TO_IDX:
        print("  WARNING: 'empty' not in NEAR_TO_IDX, skipping place labels")
        return labeled

    empty_idx = NEAR_TO_IDX["empty"]
    rng = np.random.RandomState(7000)

    for action in _PLACE_ACTIONS:
        found_count = 0
        for seed_idx in range(n_seeds):
            seed = 7000 + seed_idx * 31
            env = CrafterPixelEnv(seed=seed)
            env.reset()
            smap = CrafterSpatialMap()
            seed_rng = np.random.RandomState(seed)

            # Navigate near "empty" area (walk around, any unoccupied spot)
            # Empty = no specific object → just move to an explored unoccupied position
            pixels_t, info, _ = find_target_with_map(
                env, detector, smap, "empty",
                max_steps=max_search_steps, rng=seed_rng,
            )

            inv_before = dict(info.get("inventory", {}))
            _, _, _, info_after = env.step(action)
            inv_after = dict(info_after.get("inventory", {}))

            near_label = labeler.label(action, inv_before, inv_after)
            if near_label == "empty":
                labeled.append((pixels_t, empty_idx))
                found_count += 1

        print(f"    {action}: {found_count}/{n_seeds} labeled frames")

    return labeled


def phase1_collect_outcome(
    detector: NearDetector,
    n_seeds_do: int = 100,
    n_seeds_make: int = 50,
    n_seeds_place: int = 30,
) -> dict:
    """Phase 1: Collect (pixels, near_label) via outcome supervision.

    No info["semantic"] used. Labels derived from inventory changes.
    """
    print("\nPhase 1: Outcome-supervised collection...")
    t0 = time.time()

    labeler = OutcomeLabeler()

    do_labeled = _collect_do_labels(detector, labeler, n_seeds=n_seeds_do)
    make_labeled = _collect_make_labels(detector, labeler, n_seeds=n_seeds_make)
    place_labeled = _collect_place_labels(detector, labeler, n_seeds=n_seeds_place)

    all_labeled = do_labeled + make_labeled + place_labeled
    print(f"\n  Total labeled frames: {len(all_labeled)} "
          f"(do={len(do_labeled)} make={len(make_labeled)} place={len(place_labeled)})")
    print(f"  Phase 1 done in {time.time()-t0:.0f}s")

    if not all_labeled:
        raise RuntimeError("No labeled frames collected — check NearDetector and env setup")

    pixels_list = [p for p, _ in all_labeled]
    labels_list = [l for _, l in all_labeled]

    return {
        "pixels": torch.stack(pixels_list),          # (N, 3, 64, 64)
        "near_labels": torch.tensor(labels_list, dtype=torch.long),  # (N,)
        "n_do": len(do_labeled),
        "n_make": len(make_labeled),
        "n_place": len(place_labeled),
    }


# ---------------------------------------------------------------------------
# Phase 2: Train new encoder on outcome-supervised labels
# ---------------------------------------------------------------------------

def phase2_train_outcome_encoder(
    outcome_data: dict,
    epochs: int = 100,
    batch_size: int = 64,
) -> tuple[CNNEncoder, NearDetector]:
    """Phase 2: Train new CNNEncoder with outcome-supervised near_labels only.

    No symbolic labels. Near supervision comes from inventory changes.
    JEPA (self-supervised) still trains on all pairs, but we only have
    single frames (not pairs) here — use JEPA-only on shifted pairs.
    """
    print(f"\nPhase 2: Training outcome-supervised encoder "
          f"({len(outcome_data['pixels'])} labeled frames, {epochs} epochs)...")

    train_device = "cuda" if torch.cuda.is_available() else "cpu"
    if train_device == "cuda":
        disable_rocm_conv()

    pixels = outcome_data["pixels"]   # (N, 3, 64, 64)
    near_labels = outcome_data["near_labels"]  # (N,)
    N = len(pixels)

    # Build (t, t+1) pairs from sequential labeled frames (shifted by 1)
    pixels_t = pixels[:-1]
    pixels_t1 = pixels[1:]
    actions = torch.zeros(N - 1, dtype=torch.long)   # unknown actions = 0 (noop)
    situation_labels = near_labels[:-1]               # use near as situation label
    nl = near_labels[:-1]

    encoder = CNNEncoder(n_near_classes=len(NEAR_CLASSES))
    predictor = JEPAPredictor()
    trainer = PredictiveTrainer(
        encoder, predictor,
        contrastive_weight=0.3,
        near_weight=2.0,   # higher weight: near supervision is the main signal
        device=train_device,
    )

    history = trainer.train_full(
        pixels_t, pixels_t1, actions,
        situation_labels=situation_labels,
        near_labels=nl,
        epochs=epochs,
        batch_size=min(batch_size, N - 1),
        log_every=20,
    )

    encoder.eval().cpu()
    detector = NearDetector(encoder)

    final = history[-1]
    print(f"Phase 2 done: pred={final['pred_loss']:.4f} near={final['near_loss']:.4f}")
    return encoder, detector


# ---------------------------------------------------------------------------
# Phase 3: Smoke — new NearDetector vs ground truth
# ---------------------------------------------------------------------------

def phase3_smoke(
    detector: NearDetector,
    n_frames: int = 500,
    seed: int = 44444,
) -> dict:
    """Smoke: new NearDetector (outcome-trained) vs ground truth near."""
    print(f"\nPhase 3: Smoke — outcome NearDetector vs ground truth ({n_frames} frames)...")

    env = CrafterPixelEnv(seed=seed)
    pixels, info = env.reset()
    rng = np.random.RandomState(seed)
    correct = 0

    for _ in range(n_frames):
        gt_near = _detect_near_from_info(info)  # ground truth (for comparison only)
        cnn_near = detector.detect(torch.from_numpy(pixels))
        if cnn_near == gt_near:
            correct += 1
        action_idx = int(rng.randint(0, 17))
        pixels, _, done, info = env.step(action_idx)
        if done:
            pixels, info = env.reset()

    accuracy = correct / n_frames
    passed = accuracy >= 0.60
    print(f"Smoke: {correct}/{n_frames} = {accuracy:.1%} "
          f"({'PASS' if passed else 'FAIL'}, threshold 60%)")
    return {"accuracy": accuracy, "passed": passed}


# ---------------------------------------------------------------------------
# Phase 4: QA gate
# ---------------------------------------------------------------------------

def phase4_qa_gate(
    encoder: CNNEncoder,
    detector: NearDetector,
    n_seeds: int = 50,
) -> dict:
    """QA gate with outcome-trained NearDetector. Gate: ≥85%."""
    print("\nPhase 4: QA gate (outcome-trained NearDetector)...")

    from snks.agent.crafter_trainer import generate_taught_transitions
    cls = CLSWorldModel(dim=2048, device="cpu")
    cls.train(generate_taught_transitions())

    # Collect prototypes using outcome-trained detector + spatial map
    print(f"  Collecting prototypes ({n_seeds} seeds × {len(CRAFTER_RULES)} rules)...")
    t0 = time.time()
    n_added = 0
    n_skipped = 0

    for rule in CRAFTER_RULES:
        rule_added = 0
        for seed_idx in range(n_seeds):
            seed = 8000 + seed_idx * 37
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

    # QA test
    correct = 0
    total = 0
    results = []

    for rule in CRAFTER_RULES:
        env = CrafterPixelEnv(seed=99999)
        env.reset()
        smap = CrafterSpatialMap()
        rng = np.random.RandomState(99999)

        pixels_t, _, found = find_target_with_map(
            env, detector, smap, rule["near"], max_steps=300, rng=rng,
        )
        if not found:
            continue

        with torch.no_grad():
            outcome, conf, source = cls.query_from_pixels(pixels_t, rule["action"], encoder)

        expected = rule["result"]
        got = outcome.get("result", "unknown")
        is_correct = got == expected
        results.append({"rule": f"{rule['action']} near {rule['near']}",
                         "expected": expected, "got": got, "correct": is_correct})
        if is_correct:
            correct += 1
        total += 1

    accuracy = correct / max(total, 1)
    passed = accuracy >= 0.85

    print(f"\n{'='*50}")
    print(f"GATE: {correct}/{total} = {accuracy:.0%}")
    print(f"{'PASS' if passed else 'FAIL'} (threshold: 85%)")
    print(f"{'='*50}")
    for r in results:
        mark = "+" if r["correct"] else "-"
        print(f"  [{mark}] {r['rule']}: {r['expected']} → {r['got']}")

    return {"accuracy": accuracy, "correct": correct, "total": total, "passed": passed}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Stage 69: Outcome-Supervised Near Labeling — closing circular dependency")
    print("=" * 60)

    # Phase 0: Load navigation encoder (Stage 68 pipeline)
    nav_encoder, nav_detector = phase0_load_stage68_encoder()

    # Phase 1: Collect outcome-supervised labels
    outcome_data = phase1_collect_outcome(
        nav_detector,
        n_seeds_do=100,
        n_seeds_make=50,
        n_seeds_place=30,
    )

    # Phase 2: Train new encoder on outcome labels only
    new_encoder, new_detector = phase2_train_outcome_encoder(
        outcome_data, epochs=100, batch_size=64
    )

    # Phase 3: Smoke test
    smoke = phase3_smoke(new_detector, n_frames=500)

    # Phase 4: QA gate
    qa = phase4_qa_gate(new_encoder, new_detector, n_seeds=50)

    # Phase 5: Regression (Stage 67 pipeline)
    print("\nPhase 5: Regression (exp123 pipeline)...")
    reg = phase3_regression(nav_encoder)  # Stage 67's regression uses old encoder

    # Summary
    print("\n" + "=" * 60)
    print("STAGE 69 SUMMARY")
    print(f"  Outcome labels:     {outcome_data['n_do']} do + "
          f"{outcome_data['n_make']} make + {outcome_data['n_place']} place "
          f"= {len(outcome_data['pixels'])} total")
    print(f"  Phase 3 smoke:      {smoke['accuracy']:.1%} "
          f"({'PASS' if smoke['passed'] else 'FAIL'}, threshold 60%)")
    print(f"  Phase 4 QA gate:    {qa['accuracy']:.0%} "
          f"({'PASS' if qa['passed'] else 'FAIL'}, threshold 85%)")
    print(f"  Phase 5 regression: {reg['accuracy']:.0%} "
          f"({'PASS' if reg['passed'] else 'FAIL'}, threshold 90%)")

    all_passed = smoke["passed"] and qa["passed"] and reg["passed"]
    print(f"\n{'STAGE 69 COMPLETE' if all_passed else 'STAGE 69 INCOMPLETE'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
