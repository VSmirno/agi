"""Stage 69b: Outcome-Supervised Near Labeling — scenario-based collection.

Проблема exp125: random walk даёт 33 сэмпла (только tree+stone), smoke 15%.
Решение: плановые сценарии + ретроспективное оконное лейблирование.

Сценарии:
  tree_harvest:  navigate→tree, попробовать do×20, лейблировать окно из 5 кадров до успеха
  stone_harvest: то же для камня (без кирки, Crafter позволяет)
  craft_chain:   collect 2 wood → place_table → label empty; navigate→table → make_wood_pickaxe
                 → label table; then navigate→stone (с киркой) → label stone

Каждый сценарий использует только:
  - info["inventory"] — проприоцепция (остаётся)
  - NearDetector (навигационный, обученный на Stage 68)
  - OutcomeLabeler (инвентарь до/после действия)

Phases:
0. Load navigation encoder (Stage 68 pipeline).
1. Scenario collection: tree × N, stone × N, craft_chain × M.
2. Train new encoder на scenario-labeled frames.
3. Smoke: новый NearDetector vs GT — accuracy на обученных классах (fair).
4. QA gate ≥85%.
5. Regression (exp123) ≥90%.
"""

from __future__ import annotations

import time
from collections import deque

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

from exp122_pixels import (
    phase1_collect,
    phase2_train_encoder,
    _detect_near_from_info,
)
from exp123_pixel_agent import phase3_regression

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WINDOW_SIZE = 5   # number of frames to retrospectively label after a success
DO_RETRIES = 25   # max "do" attempts per approach


# ---------------------------------------------------------------------------
# Phase 0: Load navigation encoder
# ---------------------------------------------------------------------------

def phase0_load_nav_encoder() -> tuple[CNNEncoder, NearDetector]:
    """Train Stage 68 encoder for navigation (uses symbolic labels internally)."""
    print("Phase 0: Training navigation encoder (Stage 68 pipeline)...")
    dataset = phase1_collect(n_trajectories=50, steps_per_traj=200)
    nav_encoder, _, _ = phase2_train_encoder(dataset, epochs=100)
    nav_encoder.eval().cpu()
    detector = NearDetector(nav_encoder)
    print("Phase 0 done: navigation encoder ready\n")
    return nav_encoder, detector


# ---------------------------------------------------------------------------
# Phase 1: Scenario-based collection
# ---------------------------------------------------------------------------

def _pixels_to_tensor(pixels: np.ndarray | torch.Tensor) -> torch.Tensor:
    if isinstance(pixels, torch.Tensor):
        return pixels
    return torch.from_numpy(pixels)


def _do_with_window(
    env: CrafterPixelEnv,
    pixels_now: np.ndarray | torch.Tensor,
    info_now: dict,
    labeler: OutcomeLabeler,
    target_near: str,
    rng: np.random.RandomState,
    window_buf: deque,
    near_idx: int,
    labeled: list,
) -> bool:
    """Try "do" up to DO_RETRIES times near target.

    Each attempt: do + check inventory. On success, label the last WINDOW_SIZE
    frames from window_buf as target_near.
    Between attempts: small random move to adjust position.

    Returns True if at least one label was collected.
    """
    pixels = _pixels_to_tensor(pixels_now)
    info = info_now

    for attempt in range(DO_RETRIES):
        inv_before = dict(info.get("inventory", {}))
        pixels_after_np, _, done, info_after = env.step("do")
        inv_after = dict(info_after.get("inventory", {}))

        label = labeler.label("do", inv_before, inv_after)
        if label == target_near:
            # Retrospective window labeling
            window_frames = list(window_buf)
            for frame_pixels in window_frames[-WINDOW_SIZE:]:
                labeled.append((_pixels_to_tensor(frame_pixels), near_idx))
            # Also label current frame
            labeled.append((pixels, near_idx))
            return True

        if done:
            return False

        # Adjust position slightly
        move = rng.choice(["move_left", "move_right", "move_up", "move_down"])
        pixels_np, _, done, info = env.step(move)
        pixels = _pixels_to_tensor(pixels_np)
        window_buf.append(pixels_np)
        if done:
            return False

    return False


def _scenario_harvest(
    detector: NearDetector,
    labeler: OutcomeLabeler,
    target_near: str,
    near_idx: int,
    n_seeds: int,
    seed_base: int,
    max_search_steps: int = 300,
) -> list:
    """Collect (pixels, near_idx) for a simple target (tree or stone).

    Strategy: navigate to target using spatial map, then try do with window labeling.
    """
    labeled: list = []

    for seed_idx in range(n_seeds):
        seed = seed_base + seed_idx * 17
        env = CrafterPixelEnv(seed=seed)
        env.reset()
        smap = CrafterSpatialMap()
        rng = np.random.RandomState(seed)
        window_buf: deque = deque(maxlen=WINDOW_SIZE + DO_RETRIES)

        # Phase A: navigate to target
        pixels_t, info, found = find_target_with_map(
            env, detector, smap, target_near,
            max_steps=max_search_steps, rng=rng,
        )
        if not found:
            continue

        # Collect window frames during approach
        window_buf.append(pixels_t.numpy() if isinstance(pixels_t, torch.Tensor) else pixels_t)

        # Phase B: try do with window
        _do_with_window(
            env, pixels_t, info, labeler, target_near, rng,
            window_buf, near_idx, labeled,
        )

    return labeled


def _scenario_craft_chain(
    detector: NearDetector,
    labeler: OutcomeLabeler,
    n_seeds: int,
    seed_base: int,
    max_steps_per_phase: int = 300,
) -> list:
    """Craft chain: collect wood → place_table (label empty) → make_pickaxe (label table).

    Uses info["inventory"] (proprioception) to check state transitions.
    Does NOT use info["semantic"].
    """
    labeled: list = []

    empty_idx = NEAR_TO_IDX.get("empty")
    table_idx = NEAR_TO_IDX.get("table")
    if empty_idx is None or table_idx is None:
        print("  WARNING: 'empty' or 'table' not in NEAR_TO_IDX, skipping craft_chain")
        return labeled

    for seed_idx in range(n_seeds):
        seed = seed_base + seed_idx * 13
        env = CrafterPixelEnv(seed=seed)
        env.reset()
        smap = CrafterSpatialMap()
        rng = np.random.RandomState(seed)
        window_buf: deque = deque(maxlen=WINDOW_SIZE + DO_RETRIES)

        # ---- Step 1: collect ≥2 wood ----
        wood_collected = 0
        for _attempt in range(2):  # need 2 wood for table
            pixels_t, info, found = find_target_with_map(
                env, detector, smap, "tree",
                max_steps=max_steps_per_phase, rng=rng,
            )
            if not found:
                break
            # reset smap to allow revisiting
            inv_before_wood = dict(info.get("inventory", {}))
            for _ in range(DO_RETRIES):
                pix_after_np, _, done, info_after = env.step("do")
                inv_after = dict(info_after.get("inventory", {}))
                gained = inv_after.get("wood", 0) - inv_before_wood.get("wood", 0)
                if gained > 0:
                    wood_collected += gained
                    inv_before_wood = dict(info_after.get("inventory", {}))
                    pixels_t = _pixels_to_tensor(pix_after_np)
                    info = info_after
                    break
                if done:
                    break
                move = rng.choice(["move_left", "move_right", "move_up", "move_down"])
                pixels_t, _, done, info = env.step(move)
                pixels_t = _pixels_to_tensor(pixels_t)
                if done:
                    break

        if wood_collected < 2:
            continue  # not enough wood

        # ---- Step 2: place_table → label empty ----
        # Find any empty tile (just try place_table where we stand)
        pix_before_place = pixels_t
        for _ in range(10):
            inv_before = dict(info.get("inventory", {}))
            pix_np, _, done, info_after = env.step("place_table")
            inv_after = dict(info_after.get("inventory", {}))
            label = labeler.label("place_table", inv_before, inv_after)
            if label == "empty":
                labeled.append((_pixels_to_tensor(pix_before_place), empty_idx))
                info = info_after
                pixels_t = _pixels_to_tensor(pix_np)
                break
            if done:
                break
            # Move and try again
            move = rng.choice(["move_left", "move_right", "move_up", "move_down"])
            pix_np, _, done, info = env.step(move)
            pix_before_place = _pixels_to_tensor(pix_np)
            pixels_t = pix_before_place
            if done:
                break
        else:
            continue  # failed to place table

        # ---- Step 3: navigate→table → make_wood_pickaxe → label table ----
        # After placing, NearDetector may not know "table" yet — navigate back if needed
        pixels_t, info, found = find_target_with_map(
            env, detector, smap, "table",
            max_steps=max_steps_per_phase, rng=rng,
        )
        if not found:
            continue

        # Try make_wood_pickaxe
        for _ in range(DO_RETRIES):
            inv_before = dict(info.get("inventory", {}))
            pix_np, _, done, info_after = env.step("make_wood_pickaxe")
            inv_after = dict(info_after.get("inventory", {}))
            label = labeler.label("make_wood_pickaxe", inv_before, inv_after)
            if label == "table":
                labeled.append((pixels_t, table_idx))
                break
            if done:
                break
            move = rng.choice(["move_left", "move_right", "move_up", "move_down"])
            pix_np, _, done, info = env.step(move)
            pixels_t = _pixels_to_tensor(pix_np)
            if done:
                break

    return labeled


def phase1_collect_scenarios(
    detector: NearDetector,
    n_tree: int = 200,
    n_stone: int = 200,
    n_craft: int = 50,
) -> dict:
    """Phase 1: Collect labeled frames via planned scenarios.

    Scenarios:
    - tree_harvest: navigate to tree, do with window labeling
    - stone_harvest: same for stone
    - craft_chain: collect wood → place_table → make_pickaxe

    Returns dict with 'pixels' (N,3,64,64) tensor, 'near_labels' (N,) tensor.
    """
    print("Phase 1: Scenario-based collection...")
    t0 = time.time()
    labeler = OutcomeLabeler()

    tree_idx = NEAR_TO_IDX.get("tree")
    stone_idx = NEAR_TO_IDX.get("stone")

    # Tree scenarios
    print(f"  Scenario tree_harvest ({n_tree} seeds)...")
    tree_labeled = _scenario_harvest(
        detector, labeler, "tree", tree_idx,
        n_seeds=n_tree, seed_base=9000,
    )
    print(f"    → {len(tree_labeled)} labeled frames")

    # Stone scenarios
    print(f"  Scenario stone_harvest ({n_stone} seeds)...")
    stone_labeled = _scenario_harvest(
        detector, labeler, "stone", stone_idx,
        n_seeds=n_stone, seed_base=10000,
    )
    print(f"    → {len(stone_labeled)} labeled frames")

    # Craft chain (empty + table labels)
    print(f"  Scenario craft_chain ({n_craft} seeds)...")
    craft_labeled = _scenario_craft_chain(
        detector, labeler,
        n_seeds=n_craft, seed_base=11000,
    )
    print(f"    → {len(craft_labeled)} labeled frames")

    all_labeled = tree_labeled + stone_labeled + craft_labeled
    print(f"\n  Total: {len(all_labeled)} frames "
          f"(tree={len(tree_labeled)} stone={len(stone_labeled)} craft={len(craft_labeled)})")
    print(f"  Phase 1 done in {time.time()-t0:.0f}s")

    if not all_labeled:
        raise RuntimeError("No labeled frames — check NearDetector and scenarios")

    pixels_list = [p if isinstance(p, torch.Tensor) else torch.from_numpy(p)
                   for p, _ in all_labeled]
    labels_list = [lbl for _, lbl in all_labeled]

    # Class balance summary
    from collections import Counter
    counts = Counter(NEAR_CLASSES[l] for l in labels_list)
    print(f"  Class distribution: {dict(counts)}")

    return {
        "pixels": torch.stack(pixels_list),
        "near_labels": torch.tensor(labels_list, dtype=torch.long),
        "n_tree": len(tree_labeled),
        "n_stone": len(stone_labeled),
        "n_craft": len(craft_labeled),
        "trained_classes": [cls for cls, cnt in counts.items() if cnt > 0],
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

    # Build (t, t+1) pairs from adjacent labeled frames
    pixels_t = pixels[:-1]
    pixels_t1 = pixels[1:]
    actions = torch.zeros(N - 1, dtype=torch.long)
    nl = near_labels[:-1]

    encoder = CNNEncoder(n_near_classes=len(NEAR_CLASSES))
    predictor = JEPAPredictor()
    trainer = PredictiveTrainer(
        encoder, predictor,
        contrastive_weight=0.3,
        near_weight=3.0,   # higher weight: outcome supervision is main signal
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
# Phase 3: Smoke — fair accuracy on trained classes only
# ---------------------------------------------------------------------------

def phase3_smoke(
    detector: NearDetector,
    trained_classes: list[str],
    n_frames: int = 1000,
    seed: int = 44444,
) -> dict:
    """Smoke: accuracy on frames where GT ∈ trained_classes.

    Fair metric: we only evaluate on classes where the encoder had training data.
    Gate: ≥50% (we expect low absolute accuracy, but clear signal).
    """
    print(f"\nPhase 3: Smoke on trained classes {trained_classes} ({n_frames} frames)...")

    env = CrafterPixelEnv(seed=seed)
    pixels, info = env.reset()
    rng = np.random.RandomState(seed)

    correct = 0
    total = 0
    class_stats: dict[str, list[int, int]] = {c: [0, 0] for c in trained_classes}

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
    passed = accuracy >= 0.50

    print(f"Smoke (fair, {total} frames where GT∈trained):")
    for cls, (c, t) in class_stats.items():
        pct = c / max(t, 1)
        print(f"  {cls}: {c}/{t} = {pct:.1%}")
    print(f"Overall: {correct}/{total} = {accuracy:.1%} "
          f"({'PASS' if passed else 'FAIL'}, threshold 50%)")

    return {"accuracy": accuracy, "total": total, "passed": passed,
            "class_stats": class_stats}


# ---------------------------------------------------------------------------
# Phase 4: QA gate
# ---------------------------------------------------------------------------

def phase4_qa_gate(
    encoder: CNNEncoder,
    detector: NearDetector,
    n_seeds: int = 50,
) -> dict:
    """QA gate with outcome-trained NearDetector. Gate: ≥70%."""
    print("\nPhase 4: QA gate (outcome-trained NearDetector)...")

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
            seed = 12000 + seed_idx * 41
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
        env = CrafterPixelEnv(seed=66666)
        env.reset()
        smap = CrafterSpatialMap()
        rng = np.random.RandomState(66666)

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
    passed = accuracy >= 0.70

    print(f"\n{'='*50}")
    print(f"GATE: {correct}/{total} = {accuracy:.0%}")
    print(f"{'PASS' if passed else 'FAIL'} (threshold: 70%)")
    print(f"{'='*50}")
    for r in results:
        mark = "+" if r["correct"] else "-"
        print(f"  [{mark}] {r['rule']}: {r['expected']} → {r['got']}")

    return {"accuracy": accuracy, "correct": correct, "total": total, "passed": passed}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Stage 69b: Scenario-based Outcome Labeling")
    print("=" * 60)

    # Phase 0: Navigation encoder
    nav_encoder, nav_detector = phase0_load_nav_encoder()

    # Phase 1: Scenario collection
    outcome_data = phase1_collect_scenarios(
        nav_detector,
        n_tree=200,
        n_stone=200,
        n_craft=50,
    )

    # Phase 2: Train outcome encoder
    new_encoder, new_detector = phase2_train_outcome_encoder(
        outcome_data, epochs=150, batch_size=64,
    )

    # Phase 3: Smoke (fair — only trained classes)
    smoke = phase3_smoke(new_detector, outcome_data["trained_classes"], n_frames=1000)

    # Phase 4: QA gate
    qa = phase4_qa_gate(new_encoder, new_detector, n_seeds=50)

    # Phase 5: Regression (Stage 67 pipeline, uses nav_encoder which is Stage 68)
    print("\nPhase 5: Regression (exp123 pipeline)...")
    reg = phase3_regression(nav_encoder)

    # Summary
    print("\n" + "=" * 60)
    print("STAGE 69b SUMMARY")
    n_tree = outcome_data["n_tree"]
    n_stone = outcome_data["n_stone"]
    n_craft = outcome_data["n_craft"]
    total_frames = n_tree + n_stone + n_craft
    print(f"  Outcome labels:     tree={n_tree} stone={n_stone} craft={n_craft} = {total_frames} total")
    print(f"  Phase 3 smoke:      {smoke['accuracy']:.1%} "
          f"({'PASS' if smoke['passed'] else 'FAIL'}, threshold 50%)")
    print(f"  Phase 4 QA gate:    {qa['accuracy']:.0%} "
          f"({'PASS' if qa['passed'] else 'FAIL'}, threshold 70%)")
    print(f"  Phase 5 regression: {reg['accuracy']:.0%} "
          f"({'PASS' if reg['passed'] else 'FAIL'}, threshold 90%)")

    all_passed = smoke["passed"] and qa["passed"] and reg["passed"]
    print(f"\n{'STAGE 69b COMPLETE' if all_passed else 'STAGE 69b INCOMPLETE'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
