"""exp135: Stage 75 — Retrain CNN with grid_size=8 + tile_head.

Root cause from exp134: grid_size=4 feature map cells cover ~2.3 tiles each.
Features are mixed → intra-class sim ≈ inter-class sim → tile_head can't classify.

Fix: grid_size=8 → 8×8 feature map → each cell = 8×8 pixels ≈ 1 Crafter tile.
3 conv layers (stride=2 each): 64→32→16→8.

Pipeline:
  Phase 0: Load nav encoder (for data collection navigation)
  Phase 1: Collect labeled data (reuse exp128 pipeline)
  Phase 2: Train NEW CNN encoder (grid_size=8, feature_channels=256)
  Phase 3: Train tile_head on new encoder
  Phase 4: Tile accuracy gate (≥60%)
  Phase 5: Smoke test — wood collection
  Phase 6: Survival with enemies
  Phase 7: Save checkpoint

Design: docs/superpowers/specs/2026-04-08-stage75-per-tile-visual-field-design.md
"""

from __future__ import annotations

import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

CHECKPOINT_DIR = Path("demos/checkpoints/exp135")
EXP128_CHECKPOINT = Path("demos/checkpoints/exp128")

from snks.encoder.cnn_encoder import CNNEncoder, disable_rocm_conv
from snks.encoder.predictive_trainer import JEPAPredictor, PredictiveTrainer
from snks.encoder.near_detector import NearDetector
from snks.encoder.tile_head_trainer import (
    collect_tile_training_data,
    train_tile_head,
    semantic_cell_label,
)
from snks.agent.decode_head import NEAR_CLASSES, NEAR_TO_IDX
from snks.agent.concept_store import ConceptStore
from snks.agent.crafter_textbook import CrafterTextbook
from snks.agent.crafter_pixel_env import CrafterPixelEnv, CrafterControlledEnv
from snks.agent.crafter_spatial_map import CrafterSpatialMap, _step_toward, MOVE_ACTIONS
from snks.agent.outcome_labeler import OutcomeLabeler
from snks.agent.scenario_runner import ScenarioRunner
from snks.agent.perception import (
    perceive_tile_field,
    HomeostaticTracker,
    on_action_outcome,
    select_goal,
    explore_action,
    verify_outcome,
    outcome_to_verify,
)

# Reuse collection from exp128/exp127
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
from exp128_text_visual import _collect_zombie_walk_frames


# ---------------------------------------------------------------------------
# Phase 0: Load nav encoder for data collection
# ---------------------------------------------------------------------------

def phase0_nav_encoder():
    """Load exp128 encoder (grid_size=4) for navigation during data collection."""
    print("=" * 60)
    print("Phase 0: Load exp128 encoder for navigation")
    print("=" * 60)
    t0 = time.time()

    # Use exp128 encoder directly — it works for NearDetector (center 2×2)
    for tag in ["final", "phase1"]:
        path = EXP128_CHECKPOINT / tag / "encoder.pt"
        if path.exists():
            encoder = CNNEncoder(feature_channels=256, grid_size=4)
            state = torch.load(path, map_location="cpu", weights_only=True)
            encoder.load_state_dict(state, strict=False)
            encoder.eval()
            detector = NearDetector(encoder)
            print(f"  Loaded exp128 encoder from {path} ({time.time()-t0:.1f}s)")
            return encoder, detector

    # Fallback: train from scratch
    print("  WARNING: exp128 not found, training nav encoder from scratch...")
    nav_encoder, detector = phase0_load_nav_encoder()
    print(f"  Done ({time.time()-t0:.1f}s)")
    return nav_encoder, detector


# ---------------------------------------------------------------------------
# Phase 1: Collect labeled data (same as exp128)
# ---------------------------------------------------------------------------

def phase1_collect(detector: NearDetector, n_frames: int = 10000, n_episodes: int = 200) -> dict:
    """Collect pixel frames + semantic map GT from random walks.

    Random walks give diverse scenes. Semantic map provides per-tile GT
    for tile_head supervision during encoder training.
    """
    print("\n" + "=" * 60)
    print("Phase 1: Collect frames with semantic maps (random walks)")
    print("=" * 60)
    t0 = time.time()

    from snks.encoder.tile_head_trainer import semantic_cell_label
    from exp122_pixels import _detect_near_from_info

    GRID_SIZE = 8
    all_pixels: list[torch.Tensor] = []
    all_near_labels: list[int] = []
    all_tile_labels: list[torch.Tensor] = []
    frames_per_ep = max(1, n_frames // n_episodes)

    for ep in range(n_episodes):
        env = CrafterPixelEnv(seed=ep * 17 + 42)
        pixels, info = env.reset()

        for step in range(frames_per_ep * 3):
            action = np.random.randint(0, env.n_actions)
            pixels, _, done, info = env.step(action)
            if done:
                pixels, info = env.reset()

            if step % 3 != 0:
                continue

            semantic = info.get("semantic")
            if semantic is None:
                continue

            # Near label from GT semantic map (center object)
            near_str = _detect_near_from_info(info)
            near_idx = NEAR_TO_IDX.get(near_str, 0)

            # Tile labels: (H, W) GT class for each feature map position
            tile_gt = torch.zeros(GRID_SIZE, GRID_SIZE, dtype=torch.long)
            for gy in range(GRID_SIZE):
                for gx in range(GRID_SIZE):
                    tile_gt[gy, gx] = semantic_cell_label(semantic, gy, gx, GRID_SIZE)

            all_pixels.append(torch.from_numpy(pixels))
            all_near_labels.append(near_idx)
            all_tile_labels.append(tile_gt)

            if len(all_pixels) >= n_frames:
                break
        if len(all_pixels) >= n_frames:
            break

    pixels_t = torch.stack(all_pixels).float()
    near_labels_t = torch.tensor(all_near_labels, dtype=torch.long)
    tile_labels_t = torch.stack(all_tile_labels)  # (N, H, W)

    # Near label distribution
    counter = Counter(all_near_labels)
    print(f"  Collected {len(pixels_t)} frames ({time.time()-t0:.0f}s)")
    print(f"  Near label distribution:")
    for idx in sorted(counter.keys()):
        name = NEAR_CLASSES[idx] if idx < len(NEAR_CLASSES) else f"unk_{idx}"
        print(f"    {name}: {counter[idx]}")

    # Tile label distribution (across all positions)
    tile_flat = tile_labels_t.flatten().tolist()
    tile_counter = Counter(tile_flat)
    print(f"  Tile label distribution ({GRID_SIZE}×{GRID_SIZE} per frame):")
    for idx in sorted(tile_counter.keys()):
        name = NEAR_CLASSES[idx] if idx < len(NEAR_CLASSES) else f"unk_{idx}"
        print(f"    {name}: {tile_counter[idx]}")

    return {
        "pixels": pixels_t,
        "near_labels": near_labels_t,
        "tile_labels": tile_labels_t,
        "trained_classes": {NEAR_CLASSES[idx] for idx in counter.keys() if idx < len(NEAR_CLASSES)},
    }


def _run_chain_batch(runner, detector, labeler, chain, n_seeds, seed_base, label):
    """Run chain on n_seeds, return labeled frames."""
    all_labeled = []
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

    print(f"    {label}: {n_success}/{n_seeds} seeds, {len(all_labeled)} frames ({time.time()-t0:.0f}s)")
    return all_labeled


# ---------------------------------------------------------------------------
# Phase 2: Train new CNN encoder (grid_size=8)
# ---------------------------------------------------------------------------

def phase2_train_encoder(dataset: dict, epochs: int = 150) -> tuple[CNNEncoder, NearDetector]:
    """Train CNN encoder with grid_size=8."""
    print("\n" + "=" * 60)
    print("Phase 2: Train CNN encoder (grid_size=8)")
    print("=" * 60)
    t0 = time.time()

    N = len(dataset["pixels"])
    train_device = "cuda" if torch.cuda.is_available() else "cpu"
    if train_device == "cuda":
        disable_rocm_conv()

    pixels = dataset["pixels"]
    near_labels = dataset["near_labels"]

    # Build (t, t+1) pairs
    pixels_t = pixels[:-1]
    pixels_t1 = pixels[1:]
    actions = torch.zeros(N - 1, dtype=torch.long)
    nl = near_labels[:-1]

    tile_labels = dataset.get("tile_labels")

    # NEW: grid_size=8, feature_channels=256
    encoder = CNNEncoder(
        feature_channels=256,
        grid_size=8,
        n_near_classes=len(NEAR_CLASSES),
    )
    predictor = JEPAPredictor()

    trainer = PredictiveTrainer(
        encoder, predictor,
        contrastive_weight=1.0,
        near_weight=2.0,
        tile_weight=10.0,  # dominant per-tile supervision
        device=train_device,
    )

    print(f"  {N} frames, grid_size=8, feature_channels=256")
    print(f"  Feature map: (256, 8, 8) — each cell ≈ 1 Crafter tile")
    print(f"  tile_weight=10.0 + class-weighted CE — dominant per-tile supervision")
    print(f"  Training on {train_device}...")

    # tile_labels for (t, t+1) pairs: use labels from frame t
    tl = tile_labels[:-1] if tile_labels is not None else None

    history = trainer.train_full(
        pixels_t, pixels_t1, actions,
        situation_labels=nl,
        near_labels=nl,
        tile_labels=tl,
        epochs=epochs,
        batch_size=min(64, N - 1),
        log_every=30,
    )

    encoder.eval().cpu()
    detector = NearDetector(encoder)

    final = history[-1]
    tile_final = final.get('tile_loss', 0)
    print(f"  Done: pred={final['pred_loss']:.4f} near={final['near_loss']:.4f} "
          f"tile={tile_final:.4f} ({time.time()-t0:.0f}s)")

    # Save encoder checkpoint
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(encoder.state_dict(), CHECKPOINT_DIR / "encoder_g8.pt")
    print(f"  Saved → {CHECKPOINT_DIR / 'encoder_g8.pt'}")

    # Diagnostic: tile accuracy from joint training (before Phase 3 re-trains)
    print("  Diagnostic: tile accuracy from joint training...")
    joint_acc = phase4_accuracy_gate(encoder, n_frames=300)
    print(f"  Joint-trained tile accuracy: {joint_acc:.1%}")

    return encoder, detector


# ---------------------------------------------------------------------------
# Phase 3: Train tile_head on new encoder
# ---------------------------------------------------------------------------

def phase3_train_tile_head(encoder: CNNEncoder, n_frames: int = 5000) -> dict:
    """Collect features from new encoder and train tile_head."""
    print("\n" + "=" * 60)
    print("Phase 3: Train tile_head (grid_size=8)")
    print("=" * 60)
    t0 = time.time()

    features, labels = collect_tile_training_data(
        encoder, n_frames=n_frames, n_episodes=100,
    )

    stats = train_tile_head(
        encoder, features, labels,
        epochs=80, lr=1e-3, batch_size=512,
    )

    print(f"  Best acc: {stats['train_acc']:.1%} ({time.time()-t0:.1f}s)")

    # Save updated encoder with tile_head
    torch.save(encoder.state_dict(), CHECKPOINT_DIR / "encoder_g8_tile.pt")
    print(f"  Saved → {CHECKPOINT_DIR / 'encoder_g8_tile.pt'}")

    return stats


# ---------------------------------------------------------------------------
# Phase 4: Accuracy gate
# ---------------------------------------------------------------------------

def phase4_accuracy_gate(encoder: CNNEncoder, n_frames: int = 500) -> float:
    """Test tile_head accuracy on held-out data."""
    print("\n" + "=" * 60)
    print("Phase 4: Tile accuracy gate (≥60%)")
    print("=" * 60)

    correct = 0
    total = 0
    grid_size = encoder.grid_size
    per_class_correct: dict[int, int] = {}
    per_class_total: dict[int, int] = {}

    for ep in range(20):
        env = CrafterPixelEnv(seed=ep * 31 + 100)
        pixels, info = env.reset()

        for step in range(n_frames // 20):
            action = np.random.randint(0, env.n_actions)
            pixels, _, done, info = env.step(action)
            if done:
                break

            semantic = info.get("semantic")
            if semantic is None:
                continue

            px_tensor = torch.from_numpy(pixels)
            class_ids, _ = encoder.classify_tiles(px_tensor)

            H, W = class_ids.shape
            for gy in range(H):
                for gx in range(W):
                    gt = semantic_cell_label(semantic, gy, gx, grid_size)
                    pred = int(class_ids[gy, gx].item())
                    per_class_total[gt] = per_class_total.get(gt, 0) + 1
                    if gt == pred:
                        correct += 1
                        per_class_correct[gt] = per_class_correct.get(gt, 0) + 1
                    total += 1

    acc = correct / max(1, total)
    print(f"  Accuracy: {acc:.1%} ({correct}/{total})")
    print(f"  Per-class accuracy:")
    for cls_idx in sorted(per_class_total.keys()):
        name = NEAR_CLASSES[cls_idx] if cls_idx < len(NEAR_CLASSES) else f"unk_{cls_idx}"
        cls_correct = per_class_correct.get(cls_idx, 0)
        cls_total = per_class_total[cls_idx]
        cls_acc = cls_correct / max(1, cls_total)
        print(f"    {name}: {cls_acc:.1%} ({cls_correct}/{cls_total})")
    print(f"  {'PASS' if acc >= 0.60 else 'FAIL'}: {'≥' if acc >= 0.60 else '<'}60%")
    return acc


# ---------------------------------------------------------------------------
# Phase 5: Smoke test — wood collection
# ---------------------------------------------------------------------------

def phase5_smoke(encoder: CNNEncoder, n_episodes: int = 20, max_steps: int = 200) -> dict:
    """Wood collection with tile perception."""
    print("\n" + "=" * 60)
    print("Phase 5: Smoke test — wood collection")
    print("=" * 60)

    store = ConceptStore()
    tb = CrafterTextbook("configs/crafter_textbook.yaml")
    tb.load_into(store)
    labeler = OutcomeLabeler()

    grid_size = encoder.grid_size
    center = grid_size // 2  # grid center ≈ 4 for grid_size=8

    wood_collected = []
    steps_to_3wood = []

    for ep in range(n_episodes):
        env = CrafterPixelEnv(seed=ep * 7 + 200)
        try:
            env._env._balance_chunk = lambda *a, **kw: None
        except Exception:
            pass

        pixels, info = env.reset()
        rng = np.random.RandomState(ep)
        spatial_map = CrafterSpatialMap()
        wood = 0
        found_3wood_step = None

        for step in range(max_steps):
            inv = dict(info.get("inventory", {}))
            player_pos = info.get("player_pos", (32, 32))

            px_tensor = torch.from_numpy(pixels)
            vf = perceive_tile_field(px_tensor, encoder)

            # Update spatial map
            spatial_map.update(player_pos, vf.near_concept)
            for cid, conf, gy, gx in vf.detections:
                wx = int(player_pos[0]) + (gx - center)
                wy = int(player_pos[1]) + (gy - center)
                spatial_map.update((wx, wy), cid)

            # Simple policy: find tree, do
            if vf.near_concept == "tree":
                action_str = "do"
            else:
                tree_pos = spatial_map.find_nearest("tree", player_pos)
                if tree_pos:
                    action_str = _step_toward(player_pos, tree_pos, rng)
                else:
                    tree_dets = vf.find("tree")
                    if tree_dets:
                        _, tgy, tgx = tree_dets[0]
                        if tgx < center - 1:
                            action_str = "move_left"
                        elif tgx > center:
                            action_str = "move_right"
                        elif tgy < center - 1:
                            action_str = "move_up"
                        elif tgy > center:
                            action_str = "move_down"
                        else:
                            action_str = "do"
                    else:
                        action_str = str(rng.choice(MOVE_ACTIONS))

            inv_before = inv
            pixels, _, done, info = env.step(action_str)
            inv_after = dict(info.get("inventory", {}))

            new_wood = inv_after.get("wood", 0)
            if new_wood > wood:
                wood = new_wood
                if wood >= 3 and found_3wood_step is None:
                    found_3wood_step = step

            if vf.raw_center_feature is not None:
                on_action_outcome(action_str, inv_before, inv_after,
                                  vf.raw_center_feature, store, labeler, encoder)

            if done:
                break

        wood_collected.append(wood)
        if found_3wood_step is not None:
            steps_to_3wood.append(found_3wood_step)

    avg_wood = sum(wood_collected) / len(wood_collected)
    pct_3wood = len(steps_to_3wood) / n_episodes

    print(f"  Avg wood: {avg_wood:.1f}")
    print(f"  ≥3 wood: {pct_3wood:.0%} ({len(steps_to_3wood)}/{n_episodes})")
    if steps_to_3wood:
        print(f"  Avg steps to 3 wood: {sum(steps_to_3wood)/len(steps_to_3wood):.0f}")

    return {"avg_wood": avg_wood, "pct_3wood": pct_3wood}


# ---------------------------------------------------------------------------
# Phase 6: Survival with enemies
# ---------------------------------------------------------------------------

def phase6_survival(encoder: CNNEncoder, n_episodes: int = 20, max_steps: int = 500) -> dict:
    """Survival eval with homeostatic drives."""
    print("\n" + "=" * 60)
    print("Phase 6: Survival with enemies")
    print("=" * 60)

    store = ConceptStore()
    tb = CrafterTextbook("configs/crafter_textbook.yaml")
    tb.load_into(store)
    labeler = OutcomeLabeler()
    tracker = HomeostaticTracker()

    body_rules = tb.get_body_rules() if hasattr(tb, "get_body_rules") else []
    if body_rules:
        tracker.init_from_body_rules(body_rules)

    grid_size = encoder.grid_size
    center = grid_size // 2

    episode_lengths = []
    resources = Counter()

    for ep in range(n_episodes):
        env = CrafterPixelEnv(seed=ep * 11 + 300)
        pixels, info = env.reset()
        rng = np.random.RandomState(ep)
        spatial_map = CrafterSpatialMap()
        prev_inv: dict = {}

        for step in range(max_steps):
            inv = dict(info.get("inventory", {}))
            player_pos = info.get("player_pos", (32, 32))

            px_tensor = torch.from_numpy(pixels)
            vf = perceive_tile_field(px_tensor, encoder)

            spatial_map.update(player_pos, vf.near_concept)
            for cid, conf, gy, gx in vf.detections:
                wx = int(player_pos[0]) + (gx - center)
                wy = int(player_pos[1]) + (gy - center)
                spatial_map.update((wx, wy), cid)

            if prev_inv:
                tracker.update(prev_inv, inv, vf.visible_concepts())

            goal, plan = select_goal(inv, store, tracker, vf, spatial_map)

            if plan:
                step_plan = plan[0]
                target = step_plan.target
                if vf.near_concept == target:
                    action_str = step_plan.action
                else:
                    target_pos = spatial_map.find_nearest(target, player_pos)
                    if target_pos:
                        action_str = _step_toward(player_pos, target_pos, rng)
                    else:
                        target_dets = vf.find(target)
                        if target_dets:
                            _, tgy, tgx = target_dets[0]
                            if tgx < center - 1:
                                action_str = "move_left"
                            elif tgx > center:
                                action_str = "move_right"
                            elif tgy < center - 1:
                                action_str = "move_up"
                            elif tgy > center:
                                action_str = "move_down"
                            else:
                                action_str = "do"
                        else:
                            action_str = explore_action(rng, store, inv)
                if action_str.startswith("babble_"):
                    action_str = action_str.replace("babble_", "")
            else:
                action_str = explore_action(rng, store, inv)
                if action_str.startswith("babble_"):
                    action_str = action_str.replace("babble_", "")

            inv_before = inv
            pixels, _, done, info = env.step(action_str)
            inv_after = dict(info.get("inventory", {}))

            if vf.raw_center_feature is not None:
                label = on_action_outcome(action_str, inv_before, inv_after,
                                          vf.raw_center_feature, store, labeler, encoder)
                if label:
                    resources[label] += 1

            out = outcome_to_verify(action_str, inv_before, inv_after)
            if out:
                verify_outcome(vf.near_concept, action_str, out, store)

            prev_inv = inv
            if done:
                episode_lengths.append(step + 1)
                break
        else:
            episode_lengths.append(max_steps)

    avg_len = sum(episode_lengths) / len(episode_lengths)
    print(f"  Avg episode length: {avg_len:.0f}")
    print(f"  Resources: {dict(resources)}")

    return {"avg_episode_length": avg_len, "resources": dict(resources)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()

    # Phase 0 — nav encoder for potential future use, but Phase 1 uses random walks
    # nav_encoder, detector = phase0_nav_encoder()

    # Phase 1 — random walks with semantic map GT (no nav encoder needed)
    dataset = phase1_collect(None, n_frames=10000, n_episodes=200)

    # Phase 2 (trains encoder + tile_head jointly)
    encoder, detector_new = phase2_train_encoder(dataset, epochs=150)

    # Phase 3: optional re-training (skip if joint training already good)
    # Joint accuracy already measured in Phase 2 diagnostic
    # tile_stats = phase3_train_tile_head(encoder, n_frames=10000)

    # Phase 4: final accuracy gate
    tile_acc = phase4_accuracy_gate(encoder)

    # Phase 5
    smoke = phase5_smoke(encoder, n_episodes=20, max_steps=200)

    # Phase 6
    survival = phase6_survival(encoder, n_episodes=20, max_steps=500)

    # Final save
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(encoder.state_dict(), CHECKPOINT_DIR / "encoder_final.pt")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Encoder: grid_size=8, feature_channels=256")
    print(f"  Tile head accuracy: {tile_acc:.1%}")
    print(f"  Smoke: avg wood={smoke['avg_wood']:.1f}, ≥3 wood={smoke['pct_3wood']:.0%}")
    print(f"  Survival: {survival['avg_episode_length']:.0f} steps")
    print(f"  Total time: {time.time()-t_start:.0f}s")

    gates = {
        "tile_acc_60pct": tile_acc >= 0.60,
        "wood_3_50pct": smoke["pct_3wood"] >= 0.50,
        "survival_200": survival["avg_episode_length"] >= 200,
    }
    print(f"\n  Gates: {gates}")
    all_pass = all(gates.values())
    print(f"  {'ALL PASS' if all_pass else 'SOME FAIL'}")

    return all_pass


if __name__ == "__main__":
    main()
