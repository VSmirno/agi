"""exp134: Stage 75 — Per-tile visual field via tile_head.

tile_head = Linear(256→12) trained on each feature map position.
Semantic map as teacher (training only). At runtime: classify_tiles() → full screen.
Agent names class_ids through interaction (ConceptStore).

Pipeline:
  Phase 0: Load frozen encoder from exp128
  Phase 1: Collect training data (frames + semantic map GT)
  Phase 2: Train tile_head
  Phase 3: Accuracy gate (≥60% per-tile)
  Phase 4: Smoke test — wood collection with tile perception
  Phase 5: Survival eval with enemies

Design: docs/superpowers/specs/2026-04-08-stage75-per-tile-visual-field-design.md
"""

from __future__ import annotations

import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

CHECKPOINT_DIR = Path("demos/checkpoints/exp134")
EXP128_CHECKPOINT = Path("demos/checkpoints/exp128")

from snks.encoder.cnn_encoder import CNNEncoder
from snks.encoder.tile_head_trainer import (
    collect_tile_training_data,
    train_tile_head,
    semantic_cell_label,
)
from snks.agent.concept_store import ConceptStore
from snks.agent.crafter_textbook import CrafterTextbook
from snks.agent.crafter_pixel_env import CrafterPixelEnv, SEMANTIC_NAMES
from snks.agent.crafter_spatial_map import CrafterSpatialMap, _step_toward, MOVE_ACTIONS
from snks.agent.outcome_labeler import OutcomeLabeler
from snks.agent.decode_head import NEAR_CLASSES
from snks.agent.perception import (
    perceive_tile_field,
    HomeostaticTracker,
    on_action_outcome,
    select_goal,
    explore_action,
    verify_outcome,
    outcome_to_verify,
)


# ---------------------------------------------------------------------------
# Phase 0: Load frozen encoder
# ---------------------------------------------------------------------------

def phase0_load_encoder() -> CNNEncoder:
    print("=" * 60)
    print("Phase 0: Load frozen encoder from exp128")
    print("=" * 60)
    t0 = time.time()

    for tag in ["final", "phase1"]:
        path = EXP128_CHECKPOINT / tag / "encoder.pt"
        if path.exists():
            encoder = CNNEncoder(feature_channels=256)
            state = torch.load(path, map_location="cpu", weights_only=True)
            # Load only conv/proj/ln/near_head weights, skip tile_head (new)
            missing, unexpected = encoder.load_state_dict(state, strict=False)
            encoder.eval()
            print(f"  Loaded from {path} ({time.time()-t0:.1f}s)")
            if missing:
                print(f"  Missing (expected for new tile_head): {missing}")
            return encoder

    raise FileNotFoundError(f"No encoder in {EXP128_CHECKPOINT}")


# ---------------------------------------------------------------------------
# Phase 1: Collect training data
# ---------------------------------------------------------------------------

def phase1_collect(encoder: CNNEncoder, n_frames: int = 5000) -> tuple[torch.Tensor, torch.Tensor]:
    print("\n" + "=" * 60)
    print("Phase 1: Collect tile training data")
    print("=" * 60)
    t0 = time.time()

    features, labels = collect_tile_training_data(
        encoder, n_frames=n_frames, n_episodes=100,
    )

    print(f"  Total: {len(features)} samples ({time.time()-t0:.1f}s)")
    return features, labels


# ---------------------------------------------------------------------------
# Phase 2: Train tile_head
# ---------------------------------------------------------------------------

def phase2_train(
    encoder: CNNEncoder,
    features: torch.Tensor,
    labels: torch.Tensor,
    epochs: int = 80,
) -> dict:
    print("\n" + "=" * 60)
    print("Phase 2: Train tile_head")
    print("=" * 60)
    t0 = time.time()

    stats = train_tile_head(
        encoder, features, labels,
        epochs=epochs, lr=1e-3, batch_size=512,
    )

    print(f"  Best acc: {stats['train_acc']:.1%} ({time.time()-t0:.1f}s)")

    # Save checkpoint
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(encoder.state_dict(), CHECKPOINT_DIR / "encoder_with_tile_head.pt")
    print(f"  Saved to {CHECKPOINT_DIR / 'encoder_with_tile_head.pt'}")

    return stats


# ---------------------------------------------------------------------------
# Phase 3: Accuracy gate
# ---------------------------------------------------------------------------

def phase3_accuracy_gate(encoder: CNNEncoder, n_frames: int = 500) -> float:
    print("\n" + "=" * 60)
    print("Phase 3: Accuracy gate (≥60% per-tile)")
    print("=" * 60)

    correct = 0
    total = 0

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
                    gt = semantic_cell_label(semantic, gy, gx, H)
                    pred = int(class_ids[gy, gx].item())
                    if gt == pred:
                        correct += 1
                    total += 1

    acc = correct / max(1, total)
    print(f"  Accuracy: {acc:.1%} ({correct}/{total})")

    if acc >= 0.60:
        print("  PASS: ≥60%")
    else:
        print(f"  FAIL: {acc:.1%} < 60%")

    return acc


# ---------------------------------------------------------------------------
# Phase 4: Smoke test — wood collection with tile perception
# ---------------------------------------------------------------------------

def phase4_smoke_test(
    encoder: CNNEncoder,
    n_episodes: int = 20,
    max_steps: int = 200,
) -> dict:
    print("\n" + "=" * 60)
    print("Phase 4: Smoke test — wood collection")
    print("=" * 60)

    store = ConceptStore()
    tb = CrafterTextbook("configs/crafter_textbook.yaml")
    tb.load_into(store)
    labeler = OutcomeLabeler()
    tracker = HomeostaticTracker()

    # Init body rules from textbook
    body_rules = tb.get_body_rules() if hasattr(tb, "get_body_rules") else []
    if body_rules:
        tracker.init_from_body_rules(body_rules)

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

            # Perceive via tile_head
            px_tensor = torch.from_numpy(pixels)
            vf = perceive_tile_field(px_tensor, encoder)

            # Update spatial map from ALL detections (full screen)
            spatial_map.update(player_pos, vf.near_concept)
            for cid, conf, gy, gx in vf.detections:
                wx = int(player_pos[0]) + (gx - 2)  # grid center ~2 for 4×4
                wy = int(player_pos[1]) + (gy - 2)
                spatial_map.update((wx, wy), cid)

            # Simple policy: navigate to tree, do
            if vf.near_concept == "tree":
                action_str = "do"
            else:
                tree_pos = spatial_map.find_nearest("tree", player_pos)
                if tree_pos:
                    action_str = _step_toward(player_pos, tree_pos, rng)
                else:
                    # Look for tree in current visual field
                    tree_dets = vf.find("tree")
                    if tree_dets:
                        # Navigate toward detected tree
                        _, tgy, tgx = tree_dets[0]
                        # Convert grid offset to direction
                        if tgx < 1:
                            action_str = "move_left"
                        elif tgx > 2:
                            action_str = "move_right"
                        elif tgy < 1:
                            action_str = "move_up"
                        elif tgy > 2:
                            action_str = "move_down"
                        else:
                            action_str = "do"
                    else:
                        action_str = str(rng.choice(MOVE_ACTIONS))

            inv_before = inv
            pixels, _, done, info = env.step(action_str)
            inv_after = dict(info.get("inventory", {}))

            # Track wood
            new_wood = inv_after.get("wood", 0)
            if new_wood > wood:
                wood = new_wood
                if wood >= 3 and found_3wood_step is None:
                    found_3wood_step = step

            # Grounding from interaction
            if vf.raw_center_feature is not None:
                on_action_outcome(
                    action_str, inv_before, inv_after,
                    vf.raw_center_feature, store, labeler, encoder,
                )

            if done:
                break

        wood_collected.append(wood)
        if found_3wood_step is not None:
            steps_to_3wood.append(found_3wood_step)

    avg_wood = sum(wood_collected) / len(wood_collected)
    pct_3wood = len(steps_to_3wood) / n_episodes
    avg_steps_3w = sum(steps_to_3wood) / max(1, len(steps_to_3wood))

    print(f"  Avg wood: {avg_wood:.1f}")
    print(f"  ≥3 wood: {pct_3wood:.0%} ({len(steps_to_3wood)}/{n_episodes})")
    if steps_to_3wood:
        print(f"  Avg steps to 3 wood: {avg_steps_3w:.0f}")
    print(f"  Grounded concepts: {[c.id for c in store.concepts.values() if c.visual is not None]}")

    return {
        "avg_wood": avg_wood,
        "pct_3wood": pct_3wood,
        "avg_steps_3wood": avg_steps_3w if steps_to_3wood else None,
    }


# ---------------------------------------------------------------------------
# Phase 5: Survival eval with enemies
# ---------------------------------------------------------------------------

def phase5_survival(
    encoder: CNNEncoder,
    n_episodes: int = 20,
    max_steps: int = 500,
) -> dict:
    print("\n" + "=" * 60)
    print("Phase 5: Survival with enemies")
    print("=" * 60)

    store = ConceptStore()
    tb = CrafterTextbook("configs/crafter_textbook.yaml")
    tb.load_into(store)
    labeler = OutcomeLabeler()
    tracker = HomeostaticTracker()

    body_rules = tb.get_body_rules() if hasattr(tb, "get_body_rules") else []
    if body_rules:
        tracker.init_from_body_rules(body_rules)

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

            # Perceive via tile_head
            px_tensor = torch.from_numpy(pixels)
            vf = perceive_tile_field(px_tensor, encoder)

            # Update spatial map
            spatial_map.update(player_pos, vf.near_concept)
            for cid, conf, gy, gx in vf.detections:
                wx = int(player_pos[0]) + (gx - 2)
                wy = int(player_pos[1]) + (gy - 2)
                spatial_map.update((wx, wy), cid)

            # Homeostatic tracking
            if prev_inv:
                tracker.update(prev_inv, inv, vf.visible_concepts())

            # Goal selection from drives
            goal, plan = select_goal(inv, store, tracker, vf, spatial_map)

            # Action selection
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
                        # Check visual field for target
                        target_dets = vf.find(target)
                        if target_dets:
                            _, tgy, tgx = target_dets[0]
                            if tgx < 1:
                                action_str = "move_left"
                            elif tgx > 2:
                                action_str = "move_right"
                            elif tgy < 1:
                                action_str = "move_up"
                            elif tgy > 2:
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

            # Grounding
            if vf.raw_center_feature is not None:
                label = on_action_outcome(
                    action_str, inv_before, inv_after,
                    vf.raw_center_feature, store, labeler, encoder,
                )
                if label:
                    resources[label] += 1

            # Verification
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
    print(f"  Grounded: {[c.id for c in store.concepts.values() if c.visual is not None]}")

    return {
        "avg_episode_length": avg_len,
        "resources": dict(resources),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()

    # Phase 0
    encoder = phase0_load_encoder()

    # Phase 1
    features, labels = phase1_collect(encoder, n_frames=5000)

    # Phase 2
    stats = phase2_train(encoder, features, labels, epochs=80)

    # Phase 3
    acc = phase3_accuracy_gate(encoder, n_frames=500)

    # Phase 4
    smoke = phase4_smoke_test(encoder, n_episodes=20, max_steps=200)

    # Phase 5
    survival = phase5_survival(encoder, n_episodes=20, max_steps=500)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Tile head accuracy: {acc:.1%}")
    print(f"  Smoke: avg wood={smoke['avg_wood']:.1f}, "
          f"≥3 wood={smoke['pct_3wood']:.0%}")
    if smoke["avg_steps_3wood"]:
        print(f"  Steps to 3 wood: {smoke['avg_steps_3wood']:.0f}")
    print(f"  Survival: {survival['avg_episode_length']:.0f} steps")
    print(f"  Total time: {time.time()-t_start:.0f}s")

    # Gates
    gates = {
        "tile_acc_60pct": acc >= 0.60,
        "wood_3_50pct": smoke["pct_3wood"] >= 0.50,
        "survival_200": survival["avg_episode_length"] >= 200,
    }
    print(f"\n  Gates: {gates}")
    all_pass = all(gates.values())
    print(f"  {'ALL PASS' if all_pass else 'SOME FAIL'}")

    return all_pass


if __name__ == "__main__":
    main()
