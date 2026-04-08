"""Diagnostic: verify stale map hypothesis.

Track per step:
- Where agent navigates to (spatial_map.find_nearest target)
- Whether that position actually has the object
- How many times agent revisits same position
- What perceive_field returns vs what's actually there
"""

import numpy as np
import torch
from collections import Counter
from pathlib import Path

from snks.encoder.cnn_encoder import CNNEncoder, disable_rocm_conv
from snks.agent.concept_store import ConceptStore
from snks.agent.crafter_textbook import CrafterTextbook
from snks.agent.crafter_pixel_env import CrafterPixelEnv
from snks.agent.crafter_spatial_map import CrafterSpatialMap, _step_toward, MOVE_ACTIONS
from snks.agent.outcome_labeler import OutcomeLabeler
from snks.agent.perception import (
    perceive_field, HomeostaticTracker,
    on_action_outcome, select_goal, explore_action,
    ground_empty_on_start, ground_zombie_on_damage,
)

_DIRECTIONS = ["move_up", "move_down", "move_left", "move_right"]


def main():
    disable_rocm_conv()
    print("NAVIGATION DIAGNOSTIC: stale map hypothesis")
    print("=" * 60)

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
                print(f"Loaded {ch}ch from {path}")
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
    device = next(encoder.parameters()).device

    # Bootstrap
    for i in range(10):
        env = CrafterPixelEnv(seed=60000 + i * 7)
        pixels, info = env.reset()
        rng = np.random.RandomState(60000 + i * 7)
        sm = CrafterSpatialMap()
        pt = torch.from_numpy(pixels).float()
        if device.type != "cpu":
            pt = pt.to(device)
        ground_empty_on_start(pt, encoder, store)
        for step in range(300):
            pt = torch.from_numpy(pixels).float()
            if device.type != "cpu":
                pt = pt.to(device)
            vf = perceive_field(pt, encoder, store)
            pp = info.get("player_pos", (32, 32))
            sm.update(pp, vf.near_concept)
            act = explore_action(rng, store, dict(info.get("inventory", {})))
            if act == "babble_do":
                d = _DIRECTIONS[rng.randint(0, 4)]
                pixels, _, done, info = env.step(d)
                if done: break
                pt2 = torch.from_numpy(pixels).float()
                if device.type != "cpu":
                    pt2 = pt2.to(device)
                vf2 = perceive_field(pt2, encoder, store)
                oi = dict(info.get("inventory", {}))
                pixels, _, done, info = env.step("do")
                if done: break
                ni = dict(info.get("inventory", {}))
                if vf2.center_feature is not None:
                    on_action_outcome("do", oi, ni, vf2.center_feature, store, labeler)
            else:
                if act.startswith("babble_"):
                    act = _DIRECTIONS[rng.randint(0, 4)]
                pixels, _, done, info = env.step(act)
                if done: break

    grounded = [c.id for c in store.concepts.values() if c.visual is not None]
    print(f"Grounded: {grounded}\n")

    # Now trace 5 detailed episodes
    for ep in range(5):
        seed = 90000 + ep * 7
        env = CrafterPixelEnv(seed=seed)
        pixels, info = env.reset()
        rng = np.random.RandomState(seed)
        sm = CrafterSpatialMap()

        pt = torch.from_numpy(pixels).float()
        if device.type != "cpu":
            pt = pt.to(device)
        ground_empty_on_start(pt, encoder, store)

        nav_to_stale = 0
        nav_to_valid = 0
        nav_to_unknown = 0
        tree_positions_visited = set()
        wood_count = 0
        prev_inv = {}

        print(f"--- Episode {ep+1} (seed={seed}) ---")

        for step in range(500):
            inv = dict(info.get("inventory", {}))
            pt = torch.from_numpy(pixels).float()
            if device.type != "cpu":
                pt = pt.to(device)
            vf = perceive_field(pt, encoder, store)
            near = vf.near_concept
            pp = info.get("player_pos", (32, 32))
            sm.update(pp, near)

            if prev_inv:
                tracker.update(prev_inv, inv, vf.visible_concepts())
                ground_zombie_on_damage(prev_inv, inv, vf, store)

            # Track wood
            w_now = inv.get("wood", 0)
            w_prev = prev_inv.get("wood", 0)
            if w_now > w_prev:
                wood_count += (w_now - w_prev)
                print(f"  [{step}] GOT WOOD #{wood_count} at {pp}")

            prev_inv = dict(inv)

            # What does spatial map say about tree?
            tree_pos = sm.find_nearest("tree", pp)

            # Check ground truth (semantic map)
            semantic = info.get("semantic")
            gt_near = None
            if semantic is not None:
                py, px = int(pp[0]), int(pp[1])
                if 0 <= py < semantic.shape[0] and 0 <= px < semantic.shape[1]:
                    gt_near_id = int(semantic[py, px])
                    from snks.agent.crafter_pixel_env import SEMANTIC_NAMES
                    gt_near = SEMANTIC_NAMES.get(gt_near_id, "?")

            # If navigating to tree_pos, check if it's still valid
            if tree_pos and step % 50 == 0:
                ty, tx = tree_pos
                gt_at_target = None
                if semantic is not None and 0 <= ty < semantic.shape[0] and 0 <= tx < semantic.shape[1]:
                    gt_at_target_id = int(semantic[ty, tx])
                    gt_at_target = SEMANTIC_NAMES.get(gt_at_target_id, "?")
                print(f"  [{step}] map says tree@{tree_pos}, GT@target={gt_at_target}, "
                      f"near={near}, GT@here={gt_near}, "
                      f"map_trees={sm._map.values().__class__}")

                # Count how many "tree" entries in map are actually trees in GT
                stale = 0
                valid = 0
                for (my, mx), label in sm._map.items():
                    if label == "tree":
                        if (semantic is not None and
                            0 <= my < semantic.shape[0] and
                            0 <= mx < semantic.shape[1]):
                            gt_id = int(semantic[my, mx])
                            gt_label = SEMANTIC_NAMES.get(gt_id, "?")
                            if gt_label == "tree":
                                valid += 1
                            else:
                                stale += 1
                if stale + valid > 0:
                    print(f"         map tree entries: {valid} valid, {stale} stale "
                          f"({stale/(stale+valid)*100:.0f}% stale)")

            # Simple action execution
            goal, plan = select_goal(inv, store, tracker=tracker, visual_field=vf, spatial_map=sm)
            if plan and plan[0].target == "tree":
                if tree_pos:
                    action = _step_toward(pp, tree_pos, rng)
                    pixels, _, done, info = env.step(action)
                    if done: break
                else:
                    act = explore_action(rng, store, inventory=inv)
                    if act == "babble_do":
                        d = _DIRECTIONS[rng.randint(0, 4)]
                        pixels, _, done, info = env.step(d)
                        if done: break
                        pixels, _, done, info = env.step("do")
                        if done: break
                    else:
                        pixels, _, done, info = env.step(act)
                        if done: break
            else:
                act = explore_action(rng, store, inventory=inv)
                if act == "babble_do":
                    d = _DIRECTIONS[rng.randint(0, 4)]
                    pixels, _, done, info = env.step(d)
                    if done: break
                    pt2 = torch.from_numpy(pixels).float()
                    if device.type != "cpu":
                        pt2 = pt2.to(device)
                    vf2 = perceive_field(pt2, encoder, store)
                    oi = dict(info.get("inventory", {}))
                    pixels, _, done, info = env.step("do")
                    if done: break
                    ni = dict(info.get("inventory", {}))
                    if vf2.center_feature is not None:
                        on_action_outcome("do", oi, ni, vf2.center_feature, store, labeler)
                else:
                    pixels, _, done, info = env.step(act)
                    if done: break

        cause = "starvation" if inv.get("food", 0) == 0 or inv.get("drink", 0) == 0 else "zombie"
        print(f"  DIED at step {step+1}: {cause}, wood={wood_count}\n")


if __name__ == "__main__":
    main()
