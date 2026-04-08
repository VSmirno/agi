"""Step-by-step visual diagnostic: save frames, show what agent sees and does."""

import numpy as np
from pathlib import Path
from PIL import Image

from snks.encoder.cnn_encoder import CNNEncoder, disable_rocm_conv
from snks.agent.crafter_pixel_env import CrafterPixelEnv, SEMANTIC_NAMES
from snks.agent.concept_store import ConceptStore
from snks.agent.crafter_textbook import CrafterTextbook
from snks.agent.crafter_spatial_map import CrafterSpatialMap, _step_toward
from snks.agent.outcome_labeler import OutcomeLabeler
from snks.agent.patch_perception import (
    PatchStore, extract_facing_patch, detect_collision, TILE_SIZE, VIEW_COLS, VIEW_ROWS,
)
from snks.agent.perception import HomeostaticTracker, select_goal

import torch

_DIRECTIONS = ["move_up", "move_down", "move_left", "move_right"]
OUT_DIR = Path("_docs/stepbystep")


def save_frame(pixels, step, label, extra_info=""):
    """Save frame as PNG with label."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # pixels: (3, 64, 64) float [0,1]
    img_array = (pixels.transpose(1, 2, 0) * 255).astype(np.uint8)
    # Scale up 4x for visibility
    img = Image.fromarray(img_array)
    img = img.resize((256, 256), Image.NEAREST)
    path = OUT_DIR / f"step_{step:04d}_{label}.png"
    img.save(path)
    return path


def main():
    disable_rocm_conv()
    print("STEP-BY-STEP VISUAL DIAGNOSTIC")
    print("=" * 60)

    # Load trained encoder
    encoder = None
    for ckpt_dir, kwargs in [
        (Path("demos/checkpoints/exp128"), {"feature_channels": 256, "grid_size": 4}),
    ]:
        for tag in ["final", "phase3"]:
            path = ckpt_dir / tag / "encoder.pt"
            if path.exists():
                encoder = CNNEncoder(**kwargs)
                encoder.load_state_dict(torch.load(path, weights_only=True))
                encoder.eval()
                print(f"Loaded encoder from {path}")
                break
        if encoder:
            break

    # Init
    store = ConceptStore()
    tb = CrafterTextbook("configs/crafter_textbook.yaml")
    tb.load_into(store)
    tracker = HomeostaticTracker()
    tracker.init_from_body_rules(tb.body_rules)
    patch_store = PatchStore()
    labeler = OutcomeLabeler()

    # Run ONE episode, save every 5th frame
    seed = 42
    env = CrafterPixelEnv(seed=seed)
    pixels, info = env.reset()
    rng = np.random.RandomState(seed)
    spatial_map = CrafterSpatialMap()

    # Save initial frame
    save_frame(pixels, 0, "start")

    prev_inv = {}
    current_goal = ""
    current_plan = []
    plan_step_idx = 0
    replan_counter = 0
    last_direction = "move_right"
    wood_count = 0

    for step in range(200):
        inv = dict(info.get("inventory", {}))
        pp = info.get("player_pos", (32, 32))
        semantic = info.get("semantic")

        # GT: what's around player
        gt_around = {}
        if semantic is not None:
            for d, (ddx, ddy) in {"right":(1,0),"left":(-1,0),"down":(0,1),"up":(0,-1)}.items():
                tx, ty = int(pp[0])+ddx, int(pp[1])+ddy
                if 0 <= ty < 64 and 0 <= tx < 64:
                    gt_around[d] = SEMANTIC_NAMES.get(int(semantic[ty, tx]), "?")

        # Patch detection
        detections = patch_store.match_all_visible(pixels)
        det_labels = {(r,c): l for l, r, c, d in detections}

        # CNN near_head
        near_head_pred = None
        if encoder is not None:
            with torch.no_grad():
                out = encoder(torch.from_numpy(pixels).float().unsqueeze(0))
                from snks.agent.decode_head import NEAR_CLASSES
                probs = torch.softmax(out.near_logits, dim=1).squeeze(0)
                top = probs.topk(1)
                near_head_pred = NEAR_CLASSES[top.indices[0].item()]

        # HomeostaticTracker
        if prev_inv:
            tracker.update(prev_inv, inv, {d[0] for d in detections})

        # Goal
        replan_counter += 1
        if not current_plan or plan_step_idx >= len(current_plan) or \
           (current_goal == "explore" and replan_counter >= 20):
            replan_counter = 0
            current_goal, current_plan = select_goal(
                inv, store, tracker=tracker, spatial_map=spatial_map)
            plan_step_idx = 0

        # What plan step
        plan_target = None
        if current_plan and plan_step_idx < len(current_plan):
            plan_target = f"{current_plan[plan_step_idx].action} {current_plan[plan_step_idx].target}"

        # Wood tracking
        w = inv.get("wood", 0)
        wp = prev_inv.get("wood", 0)
        if w > wp:
            wood_count += (w - wp)
        prev_inv = dict(inv)

        # Print state
        health = inv.get("health", 9)
        food = inv.get("food", 9)
        drink = inv.get("drink", 9)
        energy = inv.get("energy", 9)

        if step % 5 == 0 or wood_count != (inv.get("wood",0)):
            line = (f"[{step:3d}] pos={pp} h={health} f={food} d={drink} e={energy} "
                    f"wood={inv.get('wood',0)} goal={current_goal} "
                    f"plan={plan_target} near_head={near_head_pred} "
                    f"GT={gt_around}")

            if step % 10 == 0:
                label = f"h{health}_f{food}_w{inv.get('wood',0)}_{current_goal}"
                save_frame(pixels, step, label)

            print(line)

        # Choose action
        if current_goal == "explore" or not current_plan or plan_step_idx >= len(current_plan):
            direction = _DIRECTIONS[rng.randint(0, 4)]
        else:
            ps = current_plan[plan_step_idx]
            known = spatial_map.find_nearest(ps.target, pp)
            if known:
                direction = _step_toward(pp, known, rng)
            else:
                direction = _DIRECTIONS[rng.randint(0, 4)]

        # Execute
        pos_before = pp
        pixels, _, done, info = env.step(direction)
        if done:
            save_frame(pixels, step, "DEATH")
            print(f"[{step}] DIED")
            break
        pos_after = info.get("player_pos", (32, 32))
        last_direction = direction

        # Update spatial map
        spatial_map.update(pos_after, "empty")  # basic
        for label, row, col, diff in detections:
            wx = int(pos_after[0]) + (col - 4)
            wy = int(pos_after[1]) + (row - 3)
            spatial_map.update((wx, wy), label)

        # Collision learning
        if detect_collision(pos_before, pos_after):
            patch = extract_facing_patch(pixels, direction)
            if patch is not None:
                match = patch_store.match(patch)
                print(f"  [{step}] COLLISION {direction}, patch_match={match}, GT={gt_around.get(direction.replace('move_',''), '?')}")

                # Save collision frame
                save_frame(pixels, step, f"COLLISION_{direction}_{match or 'unknown'}")

                if match is None or match in ("tree", "stone", "coal", "iron", "cow"):
                    oi = dict(info.get("inventory", {}))
                    pixels, _, done, info = env.step("do")
                    if done: break
                    ni = dict(info.get("inventory", {}))
                    outcome = labeler.label("do", oi, ni)
                    if outcome and patch is not None:
                        patch_store.add(outcome, patch)
                        print(f"  [{step}] LEARNED: {outcome}")

    # Summary
    print(f"\nTemplates: {list(patch_store.templates.keys())}")
    print(f"Templates counts: {dict((k, t.count) for k, t in patch_store.templates.items())}")
    print(f"Wood collected: {wood_count}")
    print(f"Frames saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
