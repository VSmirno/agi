"""Trace one survival episode step-by-step with current policy."""

import numpy as np
import torch

from snks.agent.crafter_pixel_env import CrafterPixelEnv, SEMANTIC_NAMES
from snks.agent.crafter_spatial_map import CrafterSpatialMap, _step_toward, MOVE_ACTIONS
from snks.agent.decode_head import NEAR_CLASSES
from snks.encoder.tile_head_trainer import VIEWPORT_ROWS, VIEWPORT_COLS

from exp135_eval_only import TileSegmenter
from exp135_grid8_tile_perception import (
    _perceive_segmenter, _find_adjacent, _find_nearest_threat, _OPPOSITE,
)


def trace_episode(segmenter, seed: int, max_steps: int = 100) -> None:
    env = CrafterPixelEnv(seed=seed)
    pixels, info = env.reset()
    rng = np.random.RandomState(seed)
    spatial_map = CrafterSpatialMap()

    center_r = VIEWPORT_ROWS // 2
    center_c = VIEWPORT_COLS // 2

    prev_health = 9
    flee_timer = 0
    flee_dir_panic = None
    last_action = None
    last_pos = None

    for step in range(max_steps):
        inv = dict(info.get("inventory", {}))
        player_pos = tuple(info.get("player_pos", (32, 32)))
        px_player, py_player = int(player_pos[0]), int(player_pos[1])

        px_tensor = torch.from_numpy(pixels)
        vf = _perceive_segmenter(px_tensor, segmenter)

        health = inv.get("health", 9)
        damage = prev_health - health

        # Check semantic ground truth: what's REALLY around the player?
        # (for debugging — we're cheating here)
        semantic = info.get("semantic")
        gt_adjacent = {}
        if semantic is not None:
            for direction, (dx, dy) in [
                ("up", (0, -1)), ("down", (0, 1)),
                ("left", (-1, 0)), ("right", (1, 0)),
            ]:
                wy = px_player + dx  # see viewport_tile_label convention
                wx = py_player + dy
                if 0 <= wy < 64 and 0 <= wx < 64:
                    gt_adjacent[direction] = SEMANTIC_NAMES.get(int(semantic[wy, wx]), "?")

        # What detected at near tiles?
        detected_near = {}
        for cid, conf, gy, gx in vf.detections:
            if gy == center_r - 1 and gx == center_c:
                detected_near["up"] = cid
            elif gy == center_r + 1 and gx == center_c:
                detected_near["down"] = cid
            elif gy == center_r and gx == center_c - 1:
                detected_near["left"] = cid
            elif gy == center_r and gx == center_c + 1:
                detected_near["right"] = cid

        # Count total visible threats
        threat_count = {"zombie": 0, "skeleton": 0}
        for cid, _, _, _ in vf.detections:
            if cid in threat_count:
                threat_count[cid] += 1

        # Check any GT enemies in viewport
        gt_enemies = {"zombie": 0, "skeleton": 0}
        if semantic is not None:
            for tr in range(VIEWPORT_ROWS):
                for tc in range(VIEWPORT_COLS):
                    wy = px_player + tc - 4
                    wx = py_player + tr + 1 - 4
                    if 0 <= wy < 64 and 0 <= wx < 64:
                        n = SEMANTIC_NAMES.get(int(semantic[wy, wx]), "?")
                        if n in gt_enemies:
                            gt_enemies[n] += 1

        dmg_str = f"DMG-{damage}" if damage > 0 else "    "

        print(f"step{step:3d} HP={health} F={inv.get('food',9)} D={inv.get('drink',9)} E={inv.get('energy',9)} "
              f"pos={player_pos} {dmg_str} "
              f"| GT_adj={gt_adjacent} GT_enemies={gt_enemies} "
              f"| det_near={detected_near} det_thr={threat_count} "
              f"| flee={flee_timer}")

        # Policy (copy from phase6)
        if health < prev_health:
            flee_timer = 4
            t_dist, t_away = _find_nearest_threat(vf, center_r, center_c)
            flee_dir_panic = t_away if t_away is not None else rng.choice(["up", "down", "left", "right"])
        prev_health = health

        action_str = None
        if flee_timer > 0:
            action_str = f"move_{flee_dir_panic}"
            flee_timer -= 1

        if action_str is None:
            threat_dist, flee_dir = _find_nearest_threat(vf, center_r, center_c)
            if threat_dist is not None and threat_dist <= 3:
                action_str = f"move_{flee_dir}"

        if action_str is None:
            target = "water" if inv.get("drink", 9) < 4 else ("cow" if inv.get("food", 9) < 4 else "tree")
            if vf.near_concept == target:
                action_str = "do"
            else:
                action_str = str(rng.choice(MOVE_ACTIONS))

        print(f"       → action={action_str}")

        last_action = action_str
        last_pos = player_pos

        spatial_map.update((px_player, py_player), vf.near_concept)
        pixels, _, done, info = env.step(action_str)
        if done:
            print(f"DONE at step {step}, final HP={dict(info.get('inventory',{})).get('health',0)}")
            break


def main():
    segmenter = TileSegmenter(n_classes=len(NEAR_CLASSES))
    segmenter.load_state_dict(
        torch.load("demos/checkpoints/exp135/segmenter_9x9.pt", map_location="cpu")
    )
    segmenter.eval()

    # Episode that died fast
    trace_episode(segmenter, seed=311, max_steps=60)


if __name__ == "__main__":
    main()
