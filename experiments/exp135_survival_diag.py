"""Diagnostic v2: use NEW reactive policy, track flee events and why they fail.

Track:
- damage events + what threat was visible (zombie / skeleton / neither)
- flee triggers (how often)
- flee success (did damage stop?)
- skeleton detection rate when damage happens from skeleton
"""

from __future__ import annotations

import numpy as np
import torch
from collections import Counter

from snks.agent.crafter_pixel_env import CrafterPixelEnv
from snks.agent.crafter_spatial_map import CrafterSpatialMap, _step_toward, MOVE_ACTIONS
from snks.agent.decode_head import NEAR_CLASSES
from snks.encoder.tile_head_trainer import VIEWPORT_ROWS, VIEWPORT_COLS

from exp135_eval_only import TileSegmenter
from exp135_grid8_tile_perception import _perceive_segmenter, _find_adjacent, _OPPOSITE


def run_episode_diag(segmenter, seed: int, max_steps: int = 500) -> dict:
    env = CrafterPixelEnv(seed=seed)
    pixels, info = env.reset()
    rng = np.random.RandomState(seed)
    spatial_map = CrafterSpatialMap()

    center_r = VIEWPORT_ROWS // 2
    center_c = VIEWPORT_COLS // 2

    timeline = []
    damage_events = []
    visible_counts: Counter = Counter()
    flee_count = 0
    prev_health = 9
    last_action = None
    last_pos = None

    steps = 0
    for step in range(max_steps):
        steps = step + 1
        inv = dict(info.get("inventory", {}))
        player_pos = tuple(info.get("player_pos", (32, 32)))
        px_player, py_player = int(player_pos[0]), int(player_pos[1])

        # Track stats
        health = inv.get("health", 9)
        food = inv.get("food", 9)
        drink = inv.get("drink", 9)
        energy = inv.get("energy", 9)
        timeline.append({
            "step": step, "health": health, "food": food,
            "drink": drink, "energy": energy,
        })

        # Track damage
        if health < prev_health:
            # Find what's visible now
            px_tensor = torch.from_numpy(pixels)
            vf_prev = _perceive_segmenter(px_tensor, segmenter)
            visible = [cid for cid, _, _, _ in vf_prev.detections]
            damage_events.append({
                "step": step,
                "delta": health - prev_health,
                "visible": Counter(visible),
                "pos": player_pos,
            })
        prev_health = health

        # Perceive
        px_tensor = torch.from_numpy(pixels)
        vf = _perceive_segmenter(px_tensor, segmenter)

        # Count visible concepts
        for cid in {cid for cid, _, _, _ in vf.detections}:
            visible_counts[cid] += 1

        # Update spatial map
        spatial_map.update((px_player, py_player), vf.near_concept)
        for cid, conf, gy, gx in vf.detections:
            wx = px_player + (gx - center_c)
            wy = py_player + (gy - (center_r - 1))
            spatial_map.update((wx, wy), cid)

        # NEW policy: reactive flee, then needs, then wood
        action_str = None
        flee_triggered = False
        for threat in ("zombie", "skeleton"):
            danger_dir = _find_adjacent(vf, center_r, center_c, threat)
            if danger_dir is not None:
                action_str = f"move_{_OPPOSITE[danger_dir]}"
                flee_triggered = True
                flee_count += 1
                break

        if action_str is None:
            target = "water" if drink < 4 else ("cow" if food < 4 else "tree")
            if vf.near_concept == target:
                tgt_dir = _find_adjacent(vf, center_r, center_c, target)
                if tgt_dir is None:
                    action_str = "do"
                else:
                    blocked = (last_pos is not None and last_pos == player_pos
                               and last_action == f"move_{tgt_dir}")
                    if blocked or last_action == f"move_{tgt_dir}":
                        action_str = "do"
                    else:
                        action_str = f"move_{tgt_dir}"
            else:
                tgt_pos = spatial_map.find_nearest(target, (px_player, py_player))
                if tgt_pos:
                    action_str = _step_toward((px_player, py_player), tgt_pos, rng)
                else:
                    dets = vf.find(target)
                    if dets:
                        _, tgy, tgx = dets[0]
                        dx = tgx - center_c
                        dy = tgy - (center_r - 1)
                        moves = []
                        if dx > 0: moves.append("move_right")
                        elif dx < 0: moves.append("move_left")
                        if dy > 0: moves.append("move_down")
                        elif dy < 0: moves.append("move_up")
                        action_str = moves[rng.randint(len(moves))] if moves else "do"
                    else:
                        action_str = str(rng.choice(MOVE_ACTIONS))

        last_action = action_str
        last_pos = player_pos

        pixels, _, done, info = env.step(action_str)

        if done:
            break

    # Determine cause of death
    final_inv = dict(info.get("inventory", {}))
    cause = "unknown"
    if final_inv.get("health", 9) <= 0:
        # Check last damage events
        if damage_events:
            last_dmg = damage_events[-1]
            if "zombie" in last_dmg["visible"]:
                cause = "zombie"
            elif "skeleton" in last_dmg["visible"]:
                cause = "skeleton"
            else:
                cause = "other_damage"
        else:
            cause = "no_damage_recorded"
    elif final_inv.get("food", 9) <= 0:
        cause = "starvation"
    elif final_inv.get("drink", 9) <= 0:
        cause = "thirst"
    elif final_inv.get("energy", 9) <= 0:
        cause = "exhaustion"
    else:
        cause = "timeout"

    return {
        "seed": seed,
        "steps": steps,
        "final_inv": final_inv,
        "cause": cause,
        "n_damage_events": len(damage_events),
        "damage_events": damage_events[-5:],  # last 5
        "visible_counts": dict(visible_counts),
        "timeline_final": timeline[-10:],  # last 10 steps
        "flee_count": flee_count,
    }


def main():
    segmenter = TileSegmenter(n_classes=len(NEAR_CLASSES))
    segmenter.load_state_dict(
        torch.load("demos/checkpoints/exp135/segmenter_9x9.pt", map_location="cpu")
    )
    segmenter.eval()

    results = []
    for ep in range(20):
        r = run_episode_diag(segmenter, seed=ep * 11 + 300, max_steps=500)
        results.append(r)
        print(f"ep{ep:2d} seed={r['seed']} steps={r['steps']:3d} "
              f"cause={r['cause']:15s} "
              f"final={{H:{r['final_inv'].get('health',0)} "
              f"F:{r['final_inv'].get('food',0)} "
              f"D:{r['final_inv'].get('drink',0)} "
              f"E:{r['final_inv'].get('energy',0)}}} "
              f"damage_events={r['n_damage_events']}")

    # Summary
    causes = Counter(r["cause"] for r in results)
    print()
    print("=" * 60)
    print("CAUSE OF DEATH SUMMARY")
    print("=" * 60)
    for cause, n in causes.most_common():
        print(f"  {cause}: {n}/20")
    print()
    avg_steps = sum(r["steps"] for r in results) / len(results)
    total_flee = sum(r["flee_count"] for r in results)
    print(f"Avg episode length: {avg_steps:.0f}")
    print(f"Total flee triggers: {total_flee} across 20 episodes")
    print()

    # Damage events — analyze what was visible
    all_dmg = [d for r in results for d in r["damage_events"]]
    dmg_with_zombie = sum(1 for d in all_dmg if "zombie" in d["visible"])
    dmg_with_skeleton = sum(1 for d in all_dmg if "skeleton" in d["visible"])
    dmg_with_neither = sum(1 for d in all_dmg if "zombie" not in d["visible"]
                            and "skeleton" not in d["visible"])
    print(f"Total damage events (last 5 per episode): {len(all_dmg)}")
    print(f"  with zombie visible: {dmg_with_zombie}")
    print(f"  with skeleton visible: {dmg_with_skeleton}")
    print(f"  with neither: {dmg_with_neither}")

    # Stats drop patterns
    print()
    print("=" * 60)
    print("SAMPLE FINAL TIMELINES (last 10 steps)")
    print("=" * 60)
    for r in results[:5]:
        print(f"\nep seed={r['seed']} cause={r['cause']}:")
        for t in r["timeline_final"]:
            print(f"  step {t['step']:3d}: H={t['health']} F={t['food']} "
                  f"D={t['drink']} E={t['energy']}")

    # Damage events
    print()
    print("=" * 60)
    print("DAMAGE EVENTS (first 10 across all episodes)")
    print("=" * 60)
    all_dmg = [(r["seed"], d) for r in results for d in r["damage_events"]]
    for seed, d in all_dmg[:10]:
        print(f"  seed={seed} step={d['step']}: delta={d['delta']} "
              f"visible={dict(d['visible'])}")


if __name__ == "__main__":
    main()
