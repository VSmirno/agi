"""Quick diagnostic: what does the agent collide with and how often?"""

import numpy as np
from collections import Counter
from snks.agent.crafter_pixel_env import CrafterPixelEnv, SEMANTIC_NAMES
from snks.agent.patch_perception import detect_collision

_DIRECTIONS = ["move_up", "move_down", "move_left", "move_right"]


def main():
    print("COLLISION DIAGNOSTIC")
    rng = np.random.RandomState(42)

    total_steps = 0
    total_collisions = 0
    collision_types = Counter()

    for ep in range(20):
        env = CrafterPixelEnv(seed=90000 + ep * 7)
        pixels, info = env.reset()

        for step in range(300):
            total_steps += 1
            d = _DIRECTIONS[rng.randint(0, 4)]
            pos_before = info.get("player_pos", (32, 32))
            pixels, _, done, info = env.step(d)
            if done:
                break
            pos_after = info.get("player_pos", (32, 32))

            if detect_collision(pos_before, pos_after):
                total_collisions += 1
                # What did we collide with? (GT)
                semantic = info.get("semantic")
                if semantic is not None:
                    py, px = int(pos_before[0]), int(pos_before[1])
                    dy, dx = {"move_up":(-1,0),"move_down":(1,0),"move_left":(0,-1),"move_right":(0,1)}[d]
                    ty, tx = py+dy, px+dx
                    if 0 <= ty < semantic.shape[0] and 0 <= tx < semantic.shape[1]:
                        gt = SEMANTIC_NAMES.get(int(semantic[ty, tx]), "?")
                        collision_types[gt] += 1

    print(f"Total steps: {total_steps}")
    print(f"Total collisions: {total_collisions} ({total_collisions/total_steps*100:.1f}%)")
    print(f"Collision types:")
    for t, c in sorted(collision_types.items(), key=lambda x: -x[1]):
        print(f"  {t}: {c} ({c/total_collisions*100:.1f}%)")
    print(f"\nWasted 'do' attempts per episode: ~{total_collisions/20:.0f}")


if __name__ == "__main__":
    main()
