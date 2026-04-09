"""Isolating diagnostic: phase6 with and without enemies.

If phase6(no enemies) collects ~4.7 wood (like phase5) → policy OK, combat = problem
If phase6(no enemies) collects <2 wood → plan execution is broken, not combat
"""

import torch
from snks.agent.decode_head import NEAR_CLASSES
from exp135_eval_only import TileSegmenter
from exp135_grid8_tile_perception import phase6_survival


def main():
    segmenter = TileSegmenter(n_classes=len(NEAR_CLASSES))
    segmenter.load_state_dict(torch.load(
        "demos/checkpoints/exp135/segmenter_9x9.pt", map_location="cpu"))
    segmenter.eval()

    print("\n" + "#" * 60)
    print("# TEST 1: phase6 NO ENEMIES + verbose first episode")
    print("#" * 60)
    r1 = phase6_survival(segmenter, n_episodes=5, max_steps=200,
                         enemies=False, verbose=True)

    print("\n" + "#" * 60)
    print("# TEST 2: phase6 WITH ENEMIES, no verbose")
    print("#" * 60)
    r2 = phase6_survival(segmenter, n_episodes=5, max_steps=200,
                         enemies=True, verbose=False)

    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"NO enemies:  length={r1['avg_episode_length']:.0f} resources={r1['resources']}")
    print(f"WITH enemies: length={r2['avg_episode_length']:.0f} resources={r2['resources']}")


if __name__ == "__main__":
    main()
