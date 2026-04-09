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
    print("# phase6 WITH ENEMIES, verbose first episode")
    print("#" * 60)
    r1 = phase6_survival(segmenter, n_episodes=5, max_steps=500,
                         enemies=True, verbose=True)

    print("\n" + "=" * 60)
    print(f"Result: length={r1['avg_episode_length']:.0f} resources={r1['resources']}")


if __name__ == "__main__":
    main()
