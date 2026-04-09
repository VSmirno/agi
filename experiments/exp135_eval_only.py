"""Load saved segmenter and run phase4/5/6 evaluation only."""

import torch
import torch.nn as nn
from snks.encoder.tile_head_trainer import VIEWPORT_ROWS, VIEWPORT_COLS
from snks.agent.decode_head import NEAR_CLASSES


class TileSegmenter(nn.Module):
    def __init__(self, n_classes=12):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((VIEWPORT_ROWS, VIEWPORT_COLS))
        self.head = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        return self.head(self.pool(self.features(x)))

    def classify_tiles(self, pixels):
        with torch.no_grad():
            logits = self.forward(pixels)
            if logits.shape[0] == 1:
                logits = logits.squeeze(0)
                probs = torch.softmax(logits, dim=0)
                return probs.argmax(dim=0), probs.max(dim=0).values
            probs = torch.softmax(logits, dim=1)
            return probs.argmax(dim=1), probs.max(dim=1).values


def main():
    segmenter = TileSegmenter(n_classes=len(NEAR_CLASSES))
    segmenter.load_state_dict(
        torch.load("demos/checkpoints/exp135/segmenter_9x9.pt", map_location="cpu")
    )
    segmenter.eval()

    from exp135_grid8_tile_perception import (
        phase4_accuracy_gate, phase5_smoke, phase6_survival,
    )

    tile_acc = phase4_accuracy_gate(segmenter)
    smoke = phase5_smoke(segmenter, n_episodes=20, max_steps=200)
    survival = phase6_survival(segmenter, n_episodes=20, max_steps=500)

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Tile accuracy: {tile_acc:.1%}")
    print(f"  Smoke: avg wood={smoke['avg_wood']:.1f}, "
          f">=3 wood={smoke['pct_3wood']:.0%}")
    print(f"  Survival: {survival['avg_episode_length']:.0f} steps")


if __name__ == "__main__":
    main()
