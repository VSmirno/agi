"""Stage 75: Train tile_head — per-position classification using semantic map as teacher.

Semantic map provides ground truth class for each feature map cell.
Mapping: 64×64 semantic map → 4×4 grid (each cell covers ~16×16 pixel region).
For each cell, majority non-terrain class wins; if all terrain → "empty" (class 0).

Training data: random episodes, each frame → 16 (feature, GT_class) samples.
Semantic map used ONLY during training. NOT at inference.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from snks.agent.crafter_pixel_env import (
    CrafterPixelEnv,
    SEMANTIC_NAMES,
    NEAR_OBJECTS,
)
from snks.agent.decode_head import NEAR_CLASSES, NEAR_TO_IDX
from snks.encoder.cnn_encoder import CNNEncoder

# Terrain types → mapped to "empty" (class 0)
_TERRAIN = {"grass", "path", "sand", "lava", "unknown", "player"}


def semantic_cell_label(semantic: np.ndarray, gy: int, gx: int, grid_size: int = 4) -> int:
    """Get GT class index for one feature map cell from semantic map.

    Each cell covers a (64/grid_size × 64/grid_size) pixel region.
    Priority: non-terrain object > empty (terrain).

    Args:
        semantic: (64, 64) semantic map from Crafter info.
        gy, gx: grid position (0..grid_size-1).
        grid_size: feature map spatial size.

    Returns:
        Class index into NEAR_CLASSES (0 = empty).
    """
    cell_h = 64 // grid_size
    cell_w = 64 // grid_size
    patch = semantic[gy * cell_h:(gy + 1) * cell_h, gx * cell_w:(gx + 1) * cell_w]

    counts: Counter[str] = Counter()
    for val in patch.flat:
        name = SEMANTIC_NAMES.get(int(val), "unknown")
        counts[name] += 1

    # Find best non-terrain class
    for name, count in counts.most_common():
        if name not in _TERRAIN and name in NEAR_TO_IDX:
            return NEAR_TO_IDX[name]

    return 0  # empty


def collect_tile_training_data(
    encoder: CNNEncoder,
    n_frames: int = 5000,
    n_episodes: int = 50,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect (feature, GT_class) pairs from random episodes.

    Returns:
        features: (N, C) tensor of per-position features.
        labels: (N,) tensor of class indices.
    """
    encoder.eval()
    features_list: list[torch.Tensor] = []
    labels_list: list[int] = []

    frames_per_ep = max(1, n_frames // n_episodes)
    total = 0

    for ep in range(n_episodes):
        env = CrafterPixelEnv(seed=ep * 17 + 42)
        pixels, info = env.reset()
        steps_in_ep = 0

        for step in range(frames_per_ep * 3):  # walk more, sample subset
            # Random action for diversity
            action = np.random.randint(0, env.n_actions)
            pixels, _, done, info = env.step(action)

            if done:
                break

            # Sample every ~3 steps for diversity
            if step % 3 != 0:
                continue

            semantic = info.get("semantic")
            if semantic is None:
                continue

            px_tensor = torch.from_numpy(pixels).to(device)
            with torch.no_grad():
                out = encoder(px_tensor)
                fmap = out.feature_map  # (C, H, W) — single input

            H, W = fmap.shape[1], fmap.shape[2]
            for gy in range(H):
                for gx in range(W):
                    feat = fmap[:, gy, gx].cpu()
                    label = semantic_cell_label(semantic, gy, gx, H)
                    features_list.append(feat)
                    labels_list.append(label)

            total += 1
            if total >= n_frames:
                break

        if total >= n_frames:
            break

    features = torch.stack(features_list)
    labels = torch.tensor(labels_list, dtype=torch.long)
    print(f"  Collected {len(features)} samples from {total} frames")

    # Class distribution
    dist = Counter(labels_list)
    for cls_idx in sorted(dist.keys()):
        name = NEAR_CLASSES[cls_idx] if cls_idx < len(NEAR_CLASSES) else f"unk_{cls_idx}"
        print(f"    {name}: {dist[cls_idx]}")

    return features, labels


def train_tile_head(
    encoder: CNNEncoder,
    features: torch.Tensor,
    labels: torch.Tensor,
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 512,
    balance: bool = True,
    device: str = "cpu",
) -> dict[str, float]:
    """Train tile_head on collected (feature, label) pairs.

    Args:
        encoder: CNNEncoder with tile_head to train.
        features: (N, C) features (detached, from collect step).
        labels: (N,) class indices.
        epochs: training epochs.
        lr: learning rate.
        batch_size: mini-batch size.
        balance: if True, balance classes by oversampling minority.
        device: device for training.

    Returns:
        Dict with train_acc, train_loss, per-class accuracy.
    """
    features = features.to(device)
    labels = labels.to(device)

    # Balance classes via weighted sampler
    if balance:
        class_counts = torch.bincount(labels, minlength=len(NEAR_CLASSES)).float()
        class_counts = class_counts.clamp(min=1)
        weights = 1.0 / class_counts
        sample_weights = weights[labels]
        sampler = torch.utils.data.WeightedRandomSampler(
            sample_weights, len(sample_weights), replacement=True,
        )
    else:
        sampler = None

    dataset = torch.utils.data.TensorDataset(features, labels)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size,
        sampler=sampler, shuffle=(sampler is None),
    )

    # Only train tile_head, encoder is frozen
    tile_head = encoder.tile_head.to(device)
    tile_head.train()
    optimizer = torch.optim.Adam(tile_head.parameters(), lr=lr)

    best_acc = 0.0
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_feat, batch_label in loader:
            logits = tile_head(batch_feat)
            loss = F.cross_entropy(logits, batch_label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch_feat)
            correct += (logits.argmax(dim=-1) == batch_label).sum().item()
            total += len(batch_feat)

        acc = correct / max(1, total)
        avg_loss = total_loss / max(1, total)

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"    epoch {epoch:3d}: loss={avg_loss:.4f} acc={acc:.1%}")

        if acc > best_acc:
            best_acc = acc

    tile_head.eval()

    # Per-class accuracy
    with torch.no_grad():
        all_logits = tile_head(features)
        preds = all_logits.argmax(dim=-1)
        per_class: dict[str, float] = {}
        for cls_idx in range(len(NEAR_CLASSES)):
            mask = labels == cls_idx
            if mask.sum() == 0:
                continue
            cls_acc = (preds[mask] == cls_idx).float().mean().item()
            per_class[NEAR_CLASSES[cls_idx]] = cls_acc

    print(f"  Per-class accuracy:")
    for name, acc in sorted(per_class.items(), key=lambda x: -x[1]):
        print(f"    {name}: {acc:.1%}")

    return {"train_acc": best_acc, "train_loss": avg_loss, "per_class": per_class}
