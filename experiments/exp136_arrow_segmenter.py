"""exp136: Retrain TileSegmenter head with arrow class.

Adds 'arrow' (semantic ID 17) to NEAR_OBJECTS/NEAR_CLASSES and retrains
the tile classification head. Backbone (features) is loaded from exp135
and frozen — only the Conv1x1 head is retrained.

Arrow is rare in random play, so we oversample frames containing arrows
from a dedicated skeleton-focused collection pass.

Run on minipc ONLY:
  ssh minipc "cd /opt/agi && git pull origin main"
  ssh minipc "tmux new-session -d -s exp136 'cd /opt/agi && source venv/bin/activate && \
    HSA_OVERRIDE_GFX_VERSION=11.0.0 PYTHONPATH=src python -u experiments/exp136_arrow_segmenter.py 2>&1 | \
    tee _docs/exp136.log'"
"""

from __future__ import annotations

import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.backends.cudnn.enabled = False

EXP135_CHECKPOINT = Path("demos/checkpoints/exp135/segmenter_9x9.pt")
CHECKPOINT_DIR = Path("demos/checkpoints/exp136")

# Collection parameters
N_FRAMES_GENERAL = 4000   # general episodes (terrain + mobs)
N_FRAMES_SKELETON = 2000  # skeleton-focused episodes (for arrow examples)
EPOCHS = 200
BATCH_SIZE = 64
LR = 1e-3


def collect_general_frames(n_frames: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect frames from random-walk episodes."""
    from snks.agent.crafter_pixel_env import CrafterPixelEnv
    from snks.encoder.tile_head_trainer import collect_tile_training_data, VIEWPORT_ROWS, VIEWPORT_COLS

    all_pixels = []
    all_labels = []
    ep = 0
    frames_collected = 0
    frames_per_ep = max(1, n_frames // 30)

    print(f"  Collecting {n_frames} general frames (~{frames_per_ep}/ep)...")
    while frames_collected < n_frames:
        env = CrafterPixelEnv(seed=100 + ep)
        px, lab = collect_tile_training_data(env, n_frames=frames_per_ep, device=device)
        all_pixels.append(px)
        all_labels.append(lab)
        frames_collected += px.shape[0]
        ep += 1
        if ep % 10 == 0:
            print(f"    ep{ep}: {frames_collected}/{n_frames} frames")

    pixels = torch.cat(all_pixels, dim=0)[:n_frames]
    labels = torch.cat(all_labels, dim=0)[:n_frames]
    print(f"  General: {pixels.shape[0]} frames")
    return pixels, labels


def collect_skeleton_frames(n_frames: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect frames from episodes known to have skeletons (for arrow examples).

    Uses seeds where skeletons spawn early/frequently. Keeps ALL frames,
    then oversamples those containing arrows.
    """
    from snks.agent.crafter_pixel_env import CrafterPixelEnv
    from snks.encoder.tile_head_trainer import collect_tile_training_data
    from snks.agent.decode_head import NEAR_TO_IDX

    arrow_idx = NEAR_TO_IDX.get("arrow", -1)
    all_pixels = []
    all_labels = []
    arrow_pixels = []
    arrow_labels = []
    ep = 0
    frames_collected = 0
    frames_per_ep = max(1, n_frames // 50)

    print(f"  Collecting {n_frames} skeleton-focused frames...")
    while frames_collected < n_frames:
        env = CrafterPixelEnv(seed=200 + ep)
        px, lab = collect_tile_training_data(env, n_frames=frames_per_ep, device=device)
        # Separate arrow-containing frames
        if arrow_idx >= 0:
            has_arrow = (lab == arrow_idx).any(dim=-1).any(dim=-1)  # (N,)
            if has_arrow.any():
                arrow_pixels.append(px[has_arrow])
                arrow_labels.append(lab[has_arrow])
        all_pixels.append(px)
        all_labels.append(lab)
        frames_collected += px.shape[0]
        ep += 1
        if ep % 10 == 0:
            n_arr = sum(p.shape[0] for p in arrow_pixels) if arrow_pixels else 0
            print(f"    ep{ep}: {frames_collected}/{n_frames} frames, arrow_frames={n_arr}")

    pixels = torch.cat(all_pixels, dim=0)[:n_frames]
    labels = torch.cat(all_labels, dim=0)[:n_frames]

    if arrow_pixels:
        arr_px = torch.cat(arrow_pixels, dim=0)
        arr_lb = torch.cat(arrow_labels, dim=0)
        # Oversample arrow frames 5× to compensate for rarity
        arr_px = arr_px.repeat(5, 1, 1, 1)
        arr_lb = arr_lb.repeat(5, 1, 1)
        pixels = torch.cat([pixels, arr_px], dim=0)
        labels = torch.cat([labels, arr_lb], dim=0)
        print(f"  Skeleton: {pixels.shape[0]} frames (incl {arr_px.shape[0]} arrow oversampled)")
    else:
        print(f"  Skeleton: {pixels.shape[0]} frames (0 arrow frames found!)")

    return pixels, labels


def train_head(
    segmenter: nn.Module,
    pixels: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    epochs: int = EPOCHS,
) -> None:
    """Train only the head of the segmenter (backbone frozen)."""
    from snks.agent.decode_head import NEAR_CLASSES

    # Freeze backbone
    for p in segmenter.features.parameters():
        p.requires_grad = False
    for p in segmenter.pool.parameters():
        p.requires_grad = False

    optimizer = torch.optim.Adam(segmenter.head.parameters(), lr=LR)

    n = pixels.shape[0]
    class_counts = torch.bincount(labels.flatten(), minlength=len(NEAR_CLASSES)).float().clamp(min=1)
    class_weights = (1.0 / class_counts)
    class_weights = class_weights / class_weights.sum() * len(NEAR_CLASSES)
    class_weights = class_weights.to(device)

    from torch.utils.data import DataLoader, TensorDataset
    ds = TensorDataset(pixels, labels)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    print(f"  Training head only: {n} frames, {epochs} epochs")
    t0 = time.time()
    for epoch in range(epochs):
        segmenter.train()
        total_loss = 0.0
        n_batches = 0
        for batch_px, batch_tl in loader:
            batch_px = batch_px.to(device)
            batch_tl = batch_tl.to(device)
            logits = segmenter(batch_px)
            loss = F.cross_entropy(logits, batch_tl, weight=class_weights)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        if epoch % 40 == 0 or epoch == epochs - 1:
            segmenter.eval()
            with torch.no_grad():
                sample_px = pixels[:500].to(device)
                sample_tl = labels[:500].to(device)
                preds = segmenter(sample_px).argmax(1)
                acc = (preds == sample_tl).float().mean().item()
                # Arrow-specific accuracy
                from snks.agent.decode_head import NEAR_TO_IDX
                arr_idx = NEAR_TO_IDX.get("arrow", -1)
                if arr_idx >= 0:
                    mask = sample_tl == arr_idx
                    if mask.any():
                        arrow_acc = (preds[mask] == sample_tl[mask]).float().mean().item()
                    else:
                        arrow_acc = float("nan")
                else:
                    arrow_acc = float("nan")
            print(f"  Epoch {epoch:3d}: loss={total_loss/n_batches:.4f} "
                  f"acc={acc:.1%} arrow_acc={arrow_acc:.1%}")

    print(f"  Head training done ({time.time()-t0:.0f}s)")


def class_report(labels: torch.Tensor) -> None:
    from snks.agent.decode_head import NEAR_CLASSES
    counts = torch.bincount(labels.flatten(), minlength=len(NEAR_CLASSES))
    total = counts.sum().item()
    print("  Class distribution:")
    for i, name in enumerate(NEAR_CLASSES):
        if counts[i] > 0:
            print(f"    {name:12s}: {counts[i]:6d} ({100*counts[i]/total:.2f}%)")


def main():
    from snks.encoder.tile_segmenter import TileSegmenter, pick_device
    from snks.agent.decode_head import NEAR_CLASSES

    device = torch.device(pick_device())
    print(f"Device: {device}")
    print(f"NEAR_CLASSES ({len(NEAR_CLASSES)}): {NEAR_CLASSES}")
    assert "arrow" in NEAR_CLASSES, "arrow not in NEAR_CLASSES — check crafter_pixel_env.py!"

    # Build new segmenter (n_classes includes arrow)
    segmenter = TileSegmenter(n_classes=len(NEAR_CLASSES))

    # Load backbone from exp135 (head weights incompatible — skip)
    old_state = torch.load(str(EXP135_CHECKPOINT), map_location="cpu", weights_only=True)
    new_state = segmenter.state_dict()
    loaded = {}
    skipped = []
    for k, v in old_state.items():
        if k in new_state and new_state[k].shape == v.shape:
            loaded[k] = v
        else:
            skipped.append(k)
    new_state.update(loaded)
    segmenter.load_state_dict(new_state)
    print(f"Loaded {len(loaded)} tensors from exp135, skipped {len(skipped)}: {skipped}")

    segmenter.to(device)

    # Collect training data
    print("\n--- General frames ---")
    px_gen, lb_gen = collect_general_frames(N_FRAMES_GENERAL, device=torch.device("cpu"))
    print("\n--- Skeleton-focused frames ---")
    px_skel, lb_skel = collect_skeleton_frames(N_FRAMES_SKELETON, device=torch.device("cpu"))

    pixels = torch.cat([px_gen, px_skel], dim=0)
    labels = torch.cat([lb_gen, lb_skel], dim=0)
    print(f"\nTotal: {pixels.shape[0]} frames")
    class_report(labels)

    # Train head
    print("\n--- Training ---")
    train_head(segmenter, pixels, labels, device=device, epochs=EPOCHS)

    # Save
    segmenter.eval().cpu()
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CHECKPOINT_DIR / "segmenter_9x9.pt"
    torch.save(segmenter.state_dict(), out_path)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
