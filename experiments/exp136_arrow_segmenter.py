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

N_FRAMES_GENERAL = 4000
N_FRAMES_SKELETON = 2000
EPOCHS = 200
BATCH_SIZE = 64
LR = 1e-3
ARROW_OVERSAMPLE = 5


def collect_frames(
    n_frames: int,
    seed_base: int = 100,
    frames_per_ep: int = 100,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect raw pixel frames + per-tile labels from random-walk episodes.

    Returns:
        pixels: (N, 3, H, W) float tensor
        labels: (N, VIEWPORT_ROWS, VIEWPORT_COLS) long tensor
    """
    from snks.agent.crafter_pixel_env import CrafterPixelEnv
    from snks.encoder.tile_head_trainer import viewport_tile_label, VIEWPORT_ROWS, VIEWPORT_COLS

    all_pixels: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    ep = 0

    while sum(p.shape[0] for p in all_pixels) < n_frames if all_pixels else True:
        if all_pixels and sum(p.shape[0] for p in all_pixels) >= n_frames:
            break
        env = CrafterPixelEnv(seed=seed_base + ep * 17)
        pixels_frame, info = env.reset()
        ep_px: list[torch.Tensor] = []
        ep_lb: list[torch.Tensor] = []

        for step in range(frames_per_ep * 3):
            action = np.random.randint(0, env.n_actions)
            pixels_frame, _, done, info = env.step(action)
            if done:
                break
            if step % 3 != 0:
                continue

            semantic = info.get("semantic")
            player_pos = info.get("player_pos")
            if semantic is None or player_pos is None:
                continue

            tile_gt = torch.zeros(VIEWPORT_ROWS, VIEWPORT_COLS, dtype=torch.long)
            for tr in range(VIEWPORT_ROWS):
                for tc in range(VIEWPORT_COLS):
                    tile_gt[tr, tc] = viewport_tile_label(semantic, player_pos, tr, tc)

            ep_px.append(torch.from_numpy(pixels_frame.copy()).float())
            ep_lb.append(tile_gt)

        if ep_px:
            all_pixels.append(torch.stack(ep_px))
            all_labels.append(torch.stack(ep_lb))

        ep += 1
        total = sum(p.shape[0] for p in all_pixels)
        if ep % 10 == 0:
            print(f"    ep{ep}: {total}/{n_frames} frames")
        if total >= n_frames:
            break

    pixels = torch.cat(all_pixels, dim=0)[:n_frames]
    labels = torch.cat(all_labels, dim=0)[:n_frames]
    return pixels, labels


def oversample_arrow_frames(
    pixels: torch.Tensor,
    labels: torch.Tensor,
    arrow_idx: int,
    factor: int = ARROW_OVERSAMPLE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Add extra copies of frames containing arrow tiles."""
    has_arrow = (labels == arrow_idx).any(dim=-1).any(dim=-1)
    n_arrow = has_arrow.sum().item()
    if n_arrow == 0:
        print(f"  WARNING: 0 arrow frames found — arrow will not be learned!")
        return pixels, labels

    arr_px = pixels[has_arrow].repeat(factor, 1, 1, 1)
    arr_lb = labels[has_arrow].repeat(factor, 1, 1)
    print(f"  Arrow frames: {n_arrow} original → {arr_px.shape[0]} after {factor}x oversample")
    return torch.cat([pixels, arr_px], dim=0), torch.cat([labels, arr_lb], dim=0)


def class_report(labels: torch.Tensor, near_classes: list[str]) -> None:
    n_classes = len(near_classes)
    counts = torch.bincount(labels.flatten(), minlength=n_classes)
    total = counts.sum().item()
    print("  Class distribution:")
    for i, name in enumerate(near_classes):
        if counts[i] > 0:
            print(f"    {name:12s}: {counts[i]:7d} ({100*counts[i]/total:.3f}%)")


def train_head(
    segmenter: nn.Module,
    pixels: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    near_classes: list[str],
    arrow_idx: int,
    epochs: int = EPOCHS,
) -> None:
    """Train only the head (backbone frozen)."""
    # Freeze backbone
    for name, p in segmenter.named_parameters():
        if not name.startswith("head"):
            p.requires_grad = False

    n_classes = len(near_classes)
    optimizer = torch.optim.Adam(
        [p for p in segmenter.parameters() if p.requires_grad], lr=LR
    )

    class_counts = torch.bincount(labels.flatten(), minlength=n_classes).float().clamp(min=1)
    class_weights = (1.0 / class_counts)
    class_weights = class_weights / class_weights.sum() * n_classes
    class_weights = class_weights.to(device)

    from torch.utils.data import DataLoader, TensorDataset
    ds = TensorDataset(pixels, labels)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    n = pixels.shape[0]
    print(f"  Training head: {n} frames, {epochs} epochs, device={device}")
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

                if arrow_idx >= 0:
                    mask = sample_tl == arrow_idx
                    if mask.any():
                        arrow_acc = (preds[mask] == sample_tl[mask]).float().mean().item()
                    else:
                        arrow_acc = float("nan")
                else:
                    arrow_acc = float("nan")
            print(f"  Epoch {epoch:3d}: loss={total_loss/n_batches:.4f} "
                  f"acc={acc:.1%} arrow_acc={arrow_acc:.1%}")

    print(f"  Done ({time.time()-t0:.0f}s)")


def main():
    from snks.encoder.tile_segmenter import TileSegmenter, pick_device
    from snks.agent.decode_head import NEAR_CLASSES, NEAR_TO_IDX

    device = torch.device(pick_device())
    print(f"Device: {device}")
    print(f"NEAR_CLASSES ({len(NEAR_CLASSES)}): {NEAR_CLASSES}")
    assert "arrow" in NEAR_CLASSES, "arrow not in NEAR_CLASSES — check crafter_pixel_env.py!"

    arrow_idx = NEAR_TO_IDX["arrow"]
    print(f"arrow class index: {arrow_idx}")

    # Build new segmenter with updated n_classes
    segmenter = TileSegmenter(n_classes=len(NEAR_CLASSES))

    # Load backbone from exp135, skip head (shape mismatch)
    old_state = torch.load(str(EXP135_CHECKPOINT), map_location="cpu", weights_only=True)
    new_state = segmenter.state_dict()
    loaded, skipped = [], []
    for k, v in old_state.items():
        if k in new_state and new_state[k].shape == v.shape:
            new_state[k] = v
            loaded.append(k)
        else:
            skipped.append(k)
    segmenter.load_state_dict(new_state)
    print(f"Loaded {len(loaded)} tensors from exp135, skipped: {skipped}")

    segmenter.to(device)

    # Collect data
    print(f"\n--- General frames (seed_base=100) ---")
    t0 = time.time()
    px_gen, lb_gen = collect_frames(N_FRAMES_GENERAL, seed_base=100)
    print(f"  General: {px_gen.shape[0]} frames ({time.time()-t0:.0f}s)")

    print(f"\n--- Skeleton-focused frames (seed_base=500) ---")
    t0 = time.time()
    px_sk, lb_sk = collect_frames(N_FRAMES_SKELETON, seed_base=500)
    print(f"  Skeleton: {px_sk.shape[0]} frames ({time.time()-t0:.0f}s)")

    # Combine, then oversample arrows
    pixels = torch.cat([px_gen, px_sk], dim=0)
    labels = torch.cat([lb_gen, lb_sk], dim=0)
    print(f"\nBefore oversample: {pixels.shape[0]} frames")
    class_report(labels, NEAR_CLASSES)

    pixels, labels = oversample_arrow_frames(pixels, labels, arrow_idx)
    print(f"After oversample: {pixels.shape[0]} frames")

    # Train
    print("\n--- Training ---")
    train_head(segmenter, pixels, labels, device=device, near_classes=NEAR_CLASSES,
               arrow_idx=arrow_idx, epochs=EPOCHS)

    # Save
    segmenter.eval().cpu()
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CHECKPOINT_DIR / "segmenter_9x9.pt"
    torch.save(segmenter.state_dict(), out_path)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
