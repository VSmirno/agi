"""exp137: Full segmenter retrain for semantic agreement.

Goal:
- make pixel/CNN VisualField closely match semantic VisualField
- remove grass false positives and dynamic-object smearing

Changes vs exp136:
- full fine-tune (not just 1x1 head)
- train on world crop only (drop HUD + black band)
- mine hard negatives from frames where exp136 disagrees with semantic

Run on minipc:
  ./scripts/minipc-run.sh exp137 "from exp137_segmenter_agreement import main; main()"
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from snks.agent.perception import perceive_semantic_field, perceive_tile_field
from snks.encoder.tile_head_trainer import VIEWPORT_COLS, VIEWPORT_ROWS, viewport_tile_label
from snks.encoder.tile_segmenter import (
    TileSegmenter,
    crop_world_pixels,
    load_tile_segmenter,
    pick_device,
)

torch.backends.cudnn.enabled = False

DEFAULT_SOURCE_CHECKPOINT = Path("demos/checkpoints/exp136/segmenter_9x9.pt")
DEFAULT_OUTPUT_DIR = Path("demos/checkpoints/exp137")
N_FRAMES_GENERAL = 4000
N_FRAMES_SKELETON = 2000
N_FRAMES_HARD_NEGATIVE = 2000
EPOCHS = 120
BATCH_SIZE = 64
LR = 3e-4
WEIGHT_DECAY = 1e-4
SAMPLE_EVERY = 3
HARD_NEGATIVE_JACCARD = 0.50


class CroppedSegmenter:
    """Adapter so perceive_tile_field() can evaluate a crop-trained segmenter."""

    def __init__(self, segmenter: TileSegmenter):
        self.segmenter = segmenter

    def classify_tiles(self, pixels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.segmenter.classify_tiles(crop_world_pixels(pixels))


def _tile_labels(info: dict) -> torch.Tensor | None:
    semantic = info.get("semantic")
    player_pos = info.get("player_pos")
    if semantic is None or player_pos is None:
        return None
    tile_gt = torch.zeros(VIEWPORT_ROWS, VIEWPORT_COLS, dtype=torch.long)
    for tr in range(VIEWPORT_ROWS):
        for tc in range(VIEWPORT_COLS):
            tile_gt[tr, tc] = viewport_tile_label(semantic, player_pos, tr, tc)
    return tile_gt


def _jaccard(pixel_vf, symbolic_vf) -> float:
    pixel_set = {(cid, int(gy), int(gx)) for cid, _conf, gy, gx in pixel_vf.detections}
    symbolic_set = {
        (cid, int(gy), int(gx)) for cid, _conf, gy, gx in symbolic_vf.detections
    }
    return len(pixel_set & symbolic_set) / max(len(pixel_set | symbolic_set), 1)


def _collect_frames(
    *,
    n_frames: int,
    seed_base: int,
    frames_per_ep: int,
    hard_negative_segmenter: TileSegmenter | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    from snks.agent.crafter_pixel_env import CrafterPixelEnv

    pixels_list: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []
    stats = {
        "episodes": 0,
        "candidate_frames": 0,
        "accepted_frames": 0,
        "hard_negative_frames": 0,
    }
    ep = 0
    cropped_segmenter = (
        CroppedSegmenter(hard_negative_segmenter)
        if hard_negative_segmenter is not None
        else None
    )

    while len(pixels_list) < n_frames:
        env = CrafterPixelEnv(seed=seed_base + ep * 17)
        pixels_frame, info = env.reset()
        stats["episodes"] += 1

        for step in range(frames_per_ep * 3):
            action = np.random.randint(0, env.n_actions)
            pixels_frame, _, done, info = env.step(action)
            if done:
                break
            if step % SAMPLE_EVERY != 0:
                continue

            stats["candidate_frames"] += 1
            tile_gt = _tile_labels(info)
            if tile_gt is None:
                continue

            if cropped_segmenter is not None:
                pixel_vf = perceive_tile_field(pixels_frame, cropped_segmenter)
                symbolic_vf = perceive_semantic_field(info)
                jacc = _jaccard(pixel_vf, symbolic_vf)
                mismatch = pixel_vf.near_concept != symbolic_vf.near_concept
                if not (mismatch or jacc <= HARD_NEGATIVE_JACCARD):
                    continue
                stats["hard_negative_frames"] += 1

            pixels_list.append(torch.from_numpy(pixels_frame.copy()).float())
            labels_list.append(tile_gt)
            stats["accepted_frames"] += 1
            if len(pixels_list) >= n_frames:
                break

        ep += 1

    pixels = torch.stack(pixels_list)[:n_frames]
    labels = torch.stack(labels_list)[:n_frames]
    return pixels, labels, stats


def _class_report(labels: torch.Tensor, class_names: list[str]) -> dict[str, int]:
    counts = torch.bincount(labels.flatten(), minlength=len(class_names))
    return {
        class_names[i]: int(counts[i])
        for i in range(len(class_names))
        if int(counts[i]) > 0
    }


def _prepare_pixels(pixels: torch.Tensor) -> torch.Tensor:
    cropped = crop_world_pixels(pixels)
    assert isinstance(cropped, torch.Tensor)
    return cropped


def _train(
    *,
    segmenter: TileSegmenter,
    train_pixels: torch.Tensor,
    train_labels: torch.Tensor,
    device: torch.device,
    class_names: list[str],
    epochs: int,
) -> dict[str, float]:
    n_classes = len(class_names)
    optimizer = torch.optim.AdamW(
        segmenter.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    class_counts = torch.bincount(train_labels.flatten(), minlength=n_classes).float().clamp(min=1)
    class_weights = (1.0 / class_counts)
    class_weights = class_weights / class_weights.sum() * n_classes
    class_weights = class_weights.to(device)

    n = train_pixels.shape[0]
    perm = torch.randperm(n)
    split = max(int(n * 0.9), 1)
    train_idx = perm[:split]
    val_idx = perm[split:] if split < n else perm[: min(256, n)]

    from torch.utils.data import DataLoader, TensorDataset

    train_ds = TensorDataset(train_pixels[train_idx], train_labels[train_idx])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_pixels = train_pixels[val_idx].to(device)
    val_labels = train_labels[val_idx].to(device)

    best_val = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    t0 = time.time()

    for epoch in range(epochs):
        segmenter.train()
        total_loss = 0.0
        n_batches = 0
        for batch_px, batch_tl in train_loader:
            batch_px = batch_px.to(device)
            batch_tl = batch_tl.to(device)
            logits = segmenter(batch_px)
            loss = F.cross_entropy(logits, batch_tl, weight=class_weights)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            n_batches += 1

        if epoch % 20 == 0 or epoch == epochs - 1:
            segmenter.eval()
            with torch.no_grad():
                preds = segmenter(val_pixels).argmax(1)
                val_acc = float((preds == val_labels).float().mean().item())
            if val_acc >= best_val:
                best_val = val_acc
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in segmenter.state_dict().items()
                }
            print(
                f"epoch={epoch:03d} loss={total_loss/max(n_batches,1):.4f} "
                f"val_tile_acc={val_acc:.3f}"
            )

    if best_state is not None:
        segmenter.load_state_dict(best_state)

    return {
        "best_val_tile_acc": round(best_val, 4),
        "train_seconds": round(time.time() - t0, 1),
    }


def main() -> None:
    from snks.agent.decode_head import NEAR_CLASSES

    parser = argparse.ArgumentParser()
    parser.add_argument("--source-checkpoint", type=Path, default=DEFAULT_SOURCE_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-frames-general", type=int, default=N_FRAMES_GENERAL)
    parser.add_argument("--n-frames-skeleton", type=int, default=N_FRAMES_SKELETON)
    parser.add_argument("--n-frames-hard-negative", type=int, default=N_FRAMES_HARD_NEGATIVE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()

    device = torch.device(pick_device())
    print(f"device={device}")
    print(f"classes={NEAR_CLASSES}")

    if not args.source_checkpoint.exists():
        raise FileNotFoundError(f"source checkpoint missing: {args.source_checkpoint}")

    current_segmenter = load_tile_segmenter(str(args.source_checkpoint), device=device)

    print("\n--- collect general frames ---")
    px_general, lb_general, stats_general = _collect_frames(
        n_frames=args.n_frames_general,
        seed_base=100,
        frames_per_ep=100,
    )
    print(stats_general)

    print("\n--- collect skeleton-focused frames ---")
    px_skeleton, lb_skeleton, stats_skeleton = _collect_frames(
        n_frames=args.n_frames_skeleton,
        seed_base=500,
        frames_per_ep=100,
    )
    print(stats_skeleton)

    print("\n--- mine hard negatives from exp136 disagreement ---")
    px_hard, lb_hard, stats_hard = _collect_frames(
        n_frames=args.n_frames_hard_negative,
        seed_base=900,
        frames_per_ep=120,
        hard_negative_segmenter=current_segmenter,
    )
    print(stats_hard)

    train_pixels = torch.cat([px_general, px_skeleton, px_hard], dim=0)
    train_labels = torch.cat([lb_general, lb_skeleton, lb_hard], dim=0)
    train_pixels = _prepare_pixels(train_pixels)

    print(f"\nframes={train_pixels.shape[0]} cropped_shape={tuple(train_pixels.shape[1:])}")
    print("class_distribution=", _class_report(train_labels, NEAR_CLASSES))

    segmenter = TileSegmenter(n_classes=len(NEAR_CLASSES))
    old_state = torch.load(str(args.source_checkpoint), map_location="cpu", weights_only=True)
    segmenter.load_state_dict(old_state, strict=True)
    segmenter.to(device)

    print("\n--- full fine-tune ---")
    train_stats = _train(
        segmenter=segmenter,
        train_pixels=train_pixels,
        train_labels=train_labels,
        device=device,
        class_names=NEAR_CLASSES,
        epochs=args.epochs,
    )

    segmenter.eval().cpu()
    output_dir = args.output_dir
    metadata_path = output_dir / "metadata.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "segmenter_9x9.pt"
    torch.save(segmenter.state_dict(), out_path)

    metadata = {
        "source_checkpoint": str(args.source_checkpoint),
        "crop_world_input": True,
        "world_shape": [int(train_pixels.shape[-2]), int(train_pixels.shape[-1])],
        "n_frames_general": args.n_frames_general,
        "n_frames_skeleton": args.n_frames_skeleton,
        "n_frames_hard_negative": args.n_frames_hard_negative,
        "sample_every": SAMPLE_EVERY,
        "hard_negative_jaccard": HARD_NEGATIVE_JACCARD,
        "stats_general": stats_general,
        "stats_skeleton": stats_skeleton,
        "stats_hard_negative": stats_hard,
        "train_stats": train_stats,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))

    print(f"\nsaved_checkpoint={out_path}")
    print(f"saved_metadata={metadata_path}")


if __name__ == "__main__":
    main()
