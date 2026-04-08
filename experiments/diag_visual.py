"""Diagnostic: look at actual CNN features for different scenes.

Render several Crafter frames, extract center features, compute
pairwise cosine similarity. Show what the CNN actually "sees".
"""

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from snks.encoder.cnn_encoder import CNNEncoder, disable_rocm_conv
from snks.agent.crafter_pixel_env import CrafterPixelEnv, CrafterControlledEnv, SEMANTIC_NAMES


def main():
    disable_rocm_conv()

    # Load encoder
    for ckpt, ch, gs in [
        (Path("demos/checkpoints/exp128"), 256, 4),
    ]:
        for tag in ["final", "phase3"]:
            path = ckpt / tag / "encoder.pt"
            if path.exists():
                encoder = CNNEncoder(feature_channels=ch, grid_size=gs)
                encoder.load_state_dict(torch.load(path, weights_only=True))
                encoder.eval()
                if torch.cuda.is_available():
                    encoder = encoder.cuda()
                print(f"Loaded {ch}ch grid{gs} from {path}")
                break
        else:
            continue
        break

    device = next(encoder.parameters()).device

    # Collect frames from different situations
    scenes = {}

    # 1. Controlled env: known objects nearby
    for target in ["tree", "stone", "water", "cow"]:
        frames = []
        for seed in range(5):
            try:
                env = CrafterControlledEnv(seed=42 + seed * 100)
                pixels, info = env.reset_near(target, no_enemies=True)
                frames.append((pixels, f"{target}_s{seed}"))
            except Exception:
                pass
        scenes[target] = frames

    # 2. Empty terrain
    frames = []
    for seed in range(5):
        env = CrafterControlledEnv(seed=42 + seed * 100)
        pixels, info = env.reset()
        frames.append((pixels, f"empty_s{seed}"))
    scenes["empty"] = frames

    # 3. Natural random walk — what does agent actually see
    env = CrafterPixelEnv(seed=42)
    pixels, info = env.reset()
    rng = np.random.RandomState(42)
    walk_frames = []
    for step in range(200):
        action = int(rng.randint(0, 5))  # only move actions
        pixels, _, done, info = env.step(action)
        if done:
            pixels, info = env.reset()
        if step % 20 == 0:
            semantic = info.get("semantic")
            pp = info.get("player_pos", (32, 32))
            gt = "?"
            if semantic is not None:
                # What's adjacent to player
                py, px = int(pp[0]), int(pp[1])
                for dy, dx in [(0,1),(0,-1),(1,0),(-1,0)]:
                    ny, nx = py+dy, px+dx
                    if 0 <= ny < semantic.shape[0] and 0 <= nx < semantic.shape[1]:
                        sid = int(semantic[ny, nx])
                        name = SEMANTIC_NAMES.get(sid, "?")
                        if name not in ("grass", "path", "player"):
                            gt = name
                            break
                else:
                    gt = "grass"
            walk_frames.append((pixels, f"walk_{step}_gt={gt}"))

    # Extract center features
    print(f"\n{'='*60}")
    print("CENTER FEATURES: cosine similarity matrix")
    print(f"{'='*60}\n")

    all_labels = []
    all_features = []

    for category, frames in scenes.items():
        for pixels, label in frames[:3]:  # max 3 per category
            pt = torch.from_numpy(pixels).float()
            if device.type != "cpu":
                pt = pt.to(device)
            with torch.no_grad():
                out = encoder(pt.unsqueeze(0))
                fmap = out.feature_map.squeeze(0)
                g = fmap.shape[1]
                c0 = g // 2 - 1
                center = fmap[:, c0:c0+2, c0:c0+2].mean(dim=(1, 2))
            all_features.append(center)
            all_labels.append(f"{category}_{label}")

    # Add walk frames
    for pixels, label in walk_frames[:5]:
        pt = torch.from_numpy(pixels).float()
        if device.type != "cpu":
            pt = pt.to(device)
        with torch.no_grad():
            out = encoder(pt.unsqueeze(0))
            fmap = out.feature_map.squeeze(0)
            g = fmap.shape[1]
            c0 = g // 2 - 1
            center = fmap[:, c0:c0+2, c0:c0+2].mean(dim=(1, 2))
        all_features.append(center)
        all_labels.append(label)

    # Compute pairwise similarity
    features = torch.stack(all_features)
    features_norm = F.normalize(features, dim=1)
    sim_matrix = features_norm @ features_norm.T

    # Print matrix
    n = len(all_labels)
    # Short labels
    short = [l[:20] for l in all_labels]

    print(f"{'':>22}", end="")
    for j in range(n):
        print(f"{short[j]:>8.8}", end="")
    print()

    for i in range(n):
        print(f"{short[i]:>22}", end="")
        for j in range(n):
            v = sim_matrix[i, j].item()
            print(f"  {v:.3f}" if i != j else "   1.00", end="")
        print()

    # Summary: average intra vs inter
    print(f"\nSUMMARY:")
    categories = list(scenes.keys())
    for cat in categories:
        cat_idx = [i for i, l in enumerate(all_labels) if l.startswith(cat + "_")]
        if len(cat_idx) < 2:
            continue
        intra_sims = []
        for a in cat_idx:
            for b in cat_idx:
                if a != b:
                    intra_sims.append(sim_matrix[a, b].item())
        inter_sims = []
        for a in cat_idx:
            for b in range(n):
                if b not in cat_idx:
                    inter_sims.append(sim_matrix[a, b].item())
        print(f"  {cat}: intra={np.mean(intra_sims):.3f}, inter={np.mean(inter_sims):.3f}, "
              f"gap={np.mean(intra_sims)-np.mean(inter_sims):.3f}")

    # What does near_head say?
    print(f"\n{'='*60}")
    print("NEAR_HEAD classification (what CNN was trained to do)")
    print(f"{'='*60}\n")

    from snks.agent.decode_head import NEAR_CLASSES
    for label, feat in zip(all_labels, all_features):
        # near_head uses center 2×2 features — we have center mean
        # Reconstruct approximate near_head input
        # Actually near_head needs (B, C*2*2) — we only have mean
        # Let's just run full forward and check near_logits
        pass

    # Use full frames for near_head
    print("Near_head predictions:")
    for category, frames in scenes.items():
        for pixels, label in frames[:2]:
            pt = torch.from_numpy(pixels).float()
            if device.type != "cpu":
                pt = pt.to(device)
            with torch.no_grad():
                out = encoder(pt.unsqueeze(0))
                probs = torch.softmax(out.near_logits, dim=1).squeeze(0)
                top3 = probs.topk(3)
            preds = [(NEAR_CLASSES[idx.item()], f"{val.item():.2f}") for val, idx in zip(top3.values, top3.indices)]
            print(f"  {label:>25}: {preds}")

    for pixels, label in walk_frames[:5]:
        pt = torch.from_numpy(pixels).float()
        if device.type != "cpu":
            pt = pt.to(device)
        with torch.no_grad():
            out = encoder(pt.unsqueeze(0))
            probs = torch.softmax(out.near_logits, dim=1).squeeze(0)
            top3 = probs.topk(3)
        preds = [(NEAR_CLASSES[idx.item()], f"{val.item():.2f}") for val, idx in zip(top3.values, top3.indices)]
        print(f"  {label:>25}: {preds}")


if __name__ == "__main__":
    main()
