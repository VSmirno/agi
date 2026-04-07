"""Diagnostic: measure perception quality of 256-dim spatial features.

Tests: how well does one-shot grounded prototype match subsequent observations?
Uses controlled env to get known-good positions.
"""

import torch
import torch.nn.functional as F
import numpy as np
from snks.encoder.cnn_encoder import CNNEncoder, disable_rocm_conv
from snks.agent.crafter_pixel_env import CrafterPixelEnv, CrafterControlledEnv
from pathlib import Path


def main():
    disable_rocm_conv()
    print("Perception Quality Diagnostic")
    print("=" * 50)

    # Load encoder
    for tag in ["final", "phase3"]:
        path = Path(f"demos/checkpoints/exp128/{tag}/encoder.pt")
        if path.exists():
            encoder = CNNEncoder()
            encoder.load_state_dict(torch.load(path, weights_only=True))
            encoder.eval()
            if torch.cuda.is_available():
                encoder = encoder.cuda()
            print(f"Loaded encoder from {path}")
            break

    device = next(encoder.parameters()).device

    # Test: controlled env with known objects
    targets = ["tree", "stone", "coal", "iron", "water", "cow"]

    # Collect center features from controlled env
    features_per_target: dict[str, list[torch.Tensor]] = {t: [] for t in targets}

    for target in targets:
        for seed in range(10):
            try:
                env = CrafterControlledEnv(seed=42 + seed * 100)
                if target == "empty":
                    pixels, _ = env.reset()
                else:
                    pixels, _ = env.reset_near(target, no_enemies=True)
            except Exception:
                continue

            pix_t = torch.from_numpy(pixels).float()
            if device.type != "cpu":
                pix_t = pix_t.to(device)

            with torch.no_grad():
                out = encoder(pix_t.unsqueeze(0))
                fmap = out.feature_map.squeeze(0)  # (256, 4, 4)
                center = fmap[:, 1:3, 1:3].mean(dim=(1, 2))  # (256,)
                features_per_target[target].append(center.cpu())

        n = len(features_per_target[target])
        print(f"  {target}: {n} samples collected")

    # Compute intra-class similarity (same object, different seeds)
    print("\nIntra-class cosine similarity (should be HIGH):")
    prototypes = {}
    for target in targets:
        feats = features_per_target[target]
        if len(feats) < 2:
            print(f"  {target}: too few samples")
            continue

        # Prototype = mean of first 3
        proto = torch.stack(feats[:3]).mean(dim=0)
        proto_norm = F.normalize(proto.unsqueeze(0), dim=1)
        prototypes[target] = proto_norm.squeeze(0)

        sims = []
        for f in feats[3:]:
            f_norm = F.normalize(f.unsqueeze(0), dim=1)
            sim = (proto_norm @ f_norm.T).item()
            sims.append(sim)

        if sims:
            print(f"  {target}: mean={np.mean(sims):.3f} min={np.min(sims):.3f} max={np.max(sims):.3f}")

    # Compute inter-class similarity (different objects)
    print("\nInter-class cosine similarity (should be LOW):")
    for t1 in targets:
        if t1 not in prototypes:
            continue
        for t2 in targets:
            if t2 not in prototypes or t2 <= t1:
                continue
            p1 = F.normalize(prototypes[t1].unsqueeze(0), dim=1)
            p2 = F.normalize(prototypes[t2].unsqueeze(0), dim=1)
            sim = (p1 @ p2.T).item()
            print(f"  {t1} vs {t2}: {sim:.3f}")

    # Test with natural env (random walk)
    print("\nNatural env: babble near tree, measure similarity:")
    from snks.agent.outcome_labeler import OutcomeLabeler
    labeler = OutcomeLabeler()

    env = CrafterPixelEnv(seed=42)
    pixels, info = env.reset()
    rng = np.random.RandomState(42)

    tree_features = []
    for step in range(500):
        action = int(rng.randint(0, 17))
        old_inv = dict(info.get("inventory", {}))
        pixels, _, done, info = env.step(action)
        if done:
            pixels, info = env.reset()
            continue
        new_inv = dict(info.get("inventory", {}))

        label = labeler.label("do" if action == 5 else str(action), old_inv, new_inv)
        if label == "tree":
            pix_t = torch.from_numpy(pixels).float()
            if device.type != "cpu":
                pix_t = pix_t.to(device)
            with torch.no_grad():
                out = encoder(pix_t.unsqueeze(0))
                fmap = out.feature_map.squeeze(0)
                center = fmap[:, 1:3, 1:3].mean(dim=(1, 2))
                tree_features.append(center.cpu())

    print(f"  Found {len(tree_features)} natural tree observations")

    if "tree" in prototypes and tree_features:
        proto_norm = F.normalize(prototypes["tree"].unsqueeze(0), dim=1)
        sims = []
        for f in tree_features:
            f_norm = F.normalize(f.unsqueeze(0), dim=1)
            sim = (proto_norm @ f_norm.T).item()
            sims.append(sim)
        print(f"  Controlled proto vs natural: mean={np.mean(sims):.3f} min={np.min(sims):.3f}")

    # Test: one-shot grounding quality
    print("\nOne-shot grounding test (single sample prototype):")
    if tree_features:
        one_shot = F.normalize(tree_features[0].unsqueeze(0), dim=1)
        sims = []
        for f in tree_features[1:]:
            f_norm = F.normalize(f.unsqueeze(0), dim=1)
            sim = (one_shot @ f_norm.T).item()
            sims.append(sim)
        if sims:
            print(f"  tree one-shot vs rest: mean={np.mean(sims):.3f} min={np.min(sims):.3f}")
            above_05 = sum(1 for s in sims if s > 0.5)
            print(f"  Above 0.5 threshold: {above_05}/{len(sims)} = {above_05/len(sims):.1%}")
            above_04 = sum(1 for s in sims if s > 0.4)
            print(f"  Above 0.4 threshold: {above_04}/{len(sims)} = {above_04/len(sims):.1%}")


if __name__ == "__main__":
    main()
