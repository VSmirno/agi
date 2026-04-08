"""exp132: Retrain CNN with 512-channel feature map, then run Stage 74 agent.

Phase 0: Retrain CNN (512 channels) using exp128 pipeline
Phase 1-6: Same as exp131 (homeostatic agent)
"""

from __future__ import annotations

import time
from pathlib import Path

import torch

CHECKPOINT_DIR = Path("demos/checkpoints/exp132")

from snks.encoder.cnn_encoder import CNNEncoder, disable_rocm_conv

# Reuse exp128 training pipeline
from exp128_text_visual import (
    phase0_load_nav_encoder,
    phase1_textbook_grounding,
    phase2_collect_via_chains,
)
from exp127_scenario_curriculum import phase2_train_outcome_encoder

# Reuse exp131 agent phases
from exp131_autonomous_craft import (
    phase1_init_store,
    phase2_tree_nav,
    phase3_stone_nav,
    phase4_grounding_count,
    phase5_survival,
    phase6_verification,
    save_checkpoint as save_ckpt_131,
)


def phase0_retrain_512():
    """Retrain CNN with 512-channel feature map."""
    print("Phase 0: Retraining CNN with 512 channels...")
    t0 = time.time()

    # Load nav encoder (for data collection only)
    nav_encoder, detector = phase0_load_nav_encoder()

    # Textbook grounding (for chain generation)
    store, gen = phase1_textbook_grounding(nav_encoder)

    # Collect training data using existing chains
    dataset = phase2_collect_via_chains(detector, gen, store)

    # Train NEW encoder: 512 channels, 8×8 grid (~1 tile per cell)
    # 512ch = enough info per cell. 8×8 = 1 tile per cell, no mixing.
    print("  Training 8×8 grid encoder (512ch, 3 layers)...")

    encoder_trained, detector_trained = phase2_train_outcome_encoder(
        dataset, epochs=150,
        encoder_cls_kwargs={"feature_channels": 512, "grid_size": 8},
    )

    encoder_trained.eval()
    if torch.cuda.is_available():
        encoder_trained = encoder_trained.cuda()
    else:
        encoder_trained = encoder_trained.cpu()

    # Save checkpoint
    d = CHECKPOINT_DIR / "phase0"
    d.mkdir(parents=True, exist_ok=True)
    torch.save(encoder_trained.state_dict(), d / "encoder.pt")
    print(f"  512-channel encoder saved → {d} ({time.time()-t0:.0f}s)")

    return encoder_trained


def main():
    disable_rocm_conv()
    print("=" * 60)
    print("exp132: Retrain CNN 512ch + Stage 74 Homeostatic Agent")
    print("=" * 60)
    t_start = time.time()

    # Phase 0: Retrain CNN
    encoder = phase0_retrain_512()

    # Phase 1: Init store + tracker
    store, tracker = phase1_init_store()

    # Phase 2-6: Same as exp131
    tree = phase2_tree_nav(encoder, store, tracker)
    stone = phase3_stone_nav(encoder, store, tracker)
    grounding = phase4_grounding_count(store)
    survival = phase5_survival(encoder, store, tracker)
    verify = phase6_verification(store)

    # Save final
    d = CHECKPOINT_DIR / "final"
    d.mkdir(parents=True, exist_ok=True)
    torch.save(encoder.state_dict(), d / "encoder.pt")
    store.save(str(d / "concept_store"))

    elapsed = time.time() - t_start
    print("=" * 60)
    print(f"exp132 SUMMARY ({elapsed:.0f}s)")
    print("=" * 60)

    gates = {
        "tree_nav_50%": tree["gate_pass"],
        "stone_nav_20%": stone["gate_pass"],
        "grounding_5": grounding["gate_pass"],
        "survival_200": survival["gate_pass"],
        "verification_3": verify["gate_pass"],
    }

    print(f"  Tree nav:   {tree['success_rate']:.1%} (≥50%)")
    print(f"  Stone nav:  {stone['success_rate']:.1%} (≥20%)")
    print(f"  Grounded:   {grounding['count']} concepts (≥5)")
    print(f"  Survival:   {survival['mean_length']:.0f} steps (≥200)")
    print(f"  Verified:   {verify['count']} rules (≥3)")
    print()
    for g, p in gates.items():
        print(f"  Gate {g}: {'PASS' if p else 'FAIL'}")
    print(f"\n  Overall: {'ALL PASS' if all(gates.values()) else 'SOME FAILED'}")
    print(f"  Grounded concepts: {grounding['grounded']}")


if __name__ == "__main__":
    main()
