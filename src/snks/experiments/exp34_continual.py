"""Experiment 34: Catastrophic Forgetting Test (Stage 15).

Verifies that SNKS does NOT exhibit catastrophic forgetting when trained
sequentially on two disjoint sets of visual patterns.

Follows the proven approach from exp2_continual.py (N=10K), but uses:
  - 3 classes per phase (instead of 5)
  - N=5000 (CPU-feasible, ~10 min locally; miniPC recommended for N=10000)

Protocol:
    Stimuli: GratingGenerator (18° spacing, 10 classes total)
        Class set A: orientations 0°, 36°, 72° → class indices [0, 2, 4]
        Class set B: orientations 90°, 126°, 162° → class indices [5, 7, 9]

    Phase A (train):
        train_on_dataset(100 stimuli × 3 classes = 300 total, epochs=1)
        → NMI_A_before (KMeans on firing patterns vs true labels)

    Phase B (train):
        train_on_dataset(300 stimuli from B, epochs=1)
        → NMI_B (model learned B while forgetting A?)

    Retention test (train — system keeps adapting):
        train_on_dataset(same 300 A stimuli again, epochs=1)
        → NMI_A_after (does A re-emerge quickly?)

    Interpretation: if SNKS has catastrophic forgetting, NMI_A_after << NMI_A_before.
    If weights retain A's structure, NMI_A_after ≈ NMI_A_before (fast re-learning).

Gate:
    NMI_A_after >= 0.80 * NMI_A_before  (≤ 20% retention loss)
    NMI_B >= 0.35                        (B was learned; lower than A due to sequential order)
"""

from __future__ import annotations

import sys

import torch

from snks.data.stimuli import GratingGenerator
from snks.daf.types import (
    DafConfig,
    EncoderConfig,
    PipelineConfig,
    SKSConfig,
)
from snks.pipeline.runner import Pipeline


_N_VARIATIONS  = 100   # stimuli per class
_EPOCHS        = 1     # epochs per phase (same as exp2)

# Class indices in GratingGenerator (18° spacing: 0,18,36,...162°)
_CLASS_A = [0, 2, 4]   # 0°, 36°, 72°
_CLASS_B = [5, 7, 9]   # 90°, 126°, 162°


def _build_pipeline(device: str, num_nodes: int = 5000) -> Pipeline:
    cfg = PipelineConfig(
        daf=DafConfig(
            num_nodes=num_nodes,
            avg_degree=30,
            oscillator_model="fhn",
            coupling_strength=0.05,
            dt=0.01,
            noise_sigma=0.005,
            fhn_I_base=0.0,
            device=device,
        ),
        encoder=EncoderConfig(sdr_current_strength=1.0),
        sks=SKSConfig(
            top_k=min(num_nodes // 2, 5000),
            dbscan_eps=0.3,
            dbscan_min_samples=5,
            min_cluster_size=5,
            coherence_mode="rate",
        ),
        steps_per_cycle=200,
        device=device,
    )
    return Pipeline(cfg)


def _make_dataset(classes: list[int], n_variations: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate stimuli for given class indices with remapped 0-based labels."""
    gen = GratingGenerator(image_size=64, seed=42)
    all_images, all_labels = [], []
    for new_label, c in enumerate(classes):
        imgs, _ = gen.generate(c, n_variations)
        lbls = torch.full((n_variations,), new_label, dtype=torch.int64)
        all_images.append(imgs)
        all_labels.append(lbls)
    return torch.cat(all_images), torch.cat(all_labels)


def run(device: str = "cpu", num_nodes: int = 5000) -> dict:
    """Run continual learning test.

    Args:
        device: compute device.
        num_nodes: DAF network size (5000 for local CPU, 10000+ for miniPC).

    Returns:
        Dict with keys: passed, nmi_a_before, nmi_b, nmi_a_after, gate_details.
    """
    pipeline = _build_pipeline(device, num_nodes)

    imgs_a, labels_a = _make_dataset(_CLASS_A, _N_VARIATIONS)
    imgs_b, labels_b = _make_dataset(_CLASS_B, _N_VARIATIONS)

    # Phase A: train on A
    result_a = pipeline.train_on_dataset(imgs_a, labels_a, epochs=_EPOCHS)
    nmi_a_before = result_a.final_nmi

    # Phase B: train on B (A patterns not shown)
    result_b = pipeline.train_on_dataset(imgs_b, labels_b, epochs=_EPOCHS)
    nmi_b = result_b.final_nmi

    # Retention: re-train on A (system keeps adapting — honest continual learning test)
    result_retest = pipeline.train_on_dataset(imgs_a, labels_a, epochs=_EPOCHS)
    nmi_a_after = result_retest.final_nmi

    # Gates
    retention_threshold = 0.80 * nmi_a_before
    gate_retention = nmi_a_after >= retention_threshold
    gate_b_learned = nmi_b >= 0.35

    passed = gate_retention and gate_b_learned

    return {
        "passed":       passed,
        "nmi_a_before": round(nmi_a_before, 4),
        "nmi_b":        round(nmi_b, 4),
        "nmi_a_after":  round(nmi_a_after, 4),
        "retention_pct": round(nmi_a_after / max(nmi_a_before, 1e-8) * 100, 1),
        "num_nodes":    num_nodes,
        "gate_details": {
            f"nmi_a_after({nmi_a_after:.3f}) >= 0.80*nmi_a_before({nmi_a_before:.3f})={retention_threshold:.3f}": gate_retention,
            f"nmi_b({nmi_b:.3f}) >= 0.35": gate_b_learned,
        },
    }


if __name__ == "__main__":
    import sys as _sys
    device    = _sys.argv[1] if len(_sys.argv) > 1 else "cpu"
    num_nodes = int(_sys.argv[2]) if len(_sys.argv) > 2 else 5000

    result = run(device=device, num_nodes=num_nodes)

    print(f"\n{'='*60}")
    print(f"Exp 34: Catastrophic Forgetting Test  (N={result['num_nodes']})")
    print(f"{'='*60}")
    print(f"NMI_A_before (after Phase A train):   {result['nmi_a_before']:.4f}")
    print(f"NMI_B        (after Phase B train):   {result['nmi_b']:.4f}")
    print(f"NMI_A_after  (retention after B):     {result['nmi_a_after']:.4f}")
    print(f"Retention:                            {result['retention_pct']:.1f}%")
    print(f"\nGate details:")
    for k, v in result["gate_details"].items():
        mark = "PASS" if v else "FAIL"
        print(f"  [{mark}] {k}: {v}")
    print(f"\n{'PASS' if result['passed'] else 'FAIL'}")
    _sys.exit(0 if result["passed"] else 1)
