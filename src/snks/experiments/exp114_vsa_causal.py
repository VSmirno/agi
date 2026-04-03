"""Exp 114: VSA+SDM Few-Shot Causal Induction.

Proves that VSA binding identity property (bind(X,X) = zero_vector) enables
SDM to generalize same-color rule to unseen colors from just 3 demonstrations.

Phase A: same-color generalization (3 train → 3 test colors)
Phase B: scaling (1-5 train colors → accuracy curve)
Phase C: arbitrary mapping (memorization vs generalization)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from snks.agent.vsa_world_model import SDMMemory, VSACodebook

ALL_COLORS = ["red", "blue", "yellow", "green", "purple", "grey"]


def train_same_color(sdm: SDMMemory, codebook: VSACodebook,
                     train_colors: list[str], n_amplify: int = 10) -> None:
    """Train SDM on same-color demos for given colors."""
    for kc in train_colors:
        for dc in train_colors:
            kv = codebook.filler(f"color_{kc}")
            dv = codebook.filler(f"color_{dc}")
            relationship = VSACodebook.bind(kv, dv)
            reward = 1.0 if kc == dc else -1.0
            for _ in range(n_amplify):
                identity = torch.zeros(codebook.dim)
                sdm.write(relationship, identity, relationship, reward)


def train_arbitrary_mapping(sdm: SDMMemory, codebook: VSACodebook,
                            mapping: dict[str, str], n_amplify: int = 10) -> None:
    """Train SDM on arbitrary key→door mapping."""
    all_doors = list(set(mapping.values()))
    for kc, correct_dc in mapping.items():
        for dc in all_doors:
            kv = codebook.filler(f"color_{kc}")
            dv = codebook.filler(f"color_{dc}")
            relationship = VSACodebook.bind(kv, dv)
            reward = 1.0 if dc == correct_dc else -1.0
            for _ in range(n_amplify):
                identity = torch.zeros(codebook.dim)
                sdm.write(relationship, identity, relationship, reward)


def evaluate(sdm: SDMMemory, codebook: VSACodebook,
             test_pairs: list[tuple[str, str, bool]]) -> dict:
    """Evaluate SDM on test pairs. Returns accuracy and details."""
    correct = 0
    details = []
    for kc, dc, expected_success in test_pairs:
        kv = codebook.filler(f"color_{kc}")
        dv = codebook.filler(f"color_{dc}")
        relationship = VSACodebook.bind(kv, dv)
        identity = torch.zeros(codebook.dim)
        reward = sdm.read_reward(relationship, identity)
        # Positive reward = opens, negative = doesn't, zero = unknown (count as wrong)
        if reward > 0:
            predicted_success = True
        elif reward < 0:
            predicted_success = False
        else:
            predicted_success = None  # unknown
        is_correct = predicted_success == expected_success
        if is_correct:
            correct += 1
        details.append({
            "key": kc, "door": dc,
            "expected": expected_success, "predicted": predicted_success,
            "reward_signal": round(reward, 4), "correct": is_correct,
        })
    accuracy = correct / len(test_pairs) if test_pairs else 0
    return {"accuracy": round(accuracy, 4), "correct": correct, "total": len(test_pairs), "details": details}


def make_test_pairs(colors: list[str]) -> list[tuple[str, str, bool]]:
    """Generate all (key, door, expected_success) pairs for same-color rule."""
    pairs = []
    for kc in colors:
        for dc in colors:
            pairs.append((kc, dc, kc == dc))
    return pairs


def phase_a(dim: int = 512, n_locations: int = 1000) -> dict:
    """Phase A: same-color generalization, 3 train → 3 test."""
    print("=== Phase A: Same-Color Generalization ===")
    train_colors = ["red", "blue", "yellow"]
    test_colors = ["green", "purple", "grey"]

    codebook = VSACodebook(dim=dim)
    sdm = SDMMemory(n_locations=n_locations, dim=dim)

    # Verify identity property
    for c in ALL_COLORS:
        v = codebook.filler(f"color_{c}")
        identity = VSACodebook.bind(v, v)
        assert identity.sum().item() == 0, f"bind({c},{c}) != zero vector"
    print("  Identity property verified: bind(X,X) = zero for all colors")

    # Train
    train_same_color(sdm, codebook, train_colors)
    print(f"  Trained on {train_colors}, SDM writes: {sdm.n_writes}")

    # Eval on train colors (sanity check)
    train_result = evaluate(sdm, codebook, make_test_pairs(train_colors))
    print(f"  Train accuracy: {train_result['accuracy']:.1%} ({train_result['correct']}/{train_result['total']})")

    # Eval on UNSEEN test colors
    test_result = evaluate(sdm, codebook, make_test_pairs(test_colors))
    print(f"  Test accuracy:  {test_result['accuracy']:.1%} ({test_result['correct']}/{test_result['total']})")

    # Ablation: untrained SDM
    empty_sdm = SDMMemory(n_locations=n_locations, dim=dim)
    ablation = evaluate(empty_sdm, codebook, make_test_pairs(test_colors))
    print(f"  Untrained:      {ablation['accuracy']:.1%} ({ablation['correct']}/{ablation['total']})")

    delta = test_result["accuracy"] - ablation["accuracy"]
    print(f"  Delta:          {delta:+.1%}")

    gate_acc = test_result["accuracy"] >= 0.90
    gate_delta = delta >= 0.40
    print(f"  Gate (≥90%): {'PASS' if gate_acc else 'FAIL'}")
    print(f"  Gate (Δ≥40%): {'PASS' if gate_delta else 'FAIL'}")

    return {
        "phase": "A",
        "train_colors": train_colors,
        "test_colors": test_colors,
        "train_accuracy": train_result["accuracy"],
        "test_accuracy": test_result["accuracy"],
        "untrained_accuracy": ablation["accuracy"],
        "delta": round(delta, 4),
        "gate_accuracy": gate_acc,
        "gate_delta": gate_delta,
        "test_details": test_result["details"],
    }


def phase_b(dim: int = 512, n_locations: int = 1000) -> dict:
    """Phase B: scaling — more train colors → better accuracy."""
    print("\n=== Phase B: Scaling ===")
    results = []
    for n_train in range(1, 6):
        train_colors = ALL_COLORS[:n_train]
        test_colors = [c for c in ALL_COLORS if c not in train_colors]
        if not test_colors:
            continue

        codebook = VSACodebook(dim=dim)
        sdm = SDMMemory(n_locations=n_locations, dim=dim)
        train_same_color(sdm, codebook, train_colors)

        test_result = evaluate(sdm, codebook, make_test_pairs(test_colors))
        print(f"  {n_train} train colors → test accuracy: {test_result['accuracy']:.1%}")
        results.append({
            "n_train": n_train,
            "train_colors": train_colors,
            "test_accuracy": test_result["accuracy"],
        })

    # Check monotonic
    accs = [r["test_accuracy"] for r in results]
    monotonic = all(accs[i] <= accs[i + 1] for i in range(len(accs) - 1))
    print(f"  Monotonic: {monotonic}")

    return {"phase": "B", "results": results, "monotonic": monotonic}


def phase_c(dim: int = 512, n_locations: int = 1000) -> dict:
    """Phase C: arbitrary mapping (rotation) — memorization only."""
    print("\n=== Phase C: Arbitrary Mapping (Rotation) ===")
    mapping = {"red": "blue", "blue": "green", "green": "red"}

    codebook = VSACodebook(dim=dim)
    sdm = SDMMemory(n_locations=n_locations, dim=dim)
    train_arbitrary_mapping(sdm, codebook, mapping, n_amplify=30)
    print(f"  Mapping: {mapping}, SDM writes: {sdm.n_writes}")

    # Test on SEEN keys
    seen_pairs = []
    for kc, correct_dc in mapping.items():
        for dc in mapping.values():
            seen_pairs.append((kc, dc, dc == correct_dc))
    seen_result = evaluate(sdm, codebook, seen_pairs)
    print(f"  Seen pairs accuracy:   {seen_result['accuracy']:.1%}")

    # Test on UNSEEN keys — no correct mapping exists, measure confidence
    unseen_colors = ["yellow", "purple", "grey"]
    unseen_pairs = []
    for kc in unseen_colors:
        for dc in mapping.values():
            # For unseen keys, any positive prediction is a false positive
            unseen_pairs.append((kc, dc, False))
    unseen_result = evaluate(sdm, codebook, unseen_pairs)
    print(f"  Unseen pairs accuracy: {unseen_result['accuracy']:.1%} (expected ~random)")

    gate_seen = seen_result["accuracy"] > 0.50  # better than random (1/N doors)
    print(f"  Gate seen (>50%):  {'PASS' if gate_seen else 'FAIL'}")
    print(f"  Unseen (no false positives expected): {unseen_result['accuracy']:.1%}")

    return {
        "phase": "C",
        "mapping": mapping,
        "seen_accuracy": seen_result["accuracy"],
        "unseen_accuracy": unseen_result["accuracy"],
        "gate_seen": gate_seen,
        "seen_details": seen_result["details"],
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--n-locations", type=int, default=1000)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    result_a = phase_a(args.dim, args.n_locations)
    result_b = phase_b(args.dim, args.n_locations)
    result_c = phase_c(args.dim, args.n_locations)

    all_gates = result_a["gate_accuracy"] and result_a["gate_delta"] and result_b["monotonic"]
    print(f"\n=== ALL GATES: {'PASS' if all_gates else 'FAIL'} ===")

    results = {"A": result_a, "B": result_b, "C": result_c, "all_gates": all_gates}

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")
