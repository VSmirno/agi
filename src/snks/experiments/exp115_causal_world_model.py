"""Exp 115: Causal World Model — Gate Experiments.

Three phases testing the causal world model learned from synthetic demonstrations:
  115a: QA-A — True/False facts, >=90% on unseen colors
  115b: QA-B — Precondition lookup, >=80% correct
  115c: QA-C — Causal chains, >=70% correct plans
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from snks.agent.causal_world_model import CausalWorldModel

ALL_COLORS = ["red", "blue", "yellow", "green", "purple", "grey"]
TRAIN_COLORS = ["red", "blue", "yellow"]
TEST_COLORS = ["green", "purple", "grey"]


def phase_a(dim: int = 512, n_locations: int = 1000) -> dict:
    """QA-A: True/False facts on unseen colors. Gate: >=90%."""
    print("=== Exp 115a: QA-A True/False ===")
    t0 = time.time()

    model = CausalWorldModel(dim=dim, n_locations=n_locations, seed=42)
    model.learn_color_rules(TRAIN_COLORS)
    stats = model.get_stats()
    print(f"  SDM writes: {stats}")

    # Test on training colors (sanity)
    train_correct = 0
    train_total = 0
    for kc in TRAIN_COLORS:
        for dc in TRAIN_COLORS:
            result = model.query_color_match(kc, dc)
            expected = kc == dc
            if result == expected:
                train_correct += 1
            train_total += 1
    train_acc = train_correct / train_total
    print(f"  Train accuracy: {train_acc:.1%} ({train_correct}/{train_total})")

    # Test on unseen colors
    test_correct = 0
    test_total = 0
    details = []
    for kc in TEST_COLORS:
        for dc in TEST_COLORS:
            result = model.query_color_match(kc, dc)
            expected = kc == dc
            ok = result == expected
            if ok:
                test_correct += 1
            test_total += 1
            details.append({
                "key": kc, "door": dc,
                "expected": expected, "predicted": result, "correct": ok,
            })
    test_acc = test_correct / test_total
    print(f"  Test accuracy:  {test_acc:.1%} ({test_correct}/{test_total})")

    # Cross-set: train key + test door and vice versa
    cross_correct = 0
    cross_total = 0
    for kc in TRAIN_COLORS:
        for dc in TEST_COLORS:
            result = model.query_color_match(kc, dc)
            if result == (kc == dc):  # always False (different sets)
                cross_correct += 1
            cross_total += 1
    for kc in TEST_COLORS:
        for dc in TRAIN_COLORS:
            result = model.query_color_match(kc, dc)
            if result == (kc == dc):
                cross_correct += 1
            cross_total += 1
    cross_acc = cross_correct / cross_total
    print(f"  Cross accuracy: {cross_acc:.1%} ({cross_correct}/{cross_total})")

    elapsed = time.time() - t0
    gate = test_acc >= 0.90
    print(f"  Gate (>=90%): {'PASS' if gate else 'FAIL'}")
    print(f"  Time: {elapsed:.2f}s")

    return {
        "phase": "115a",
        "train_accuracy": round(train_acc, 4),
        "test_accuracy": round(test_acc, 4),
        "cross_accuracy": round(cross_acc, 4),
        "gate": gate,
        "gate_threshold": 0.90,
        "details": details,
        "elapsed_s": round(elapsed, 2),
    }


def phase_b(dim: int = 512, n_locations: int = 1000) -> dict:
    """QA-B: Precondition lookup. Gate: >=80%."""
    print("\n=== Exp 115b: QA-B Preconditions ===")
    t0 = time.time()

    model = CausalWorldModel(dim=dim, n_locations=n_locations, seed=42)
    model.learn_all_rules(TRAIN_COLORS)
    stats = model.get_stats()
    print(f"  SDM writes: {stats}")

    queries = [
        # (action, param, expected_answer, description)
        ("open", "red", "red", "red key for red door"),
        ("open", "blue", "blue", "blue key for blue door"),
        ("open", "yellow", "yellow", "yellow key for yellow door"),
        # Unseen colors
        ("open", "green", "green", "green key for green door (unseen)"),
        ("open", "purple", "purple", "purple key for purple door (unseen)"),
        # Other rules
        ("pickup", None, "adjacent", "pickup requires adjacent"),
        ("forward", "locked", "blocked", "locked door blocks passage"),
        ("open", "no_key", "need_key", "open requires key"),
    ]

    correct = 0
    details = []
    for action, param, expected, desc in queries:
        answer = model.query_precondition(action, param)
        ok = answer == expected
        if ok:
            correct += 1
        details.append({
            "query": f"{action}({param})",
            "expected": expected, "answer": answer,
            "correct": ok, "description": desc,
        })
        status = "OK" if ok else "FAIL"
        print(f"  [{status}] {desc}: expected={expected}, got={answer}")

    accuracy = correct / len(queries)
    gate = accuracy >= 0.80
    elapsed = time.time() - t0
    print(f"  Accuracy: {accuracy:.1%} ({correct}/{len(queries)})")
    print(f"  Gate (>=80%): {'PASS' if gate else 'FAIL'}")
    print(f"  Time: {elapsed:.2f}s")

    return {
        "phase": "115b",
        "accuracy": round(accuracy, 4),
        "correct": correct,
        "total": len(queries),
        "gate": gate,
        "gate_threshold": 0.80,
        "details": details,
        "elapsed_s": round(elapsed, 2),
    }


def phase_c(dim: int = 512, n_locations: int = 1000) -> dict:
    """QA-C: Causal chains. Gate: >=70%."""
    print("\n=== Exp 115c: QA-C Causal Chains ===")
    t0 = time.time()

    model = CausalWorldModel(dim=dim, n_locations=n_locations, seed=42)
    model.learn_all_rules(TRAIN_COLORS)

    scenarios = [
        # (goal, color, required_steps, description)
        ("pass_locked_door", "red",
         ["pickup_key", "open_door"],
         "pass through red locked door"),
        ("pass_locked_door", "blue",
         ["pickup_key", "open_door"],
         "pass through blue locked door"),
        ("pass_locked_door", "green",
         ["pickup_key", "open_door"],
         "pass through green locked door (unseen color)"),
    ]

    correct = 0
    details = []
    for goal, color, required, desc in scenarios:
        chain = model.query_chain(goal, color=color)
        # Check required steps present in order
        has_all = all(step in chain for step in required)
        correct_order = True
        if has_all and len(required) > 1:
            indices = [chain.index(s) for s in required]
            correct_order = all(indices[i] < indices[i + 1] for i in range(len(indices) - 1))

        ok = has_all and correct_order
        if ok:
            correct += 1

        status = "OK" if ok else "FAIL"
        print(f"  [{status}] {desc}: chain={chain}")
        details.append({
            "goal": goal, "color": color,
            "chain": chain, "required": required,
            "has_all": has_all, "correct_order": correct_order,
            "correct": ok, "description": desc,
        })

    accuracy = correct / len(scenarios)
    gate = accuracy >= 0.70
    elapsed = time.time() - t0
    print(f"  Accuracy: {accuracy:.1%} ({correct}/{len(scenarios)})")
    print(f"  Gate (>=70%): {'PASS' if gate else 'FAIL'}")
    print(f"  Time: {elapsed:.2f}s")

    return {
        "phase": "115c",
        "accuracy": round(accuracy, 4),
        "correct": correct,
        "total": len(scenarios),
        "gate": gate,
        "gate_threshold": 0.70,
        "details": details,
        "elapsed_s": round(elapsed, 2),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Exp 115: Causal World Model Gates")
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--n-locations", type=int, default=1000)
    parser.add_argument("--output", type=str, default="_docs/exp115_results.json")
    args = parser.parse_args()

    result_a = phase_a(args.dim, args.n_locations)
    result_b = phase_b(args.dim, args.n_locations)
    result_c = phase_c(args.dim, args.n_locations)

    all_gates = result_a["gate"] and result_b["gate"] and result_c["gate"]
    print(f"\n{'=' * 50}")
    print(f"ALL GATES: {'PASS' if all_gates else 'FAIL'}")
    print(f"  115a QA-A: {result_a['test_accuracy']:.1%} (gate >=90%) {'PASS' if result_a['gate'] else 'FAIL'}")
    print(f"  115b QA-B: {result_b['accuracy']:.1%} (gate >=80%) {'PASS' if result_b['gate'] else 'FAIL'}")
    print(f"  115c QA-C: {result_c['accuracy']:.1%} (gate >=70%) {'PASS' if result_c['gate'] else 'FAIL'}")

    results = {
        "experiment": "exp115_causal_world_model",
        "dim": args.dim,
        "n_locations": args.n_locations,
        "phases": {"a": result_a, "b": result_b, "c": result_c},
        "all_gates": all_gates,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")
