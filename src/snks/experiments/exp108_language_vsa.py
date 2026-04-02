"""Exp 108: Language → VSA encoding accuracy (Stage 50 gate test).

Gate: ≥90% encode→decode accuracy on 30+ varied instructions.
Also: similarity matrix, subgoal mapping accuracy.

Can run on CPU (no environment, pure vector ops).
"""

from __future__ import annotations

import time

from snks.agent.vsa_world_model import VSACodebook
from snks.language.language_grounder import LanguageGrounder


def run_experiment() -> dict:
    """Run full experiment, return results dict."""
    cb = VSACodebook(dim=512, seed=42)
    grounder = LanguageGrounder(cb)

    # --- Test set: 30 diverse instructions ---
    test_cases: list[tuple[str, dict[str, str], list[str]]] = [
        # (instruction, expected_decode, expected_subgoals)
        # Basic MiniGrid
        ("pick up the key", {"action": "action_pickup", "object": "object_key"}, ["pickup_key"]),
        ("open the door", {"action": "action_open", "object": "object_door"}, ["open_door"]),
        ("go to the goal", {"action": "action_goto", "object": "object_goal"}, ["reach_goal"]),
        ("toggle the door", {"action": "action_toggle", "object": "object_door"}, ["open_door"]),
        ("drop the key", {"action": "action_drop", "object": "object_key"}, ["drop_key"]),
        ("put the box", {"action": "action_put", "object": "object_box"}, ["put_box"]),
        ("pick up the ball", {"action": "action_pickup", "object": "object_ball"}, ["pickup_ball"]),
        ("go to the key", {"action": "action_goto", "object": "object_key"}, ["goto_key"]),
        ("go to the door", {"action": "action_goto", "object": "object_door"}, ["goto_door"]),
        ("drop the ball", {"action": "action_drop", "object": "object_ball"}, ["drop_ball"]),
        # Attributed
        ("pick up the red key", {"action": "action_pickup", "object": "object_key", "attr": "color_red"}, ["pickup_key"]),
        ("pick up the blue key", {"action": "action_pickup", "object": "object_key", "attr": "color_blue"}, ["pickup_key"]),
        ("pick up the green key", {"action": "action_pickup", "object": "object_key", "attr": "color_green"}, ["pickup_key"]),
        ("pick up the purple key", {"action": "action_pickup", "object": "object_key", "attr": "color_purple"}, ["pickup_key"]),
        ("pick up the yellow key", {"action": "action_pickup", "object": "object_key", "attr": "color_yellow"}, ["pickup_key"]),
        ("pick up the grey key", {"action": "action_pickup", "object": "object_key", "attr": "color_grey"}, ["pickup_key"]),
        ("open the red door", {"action": "action_open", "object": "object_door", "attr": "color_red"}, ["open_door"]),
        ("go to the red key", {"action": "action_goto", "object": "object_key", "attr": "color_red"}, ["goto_key"]),
        ("go to the blue ball", {"action": "action_goto", "object": "object_ball", "attr": "color_blue"}, []),
        ("pick up the green ball", {"action": "action_pickup", "object": "object_ball", "attr": "color_green"}, ["pickup_ball"]),
        # Sequential
        ("pick up the key then open the door",
         {"action": "action_pickup", "object": "object_key"},  # first part only for encode
         ["pickup_key", "open_door"]),
        ("pick up the key then open the door then go to the goal",
         {"action": "action_pickup", "object": "object_key"},
         ["pickup_key", "open_door", "reach_goal"]),
        ("open the door then go to the goal",
         {"action": "action_open", "object": "object_door"},
         ["open_door", "reach_goal"]),
        ("pick up the red key then toggle the door",
         {"action": "action_pickup", "object": "object_key", "attr": "color_red"},
         ["pickup_key", "open_door"]),
        # More objects
        ("pick up the box", {"action": "action_pickup", "object": "object_box"}, ["pickup_box"]),
        ("drop the box", {"action": "action_drop", "object": "object_box"}, []),
        ("put the ball", {"action": "action_put", "object": "object_ball"}, ["put_ball"]),
        # Robustness
        ("Pick Up The Key", {"action": "action_pickup", "object": "object_key"}, ["pickup_key"]),
        ("OPEN THE DOOR", {"action": "action_open", "object": "object_door"}, ["open_door"]),
        ("go to the GOAL", {"action": "action_goto", "object": "object_goal"}, ["reach_goal"]),
    ]

    # --- 1. Encode/Decode accuracy ---
    t0 = time.time()
    decode_correct = 0
    decode_details: list[dict] = []

    for instruction, expected, _subgoals in test_cases:
        vsa = grounder.encode(instruction)
        decoded = grounder.decode(vsa)

        match = True
        for key in ["action", "object"]:
            if decoded.get(key) != expected.get(key):
                match = False
        if "attr" in expected and decoded.get("attr") != expected["attr"]:
            match = False

        if match:
            decode_correct += 1

        decode_details.append({
            "instruction": instruction,
            "expected": expected,
            "decoded": decoded,
            "match": match,
        })

    decode_accuracy = decode_correct / len(test_cases)
    encode_time = time.time() - t0

    # --- 2. Subgoal mapping accuracy ---
    subgoal_correct = 0
    subgoal_details: list[dict] = []

    for instruction, _expected, expected_subgoals in test_cases:
        result = grounder.to_subgoals(instruction)
        match = result == expected_subgoals
        if match:
            subgoal_correct += 1
        subgoal_details.append({
            "instruction": instruction,
            "expected": expected_subgoals,
            "result": result,
            "match": match,
        })

    subgoal_accuracy = subgoal_correct / len(test_cases)

    # --- 3. Similarity matrix (first 10 instructions) ---
    similarity_matrix: list[list[float]] = []
    first_10 = test_cases[:10]
    vectors = [grounder.encode(tc[0]) for tc in first_10]

    for i, v1 in enumerate(vectors):
        row = []
        for j, v2 in enumerate(vectors):
            row.append(round(cb.similarity(v1, v2), 3))
        similarity_matrix.append(row)

    # Check distinctiveness: off-diagonal should be < 0.7
    max_offdiag = 0.0
    for i in range(len(similarity_matrix)):
        for j in range(len(similarity_matrix)):
            if i != j:
                max_offdiag = max(max_offdiag, similarity_matrix[i][j])

    # --- 4. Sequential encoding ---
    seq_test = "pick up the key then open the door then go to the goal"
    seq_vectors = grounder.encode_sequence(seq_test)
    seq_decoded = [grounder.decode(v) for v in seq_vectors]

    results = {
        "decode_accuracy": decode_accuracy,
        "decode_correct": decode_correct,
        "decode_total": len(test_cases),
        "subgoal_accuracy": subgoal_accuracy,
        "subgoal_correct": subgoal_correct,
        "subgoal_total": len(test_cases),
        "max_offdiag_similarity": max_offdiag,
        "encode_time_ms": round(encode_time * 1000, 1),
        "seq_test": seq_test,
        "seq_decoded": seq_decoded,
        "gate_decode": "PASS" if decode_accuracy >= 0.90 else "FAIL",
        "gate_subgoal": "PASS" if subgoal_accuracy >= 0.90 else "FAIL",
    }

    return results


def print_results(results: dict) -> None:
    print("=" * 60)
    print("Exp 108: Language → VSA Encoding (Stage 50)")
    print("=" * 60)
    print()
    print(f"Decode accuracy: {results['decode_accuracy']:.1%} "
          f"({results['decode_correct']}/{results['decode_total']})")
    print(f"  Gate ≥90%: {results['gate_decode']}")
    print()
    print(f"Subgoal accuracy: {results['subgoal_accuracy']:.1%} "
          f"({results['subgoal_correct']}/{results['subgoal_total']})")
    print(f"  Gate ≥90%: {results['gate_subgoal']}")
    print()
    print(f"Max off-diagonal similarity: {results['max_offdiag_similarity']:.3f}")
    print(f"Encode time ({results['decode_total']} instructions): {results['encode_time_ms']}ms")
    print()
    print("Sequential test:")
    print(f"  Input: {results['seq_test']}")
    for i, d in enumerate(results['seq_decoded']):
        print(f"  Step {i+1}: {d}")


if __name__ == "__main__":
    results = run_experiment()
    print_results(results)
