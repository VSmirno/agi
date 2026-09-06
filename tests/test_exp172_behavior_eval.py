"""Focused contract for exp172's shared depth-local search."""

from __future__ import annotations

from experiments.exp172_behavior_eval import _depth_local_search


def test_shared_search_uses_55_candidates_and_depth_local_choice():
    target = (2, 1, 4)

    def expand(pairs):
        return [state + (action,) for state, action in pairs]

    def score(states):
        return [
            float(sum(action != target[index] for index, action in enumerate(state)))
            for state in states
        ]

    result = _depth_local_search(
        (), expand, score, action_count=5, horizon=3, width=5, max_calls=55
    )

    assert result["actions"] == [2, 1, 4]
    assert result["cost"] == 0.0
    assert result["candidate_calls"] == 55
    assert [row["candidate_count"] for row in result["depths"]] == [5, 25, 25]
    assert all(row["uncertainty"] == 0.0 for row in result["trace"])
