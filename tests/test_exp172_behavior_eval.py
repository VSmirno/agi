"""Focused contract for exp172's shared depth-local search."""

from __future__ import annotations

from experiments.exp172_behavior_eval import _depth_local_search


def test_early_progress_breaks_exact_ties_but_never_beats_better_endpoint():
    delayed = (2, 3, 0)
    early = (3, 0, 0)
    delayed_costs = (4.0, 1.0, 0.0)
    early_costs = (1.0, 4.0, 0.0)

    def expand(pairs):
        return [state + (action,) for state, action in pairs]

    def score(states):
        return [
            (
                delayed_costs[len(state) - 1]
                if state == delayed[: len(state)]
                else early_costs[len(state) - 1]
                if state == early[: len(state)]
                else 9.0
            )
            for state in states
        ]

    legacy = _depth_local_search(
        (), expand, score, action_count=5, horizon=3, width=5, max_calls=55
    )
    tied = _depth_local_search(
        (),
        expand,
        score,
        action_count=5,
        horizon=3,
        width=5,
        max_calls=55,
        early_progress_tie_break=True,
    )

    assert legacy["actions"] == list(delayed)
    assert tied["actions"] == list(early)
    assert tied["cost"] == 0.0
    assert tied["prefix_costs"] == list(early_costs)
    assert tied["candidate_calls"] == 55
    assert [row["candidate_count"] for row in tied["depths"]] == [5, 25, 25]

    delayed_costs = (4.0, 1.0, -1.0)
    better_endpoint = _depth_local_search(
        (),
        expand,
        score,
        action_count=5,
        horizon=3,
        width=5,
        max_calls=55,
        early_progress_tie_break=True,
    )

    assert better_endpoint["actions"] == list(delayed)
    assert better_endpoint["cost"] == -1.0
