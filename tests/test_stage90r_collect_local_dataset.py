from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "experiments"))

from experiments.stage90r_collect_local_dataset import (
    _auxiliary_counterfactual_probe_summary,
    _collection_profile,
    _episode_rng,
    _learner_transition_summary,
    _planner_teacher_summary,
    _rescue_record_summary,
)


def test_collection_profile_full_keeps_default_runtime_budget():
    profile = _collection_profile(smoke_lite=False)

    assert profile["name"] == "full"
    assert profile["runtime_model_dim"] == 16384
    assert profile["runtime_n_locations"] == 50000
    assert profile["planner_horizon"] == 10
    assert profile["beam_width"] == 5
    assert profile["max_depth"] == 3
    assert profile["enable_post_plan_passive_rollout"] is True
    assert profile["record_local_counterfactuals"] is True


def test_collection_profile_smoke_lite_uses_bounded_runtime_budget():
    profile = _collection_profile(smoke_lite=True)

    assert profile["name"] == "smoke_lite"
    assert profile["runtime_model_dim"] == 2048
    assert profile["runtime_n_locations"] == 5000
    assert profile["planner_horizon"] == 4
    assert profile["beam_width"] == 2
    assert profile["max_depth"] == 2
    assert profile["enable_post_plan_passive_rollout"] is False
    assert profile["record_local_counterfactuals"] == "salient_only"
    assert profile["local_counterfactual_horizon"] == 1


def test_episode_rng_is_reproducible_for_same_seed():
    a = _episode_rng(900)
    b = _episode_rng(900)
    c = _episode_rng(901)

    seq_a = [int(a.randint(0, 1000000)) for _ in range(8)]
    seq_b = [int(b.randint(0, 1000000)) for _ in range(8)]
    seq_c = [int(c.randint(0, 1000000)) for _ in range(8)]

    assert seq_a == seq_b
    assert seq_a != seq_c


def test_slice_a_summary_helpers_separate_primary_and_auxiliary_records():
    learner_summary = _learner_transition_summary(
        [
            {
                "primary_regime": "hostile_near",
                "action": "move_left",
                "plan_origin": "baseline",
                "next_observation_available": True,
                "terminated": False,
            },
            {
                "primary_regime": "neutral",
                "action": "sleep",
                "plan_origin": "single:tree:do",
                "next_observation_available": False,
                "terminated": True,
            },
        ]
    )
    teacher_summary = _planner_teacher_summary(
        [
            {
                "planner_action": "move_left",
                "primary_regime": "hostile_near",
                "planner_plan_origin": "baseline",
                "teacher_mode": "planner_controlled_bootstrap",
            }
        ]
    )
    rescue_summary = _rescue_record_summary(
        [
            {
                "trigger": "stall_spike",
                "rescue_improved_outcome": True,
            }
        ]
    )
    probe_summary = _auxiliary_counterfactual_probe_summary(
        [
            {
                "action": "do",
                "primary_regime": "local_resource_facing",
            }
        ]
    )

    assert learner_summary["n_records"] == 2
    assert learner_summary["terminal_transition_count"] == 1
    assert teacher_summary["bootstrap_only"] is True
    assert teacher_summary["teacher_mode_distribution"] == {"planner_controlled_bootstrap": 1}
    assert rescue_summary["improved_count"] == 1
    assert probe_summary["action_distribution"] == {"do": 1}
