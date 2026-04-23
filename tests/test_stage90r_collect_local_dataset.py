from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "experiments"))

from experiments.stage90r_collect_local_dataset import _collection_profile


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
