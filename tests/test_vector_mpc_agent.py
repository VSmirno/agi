from __future__ import annotations

from snks.agent.vector_mpc_agent import _should_record_local_counterfactuals
from snks.agent.vector_sim import DynamicEntityState


def test_should_record_local_counterfactuals_salient_only_uses_resource_or_threat_salience():
    assert _should_record_local_counterfactuals(
        "salient_only",
        near_concept="tree",
        body={"health": 9.0, "food": 9.0, "drink": 9.0, "energy": 9.0},
        player_pos=(10, 10),
        observed_dynamic_entities=[],
    )
    assert _should_record_local_counterfactuals(
        "salient_only",
        near_concept="empty",
        body={"health": 9.0, "food": 9.0, "drink": 9.0, "energy": 9.0},
        player_pos=(10, 10),
        observed_dynamic_entities=[
            DynamicEntityState(concept_id="zombie", position=(12, 10)),
        ],
    )
    assert not _should_record_local_counterfactuals(
        "salient_only",
        near_concept="empty",
        body={"health": 9.0, "food": 9.0, "drink": 9.0, "energy": 9.0},
        player_pos=(10, 10),
        observed_dynamic_entities=[],
    )
