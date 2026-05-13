"""Unit tests for schema-v1 entity promotion in TextbookPromoter."""

from __future__ import annotations

from pathlib import Path

from snks.agent.crafter_spatial_map import CrafterSpatialMap
from snks.agent.textbook_promoter import (
    PROMOTABLE_ENTITIES,
    PROMOTED_SCHEMA_VERSION,
    TextbookPromoter,
)


def _entity_node(
    *,
    concept: str,
    position: tuple[int, int],
    observation_count: int,
    observed_in_episodes: int,
    first_seen_episode: int,
    last_seen_episode: int,
    confidence: float = 1.0,
) -> dict:
    return {
        "type": "entity_observation",
        "body": {
            "concept": concept,
            "position": [position[0], position[1]],
        },
        "provenance": {
            "source": "spatial_map_compiler",
            "observation_count": observation_count,
            "observed_in_episodes": observed_in_episodes,
            "first_seen_episode": first_seen_episode,
            "last_seen_episode": last_seen_episode,
            "confidence": confidence,
        },
    }


def test_collect_entity_observations_filters_by_concept_and_count() -> None:
    promoter = TextbookPromoter()
    spatial_map = CrafterSpatialMap()
    spatial_map._map = {
        (10, 11): ("table", 1.0, 2),
        (12, 13): ("furnace", 0.9, 4),
        (14, 15): ("tree", 1.0, 9),
        (16, 17): ("table", 1.0, 1),
    }

    nodes = promoter.collect_entity_observations(spatial_map, episode_index=3)

    assert PROMOTABLE_ENTITIES == {"table", "furnace"}
    assert len(nodes) == 2
    assert nodes[0]["body"] == {"concept": "table", "position": [10, 11]}
    assert nodes[0]["provenance"]["observation_count"] == 2
    assert nodes[0]["provenance"]["first_seen_episode"] == 3
    assert nodes[1]["body"] == {"concept": "furnace", "position": [12, 13]}
    assert nodes[1]["provenance"]["observation_count"] == 4


def test_merge_nodes_accumulates_existing_entity_observation() -> None:
    promoter = TextbookPromoter()
    prior = [
        _entity_node(
            concept="table",
            position=(10, 11),
            observation_count=4,
            observed_in_episodes=2,
            first_seen_episode=0,
            last_seen_episode=1,
            confidence=0.7,
        ),
    ]
    new = [
        _entity_node(
            concept="table",
            position=(10, 11),
            observation_count=3,
            observed_in_episodes=1,
            first_seen_episode=2,
            last_seen_episode=2,
            confidence=1.0,
        ),
        _entity_node(
            concept="furnace",
            position=(20, 21),
            observation_count=2,
            observed_in_episodes=1,
            first_seen_episode=2,
            last_seen_episode=2,
            confidence=0.8,
        ),
    ]

    merged = promoter.merge_nodes(prior, new)

    assert len(merged) == 2
    by_key = {
        (node["body"]["concept"], tuple(node["body"]["position"])): node
        for node in merged
    }
    table = by_key[("table", (10, 11))]
    assert table["provenance"]["observation_count"] == 7
    assert table["provenance"]["observed_in_episodes"] == 3
    assert table["provenance"]["first_seen_episode"] == 0
    assert table["provenance"]["last_seen_episode"] == 2
    assert table["provenance"]["confidence"] == 1.0
    assert by_key[("furnace", (20, 21))]["provenance"]["observation_count"] == 2


def test_save_load_nodes_roundtrip_schema_v1(tmp_path: Path) -> None:
    promoter = TextbookPromoter()
    path = tmp_path / "seed17_promoted.yaml"
    nodes = [
        _entity_node(
            concept="table",
            position=(28, 33),
            observation_count=4,
            observed_in_episodes=2,
            first_seen_episode=0,
            last_seen_episode=1,
        )
    ]

    promoter.save_nodes(nodes, path)

    text = path.read_text()
    assert "schema_version: 1" in text
    assert promoter.load_nodes(path) == nodes
    assert promoter.load(path) == []
    assert PROMOTED_SCHEMA_VERSION == 1


def test_load_nodes_ignores_legacy_hypothesis_store(tmp_path: Path) -> None:
    promoter = TextbookPromoter()
    path = tmp_path / "legacy_promoted.yaml"
    path.write_text(
        "hypotheses:\n"
        "  - cause: zombie\n"
        "    vital: drink\n"
        "    threshold: 3.0\n"
        "    n_supporting: 7\n"
        "    n_observed: 13\n"
    )

    assert promoter.load_nodes(path) == []
