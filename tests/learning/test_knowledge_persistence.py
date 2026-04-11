"""Stage 82 — knowledge persistence round-trip tests.

Round-trip tests for LearnedRule, HomeostaticTracker, and
ConceptStore.save_experience / load_experience. Verifies that runtime
state (category-3 experience from IDEOLOGY v2) can be serialized,
stored, and reloaded without loss, and that merges are additive rather
than destructive.

No Crafter env or segmenter required — pure in-memory state round-trip.
"""

from __future__ import annotations

import json
from pathlib import Path

from snks.agent.concept_store import ConceptStore
from snks.agent.crafter_textbook import CrafterTextbook
from snks.agent.learned_rule import LearnedRule
from snks.agent.perception import HomeostaticTracker
from snks.learning.surprise_accumulator import ContextKey


def _store_and_tracker() -> tuple[ConceptStore, HomeostaticTracker]:
    store = ConceptStore()
    tb = CrafterTextbook("configs/crafter_textbook.yaml")
    tb.load_into(store)
    tracker = HomeostaticTracker()
    tracker.init_from_textbook(tb.body_block, store.passive_rules)
    return store, tracker


# ---------------------------------------------------------------------------
# LearnedRule round-trip
# ---------------------------------------------------------------------------


def test_learned_rule_roundtrip_l1():
    original = LearnedRule(
        precondition=ContextKey(
            visible=frozenset({"zombie", "tree"}),
            body_quartiles=(0, 0, 0, 0),
            action="move_left",
        ),
        effect={"health": -0.5},
        confidence=0.7,
        n_observations=15,
        source="runtime_nursery",
    )
    data = original.to_dict()
    # Must be JSON-safe
    serialized = json.dumps(data)
    restored_data = json.loads(serialized)
    restored = LearnedRule.from_dict(restored_data)

    assert restored.precondition == original.precondition
    assert restored.effect == original.effect
    assert restored.confidence == original.confidence
    assert restored.n_observations == original.n_observations
    assert restored.source == original.source


def test_learned_rule_roundtrip_l2():
    original = LearnedRule(
        precondition=ContextKey(
            visible=frozenset({"cow"}),
            body_quartiles=(2, 0, 2, 3),
            action="sleep",
        ),
        effect={"health": -0.067, "energy": +0.2},
        confidence=0.8,
        n_observations=25,
    )
    restored = LearnedRule.from_dict(
        json.loads(json.dumps(original.to_dict()))
    )
    assert restored.precondition == original.precondition
    assert restored.effect == original.effect


def test_learned_rule_match_preserved_after_roundtrip():
    """Matches behaviour should survive a serialize→deserialize cycle."""
    rule = LearnedRule(
        precondition=ContextKey(
            visible=frozenset({"zombie"}),
            body_quartiles=(0, 0, 0, 0),
            action="move_left",
        ),
        effect={"health": -0.5},
        confidence=0.8,
    )
    restored = LearnedRule.from_dict(json.loads(json.dumps(rule.to_dict())))
    body = {"health": 5.0, "food": 5.0, "drink": 5.0, "energy": 5.0}
    # Both should match the same visible + action
    assert rule.matches({"zombie", "tree"}, body, "move_left")
    assert restored.matches({"zombie", "tree"}, body, "move_left")
    # Neither matches a different action
    assert not rule.matches({"zombie"}, body, "do")
    assert not restored.matches({"zombie"}, body, "do")


# ---------------------------------------------------------------------------
# HomeostaticTracker round-trip
# ---------------------------------------------------------------------------


def test_tracker_roundtrip_empty():
    _, tracker = _store_and_tracker()
    data = tracker.to_dict()
    assert set(data.keys()) == {"observed_rates", "observed_max", "observation_counts"}


def test_tracker_roundtrip_with_observations():
    _, tracker = _store_and_tracker()
    # Simulate some observations
    tracker.update(
        inv_before={"food": 9, "drink": 9, "energy": 9, "health": 9},
        inv_after={"food": 8, "drink": 8, "energy": 8, "health": 9},
        visible_concepts=set(),
    )
    tracker.update(
        inv_before={"food": 8, "drink": 8, "energy": 8, "health": 9},
        inv_after={"food": 7, "drink": 8, "energy": 8, "health": 9},
        visible_concepts=set(),
    )
    # Round-trip
    data = tracker.to_dict()
    serialized = json.loads(json.dumps(data))

    _, tracker2 = _store_and_tracker()
    tracker2.load_dict(serialized)
    assert tracker2.observation_counts == tracker.observation_counts
    # Rates should match (within float precision)
    for var in tracker.observed_rates:
        assert abs(tracker2.observed_rates[var] - tracker.observed_rates[var]) < 1e-9


def test_tracker_load_merges_additive():
    """Loading observations on top of existing ones should additively
    merge counts and running-mean the rates."""
    _, t1 = _store_and_tracker()
    t1.update(
        inv_before={"food": 9}, inv_after={"food": 8}, visible_concepts=set(),
    )
    # food went 9 -> 8 once, rate = -1.0, count = 1

    _, t2 = _store_and_tracker()
    t2.update(
        inv_before={"food": 9}, inv_after={"food": 9}, visible_concepts=set(),
    )
    t2.update(
        inv_before={"food": 9}, inv_after={"food": 9}, visible_concepts=set(),
    )
    # food no change twice, rate = 0.0, count = 2

    t2.load_dict(t1.to_dict())
    # Merged: 3 observations, combined running mean = (0.0*2 + -1.0*1) / 3 = -1/3
    assert t2.observation_counts["food"] == 3
    assert abs(t2.observed_rates["food"] - (-1.0 / 3.0)) < 1e-9


def test_tracker_load_before_init_raises():
    tracker = HomeostaticTracker()
    try:
        tracker.load_dict({"observed_rates": {}, "observed_max": {}, "observation_counts": {}})
    except RuntimeError:
        return
    raise AssertionError("expected RuntimeError on load before init_from_textbook")


# ---------------------------------------------------------------------------
# ConceptStore.experience_to_dict / load_experience_dict
# ---------------------------------------------------------------------------


def test_store_experience_roundtrip_empty():
    store, _ = _store_and_tracker()
    data = store.experience_to_dict()
    assert data["version"] == 1
    assert data["learned_rules"] == []
    # rule_confidences should have entries for all loaded rules
    assert len(data["rule_confidences"]) > 0


def test_store_experience_roundtrip_with_learned_rules():
    store, _ = _store_and_tracker()
    rule = LearnedRule(
        precondition=ContextKey(
            visible=frozenset({"skeleton"}),
            body_quartiles=(0, 0, 0, 0),
            action="move_left",
        ),
        effect={"health": -0.5},
        confidence=0.6,
        n_observations=20,
    )
    store.add_learned_rule(rule)
    data = store.experience_to_dict()
    serialized = json.loads(json.dumps(data))

    store2, _ = _store_and_tracker()
    store2.load_experience_dict(serialized)
    assert len(store2.learned_rules) == 1
    lr2 = store2.learned_rules[0]
    assert lr2.precondition == rule.precondition
    assert abs(lr2.effect["health"] - (-0.5)) < 1e-9


def test_store_experience_confidence_restored():
    """Rule confidences updated at runtime should round-trip."""
    store, _ = _store_and_tracker()
    # Grab one concept's first rule and modify confidence
    cow = store.concepts.get("cow")
    assert cow is not None
    assert cow.causal_links, "cow should have a do-cow rule"
    cow.causal_links[0].confidence = 0.95

    data = store.experience_to_dict()
    serialized = json.loads(json.dumps(data))

    store2, _ = _store_and_tracker()
    store2.load_experience_dict(serialized)
    assert abs(store2.concepts["cow"].causal_links[0].confidence - 0.95) < 1e-9


def test_add_learned_rule_dedup_keeps_higher_n_obs():
    store, _ = _store_and_tracker()
    key = ContextKey(
        visible=frozenset({"cow"}),
        body_quartiles=(0, 0, 0, 0),
        action="sleep",
    )
    low = LearnedRule(precondition=key, effect={"health": 0.1}, n_observations=5)
    high = LearnedRule(precondition=key, effect={"health": 0.2}, n_observations=20)

    store.add_learned_rule(low)
    assert len(store.learned_rules) == 1
    assert store.learned_rules[0].n_observations == 5

    store.add_learned_rule(high)
    assert len(store.learned_rules) == 1
    assert store.learned_rules[0].n_observations == 20

    # Adding the lower-n_obs rule again should be ignored
    another_low = LearnedRule(precondition=key, effect={"health": 0.3}, n_observations=10)
    store.add_learned_rule(another_low)
    assert len(store.learned_rules) == 1
    assert store.learned_rules[0].n_observations == 20


# ---------------------------------------------------------------------------
# ConceptStore.save_experience / load_experience — file I/O
# ---------------------------------------------------------------------------


def test_save_and_load_experience_file(tmp_path: Path):
    store, tracker = _store_and_tracker()

    # Add some state
    rule = LearnedRule(
        precondition=ContextKey(
            visible=frozenset({"tree"}),
            body_quartiles=(0, 0, 0, 0),
            action="do",
        ),
        effect={"health": 0.1},
        confidence=0.7,
        n_observations=12,
    )
    store.add_learned_rule(rule)
    tracker.update(
        inv_before={"food": 9, "drink": 9},
        inv_after={"food": 8, "drink": 9},
        visible_concepts=set(),
    )

    path = tmp_path / "experience.json"
    store.save_experience(path, tracker=tracker)
    assert path.exists()

    store2, tracker2 = _store_and_tracker()
    loaded = store2.load_experience(path, tracker=tracker2)
    assert loaded is True
    assert len(store2.learned_rules) == 1
    assert store2.learned_rules[0].precondition == rule.precondition
    assert tracker2.observation_counts.get("food") == 1


def test_load_experience_missing_file(tmp_path: Path):
    store, _ = _store_and_tracker()
    result = store.load_experience(tmp_path / "nonexistent.json")
    assert result is False
    assert store.learned_rules == []


def test_experience_file_is_valid_json(tmp_path: Path):
    """The output file should be human-inspectable JSON."""
    store, tracker = _store_and_tracker()
    store.add_learned_rule(LearnedRule(
        precondition=ContextKey(
            visible=frozenset({"water"}),
            body_quartiles=(0, 0, 0, 0),
            action="do",
        ),
        effect={"drink": 5.0},
        n_observations=8,
    ))
    path = tmp_path / "exp.json"
    store.save_experience(path, tracker=tracker)

    with path.open() as f:
        content = f.read()
    data = json.loads(content)
    assert data["version"] == 1
    assert len(data["learned_rules"]) == 1
    assert data["learned_rules"][0]["precondition"]["action"] == "do"


def test_multi_episode_accumulation_simulation(tmp_path: Path):
    """Simulate three 'episodes' persisting to the same file. Each episode
    adds new learned rules and more observations. The file should grow."""
    path = tmp_path / "acc.json"

    # Episode 1
    store1, tracker1 = _store_and_tracker()
    tracker1.update(
        inv_before={"food": 9}, inv_after={"food": 8}, visible_concepts=set(),
    )
    store1.add_learned_rule(LearnedRule(
        precondition=ContextKey(
            visible=frozenset({"zombie"}),
            body_quartiles=(0, 0, 0, 0),
            action="move_left",
        ),
        effect={"health": -0.5},
        n_observations=10,
    ))
    store1.save_experience(path, tracker=tracker1)

    # Episode 2 — fresh store, load previous, add new rule
    store2, tracker2 = _store_and_tracker()
    store2.load_experience(path, tracker=tracker2)
    assert len(store2.learned_rules) == 1
    tracker2.update(
        inv_before={"food": 8}, inv_after={"food": 7}, visible_concepts=set(),
    )
    store2.add_learned_rule(LearnedRule(
        precondition=ContextKey(
            visible=frozenset({"skeleton"}),
            body_quartiles=(0, 0, 0, 0),
            action="move_left",
        ),
        effect={"health": -0.5},
        n_observations=12,
    ))
    store2.save_experience(path, tracker=tracker2)

    # Episode 3 — load, verify both rules present
    store3, tracker3 = _store_and_tracker()
    store3.load_experience(path, tracker=tracker3)
    assert len(store3.learned_rules) == 2
    visible_sets = {tuple(sorted(lr.precondition.visible)) for lr in store3.learned_rules}
    assert ("zombie",) in visible_sets
    assert ("skeleton",) in visible_sets
    # Tracker should have combined observations
    assert tracker3.observation_counts.get("food", 0) == 2
