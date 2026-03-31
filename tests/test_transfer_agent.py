"""Tests for language/transfer_agent.py — TransferAgent."""

import pytest

from snks.agent.causal_model import CausalWorldModel
from snks.daf.types import CausalAgentConfig
from snks.language.transfer_agent import TransferAgent, TransferResult, TransferStats


def make_model(**kwargs) -> CausalWorldModel:
    config = CausalAgentConfig(**kwargs)
    return CausalWorldModel(config)


def trained_doorkey_model() -> CausalWorldModel:
    """Model with DoorKey causal links (state predicate IDs 50-54)."""
    model = make_model(causal_min_observations=1)
    # key_present + pickup → key_held
    model.observe_transition({50, 52}, action=3, post_sks={51, 52})
    # key_held + door_locked + toggle → door_open
    model.observe_transition({51, 52}, action=5, post_sks={51, 53})
    return model


class TestTransferResult:
    def test_defaults(self):
        r = TransferResult()
        assert not r.success
        assert r.steps_taken == 0
        assert not r.explored
        assert r.links_reused == 0
        assert r.links_new == 0


class TestTransferStats:
    def test_empty_stats(self):
        s = TransferStats()
        assert s.episodes == 0
        assert s.success_rate == 0.0

    def test_success_rate(self):
        s = TransferStats(episodes=10, successes=7)
        assert abs(s.success_rate - 0.7) < 0.01


class TestTransferAgent:
    def test_init_with_empty_model(self):
        agent = TransferAgent()
        assert agent.causal_model.n_links == 0

    def test_init_with_pretrained_model(self):
        model = trained_doorkey_model()
        agent = TransferAgent(causal_model=model)
        assert agent.causal_model.n_links > 0

    def test_stats_empty(self):
        agent = TransferAgent()
        stats = agent.get_stats()
        assert stats.episodes == 0

    def test_record_result(self):
        agent = TransferAgent()
        r = TransferResult(
            success=True, steps_taken=15, explored=False,
            links_reused=3, links_new=0,
        )
        agent._record_result(r)
        stats = agent.get_stats()
        assert stats.episodes == 1
        assert stats.successes == 1
        assert stats.total_steps == 15
        assert stats.total_links_reused == 3

    def test_pre_loaded_links_count(self):
        model = trained_doorkey_model()
        agent = TransferAgent(causal_model=model)
        assert agent.pre_loaded_links == model.n_links
