"""Tests for agent/causal_model.py — CausalWorldModel."""

import pytest

from snks.agent.causal_model import CausalLink, CausalWorldModel
from snks.daf.types import CausalAgentConfig


def make_model(**kwargs) -> CausalWorldModel:
    config = CausalAgentConfig(**kwargs)
    return CausalWorldModel(config)


class TestCausalWorldModel:
    def test_observe_and_predict(self):
        model = make_model(causal_min_observations=2)
        pre = {1, 2, 3}
        post = {1, 2, 3, 10, 11}
        # Observe twice (min_observations=2)
        model.observe_transition(pre, action=3, post_sks=post)
        model.observe_transition(pre, action=3, post_sks=post)

        predicted, confidence = model.predict_effect(pre, action=3)
        assert 10 in predicted
        assert 11 in predicted
        assert confidence > 0.0

    def test_no_prediction_without_min_obs(self):
        model = make_model(causal_min_observations=5)
        model.observe_transition({1}, action=0, post_sks={1, 2})
        _, confidence = model.predict_effect({1}, action=0)
        assert confidence == 0.0

    def test_different_contexts_independent(self):
        model = make_model(causal_min_observations=1)
        model.observe_transition({1}, action=0, post_sks={1, 10})
        model.observe_transition({2}, action=0, post_sks={2, 20})

        pred1, _ = model.predict_effect({1}, action=0)
        pred2, _ = model.predict_effect({2}, action=0)
        assert 10 in pred1
        assert 20 in pred2

    def test_get_causal_links(self):
        model = make_model(causal_min_observations=2)
        for _ in range(5):
            model.observe_transition({1, 2}, action=3, post_sks={1, 2, 10})

        links = model.get_causal_links(min_confidence=0.3)
        assert len(links) > 0
        assert all(isinstance(l, CausalLink) for l in links)
        assert any(l.action == 3 for l in links)

    def test_effect_is_symmetric_difference(self):
        model = make_model(causal_min_observations=1)
        pre = {1, 2}
        post = {1, 2, 5}  # 5 appeared → sym_diff = {5}
        model.observe_transition(pre, action=2, post_sks=post)
        predicted, _ = model.predict_effect(pre, action=2)
        assert predicted == {5}
        # Also test disappearing SKS
        pre2 = {10, 20}
        post2 = {10, 30}  # 20 gone, 30 new → sym_diff = {20, 30}
        model.observe_transition(pre2, action=3, post_sks=post2)
        predicted2, _ = model.predict_effect(pre2, action=3)
        assert predicted2 == {20, 30}

    def test_n_links(self):
        model = make_model(causal_min_observations=1)
        assert model.n_links == 0
        model.observe_transition({1}, action=0, post_sks={1, 2})
        assert model.n_links == 1

    def test_empty_effect(self):
        model = make_model(causal_min_observations=1)
        model.observe_transition({1, 2}, action=0, post_sks={1, 2})
        predicted, _ = model.predict_effect({1, 2}, action=0)
        # effect is empty since no new SKS appeared
        assert predicted == set()

    def test_unknown_context_returns_empty(self):
        model = make_model()
        predicted, confidence = model.predict_effect({999}, action=0)
        assert predicted == set()
        assert confidence == 0.0
