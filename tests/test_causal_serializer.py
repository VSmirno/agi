"""Tests for agent/causal_serializer.py — CausalModelSerializer."""

import json
import tempfile
from pathlib import Path

import pytest

from snks.agent.causal_model import CausalWorldModel
from snks.agent.causal_serializer import CausalModelSerializer
from snks.daf.types import CausalAgentConfig


def make_model(**kwargs) -> CausalWorldModel:
    config = CausalAgentConfig(**kwargs)
    return CausalWorldModel(config)


def train_doorkey_model() -> CausalWorldModel:
    """Create a model with DoorKey-style causal links."""
    model = make_model(causal_min_observations=1)
    # key_present + pickup → key_held (SKS 50→51)
    model.observe_transition({50, 52}, action=3, post_sks={51, 52})
    # key_held + door_locked + toggle → door_open (SKS 51,52→51,53)
    model.observe_transition({51, 52}, action=5, post_sks={51, 53})
    return model


class TestCausalModelSerializer:
    def test_roundtrip_preserves_links(self):
        model = train_doorkey_model()
        original_links = model.n_links

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        CausalModelSerializer.save(model, path, source_env="DoorKey-5x5")
        loaded = CausalModelSerializer.load(path)

        assert loaded.n_links == original_links

    def test_roundtrip_preserves_predictions(self):
        model = train_doorkey_model()
        pred_orig, conf_orig = model.predict_effect({50, 52}, action=3)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        CausalModelSerializer.save(model, path)
        loaded = CausalModelSerializer.load(path)

        pred_loaded, conf_loaded = loaded.predict_effect({50, 52}, action=3)
        assert pred_orig == pred_loaded
        assert abs(conf_orig - conf_loaded) < 0.01

    def test_roundtrip_preserves_query_by_effect(self):
        model = train_doorkey_model()
        results_orig = model.query_by_effect(frozenset({51}))

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        CausalModelSerializer.save(model, path)
        loaded = CausalModelSerializer.load(path)

        results_loaded = loaded.query_by_effect(frozenset({51}))
        assert len(results_orig) == len(results_loaded)
        # Same actions returned
        actions_orig = {r[0] for r in results_orig}
        actions_loaded = {r[0] for r in results_loaded}
        assert actions_orig == actions_loaded

    def test_roundtrip_preserves_total_observations(self):
        model = train_doorkey_model()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        CausalModelSerializer.save(model, path)
        loaded = CausalModelSerializer.load(path)

        assert loaded._total_observations == model._total_observations

    def test_version_stored(self):
        model = train_doorkey_model()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        CausalModelSerializer.save(model, path, source_env="test-env")

        with open(path) as f:
            data = json.load(f)

        assert data["version"] == CausalModelSerializer.VERSION
        assert data["source_env"] == "test-env"

    def test_version_mismatch_raises(self):
        model = train_doorkey_model()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump({"version": 999, "transitions": []}, f)
            path = f.name

        with pytest.raises(ValueError, match="version"):
            CausalModelSerializer.load(path)

    def test_config_override_on_load(self):
        model = make_model(causal_min_observations=1)
        model.observe_transition({50}, action=3, post_sks={51})

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        CausalModelSerializer.save(model, path)

        custom_config = CausalAgentConfig(causal_min_observations=5)
        loaded = CausalModelSerializer.load(path, config=custom_config)

        assert loaded.config.causal_min_observations == 5

    def test_empty_model_roundtrip(self):
        model = make_model()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        CausalModelSerializer.save(model, path)
        loaded = CausalModelSerializer.load(path)

        assert loaded.n_links == 0
        assert loaded._total_observations == 0
