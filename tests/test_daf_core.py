"""Tests for DAF engine core components."""

import pytest
import torch


class TestDeviceDetection:
    def test_get_device_cpu(self):
        from snks.device import get_device
        dev = get_device("cpu")
        assert dev == torch.device("cpu")

    def test_get_device_auto(self):
        from snks.device import get_device
        dev = get_device("auto")
        assert isinstance(dev, torch.device)

    def test_device_info(self):
        from snks.device import device_info
        info = device_info()
        assert "cuda_available" in info
        assert "pytorch_version" in info


class TestDafConfig:
    def test_default_config(self):
        from snks.daf.types import DafConfig
        cfg = DafConfig()
        assert cfg.num_nodes == 50_000
        assert cfg.state_dim == 8
        assert cfg.avg_degree == 50

    def test_custom_config(self):
        from snks.daf.types import DafConfig
        cfg = DafConfig(num_nodes=100, avg_degree=10)
        assert cfg.num_nodes == 100


class TestConfigLoading:
    def test_load_small_config(self):
        from snks.pipeline.config import load_config
        cfg = load_config("configs/small.yaml")
        assert cfg.daf.num_nodes == 10_000
        assert cfg.daf.oscillator_model == "fhn"

    def test_load_default_config(self):
        from snks.pipeline.config import load_config
        cfg = load_config("configs/default.yaml")
        assert cfg.daf.num_nodes == 50_000
