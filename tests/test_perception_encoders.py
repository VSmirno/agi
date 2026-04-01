"""Tests for Stage 42: Perception encoders — symbolic + CNN."""

import numpy as np
import pytest
import torch

from snks.encoder.symbolic import SymbolicEncoder, N_OBJ_TYPES, N_COLORS, N_STATES
from snks.encoder.rgb_conv import RGBConvEncoder
from snks.daf.types import EncoderConfig
from snks.env.obs_adapter import ObsAdapter


class TestSymbolicEncoder:
    """Symbolic encoder: MiniGrid grid → SDR."""

    def test_encode_empty_grid(self):
        enc = SymbolicEncoder(sdr_size=4096)
        obs = torch.zeros(7, 7, 3, dtype=torch.float32)
        sdr = enc.encode(obs)
        assert sdr.shape == (4096,)
        assert sdr.sum() == 0  # empty grid → no active bits

    def test_encode_single_object(self):
        enc = SymbolicEncoder(sdr_size=4096)
        obs = torch.zeros(7, 7, 3, dtype=torch.float32)
        obs[3, 3, 0] = 5  # key
        obs[3, 3, 1] = 4  # yellow
        obs[3, 3, 2] = 0  # state 0
        sdr = enc.encode(obs)
        assert sdr.sum() == 3  # type + color + state

    def test_different_objects_different_sdrs(self):
        enc = SymbolicEncoder(sdr_size=4096)

        # Key at (3,3)
        obs_key = torch.zeros(7, 7, 3, dtype=torch.float32)
        obs_key[3, 3, 0] = 5  # key
        obs_key[3, 3, 1] = 4  # yellow
        sdr_key = enc.encode(obs_key)

        # Door at (3,3)
        obs_door = torch.zeros(7, 7, 3, dtype=torch.float32)
        obs_door[3, 3, 0] = 4  # door
        obs_door[3, 3, 1] = 4  # yellow
        sdr_door = enc.encode(obs_door)

        # Goal at (3,3)
        obs_goal = torch.zeros(7, 7, 3, dtype=torch.float32)
        obs_goal[3, 3, 0] = 8  # goal
        obs_goal[3, 3, 1] = 1  # green
        sdr_goal = enc.encode(obs_goal)

        # All should be different
        assert not torch.equal(sdr_key, sdr_door)
        assert not torch.equal(sdr_key, sdr_goal)
        assert not torch.equal(sdr_door, sdr_goal)

    def test_position_matters(self):
        enc = SymbolicEncoder(sdr_size=4096)

        # Key at (1,1)
        obs1 = torch.zeros(7, 7, 3, dtype=torch.float32)
        obs1[1, 1, 0] = 5
        sdr1 = enc.encode(obs1)

        # Key at (5,5)
        obs2 = torch.zeros(7, 7, 3, dtype=torch.float32)
        obs2[5, 5, 0] = 5
        sdr2 = enc.encode(obs2)

        # Same object, different position → different SDR
        assert not torch.equal(sdr1, sdr2)

    def test_sdr_to_currents(self):
        enc = SymbolicEncoder(sdr_size=4096)
        # Create SDR with many active bits so modular hash hits some
        obs = torch.zeros(7, 7, 3, dtype=torch.float32)
        obs[1, 1, 0] = 5  # key
        obs[3, 3, 0] = 4  # door
        obs[5, 5, 0] = 8  # goal
        sdr = enc.encode(obs)
        currents = enc.sdr_to_currents(sdr, n_nodes=2000)
        assert currents.shape == (2000, 8)
        assert currents[:, 0].sum() > 0

    def test_grayscale_fallback(self):
        enc = SymbolicEncoder(sdr_size=4096)
        # 2D tensor = grayscale, should return zeros
        gray = torch.rand(64, 64)
        sdr = enc.encode(gray)
        assert sdr.sum() == 0

    def test_wall_encoding(self):
        enc = SymbolicEncoder(sdr_size=4096)
        obs = torch.zeros(7, 7, 3, dtype=torch.float32)
        obs[0, 0, 0] = 1  # wall
        obs[0, 0, 1] = 2  # grey
        sdr = enc.encode(obs)
        assert sdr.sum() >= 2  # type + color

    def test_multiple_objects(self):
        enc = SymbolicEncoder(sdr_size=4096)
        obs = torch.zeros(7, 7, 3, dtype=torch.float32)
        obs[1, 1, 0] = 5  # key
        obs[3, 3, 0] = 4  # door
        obs[5, 5, 0] = 8  # goal
        sdr = enc.encode(obs)
        assert sdr.sum() >= 6  # 3 objects × 2 bits each minimum


class TestRGBConvEncoder:
    """RGB CNN encoder."""

    def test_encode_shape(self):
        config = EncoderConfig(image_size=64, sdr_size=4096, sdr_sparsity=0.04)
        enc = RGBConvEncoder(config)
        img = torch.rand(3, 64, 64)
        sdr = enc.encode(img)
        assert sdr.shape == (4096,)

    def test_sdr_is_binary(self):
        config = EncoderConfig(image_size=64, sdr_size=4096, sdr_sparsity=0.04)
        enc = RGBConvEncoder(config)
        img = torch.rand(3, 64, 64)
        sdr = enc.encode(img)
        assert set(sdr.unique().tolist()).issubset({0.0, 1.0})

    def test_sdr_sparsity(self):
        config = EncoderConfig(image_size=64, sdr_size=4096, sdr_sparsity=0.04)
        enc = RGBConvEncoder(config)
        img = torch.rand(3, 64, 64)
        sdr = enc.encode(img)
        k = round(4096 * 0.04)
        assert int(sdr.sum()) == k

    def test_different_images_different_sdrs(self):
        config = EncoderConfig(image_size=64, sdr_size=4096, sdr_sparsity=0.04)
        enc = RGBConvEncoder(config)

        # Red image
        red = torch.zeros(3, 64, 64)
        red[0] = 1.0
        sdr_red = enc.encode(red)

        # Green image
        green = torch.zeros(3, 64, 64)
        green[1] = 1.0
        sdr_green = enc.encode(green)

        # Should be different (color preserved)
        overlap = (sdr_red * sdr_green).sum()
        total = sdr_red.sum()
        overlap_ratio = float(overlap / total)
        assert overlap_ratio < 0.8, f"SDRs too similar: overlap={overlap_ratio}"

    def test_batch_encode(self):
        config = EncoderConfig(image_size=64, sdr_size=4096, sdr_sparsity=0.04)
        enc = RGBConvEncoder(config)
        imgs = torch.rand(4, 3, 64, 64)
        sdrs = enc.encode(imgs)
        assert sdrs.shape == (4, 4096)

    def test_sdr_to_currents(self):
        config = EncoderConfig(image_size=64, sdr_size=4096, sdr_sparsity=0.04)
        enc = RGBConvEncoder(config)
        sdr = torch.zeros(4096)
        sdr[10] = 1.0
        currents = enc.sdr_to_currents(sdr, n_nodes=200)
        assert currents.shape == (200, 8)

    def test_frozen_weights(self):
        config = EncoderConfig(image_size=64, sdr_size=4096, sdr_sparsity=0.04)
        enc = RGBConvEncoder(config)
        for p in enc.parameters():
            assert not p.requires_grad


class TestObsAdapterRGB:
    """ObsAdapter with RGB mode."""

    def test_rgb_mode_shape(self):
        adapter = ObsAdapter(target_size=64, mode="rgb")
        obs = np.random.randint(0, 255, (56, 56, 3), dtype=np.uint8)
        tensor = adapter.convert(obs)
        assert tensor.shape == (3, 64, 64)
        assert tensor.max() <= 1.0
        assert tensor.min() >= 0.0

    def test_grayscale_mode_shape(self):
        adapter = ObsAdapter(target_size=64, mode="grayscale")
        obs = np.random.randint(0, 255, (56, 56, 3), dtype=np.uint8)
        tensor = adapter.convert(obs)
        assert tensor.shape == (64, 64)

    def test_rgb_preserves_color(self):
        adapter = ObsAdapter(target_size=64, mode="rgb")
        # Pure red image
        obs = np.zeros((64, 64, 3), dtype=np.uint8)
        obs[:, :, 0] = 255  # red channel
        tensor = adapter.convert(obs)
        assert tensor[0].mean() > 0.5  # red channel should be high
        assert tensor[1].mean() < 0.1  # green should be low
        assert tensor[2].mean() < 0.1  # blue should be low


class TestPipelinePreSDR:
    """Pipeline accepts pre-computed SDR."""

    def test_perception_cycle_with_pre_sdr(self):
        from snks.pipeline.runner import Pipeline
        from snks.daf.types import PipelineConfig

        config = PipelineConfig()
        config.daf.num_nodes = 200
        config.daf.avg_degree = 10
        config.daf.device = "cpu"
        config.daf.disable_csr = True
        config.daf.dt = 0.005
        config.steps_per_cycle = 50

        pipeline = Pipeline(config)
        sdr = torch.zeros(config.encoder.sdr_size)
        sdr[10] = 1.0
        sdr[100] = 1.0

        result = pipeline.perception_cycle(pre_sdr=sdr)
        assert result is not None
        assert result.mean_prediction_error >= 0
