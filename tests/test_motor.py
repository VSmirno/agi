"""Tests for agent/motor.py — MotorEncoder."""

import pytest
import torch

from snks.agent.motor import MotorEncoder


class TestMotorEncoder:
    def test_encode_shape(self):
        enc = MotorEncoder(n_actions=5, num_nodes=10000, sdr_size=512)
        currents = enc.encode(0)
        assert currents.shape == (10000,)

    def test_encode_sparsity(self):
        enc = MotorEncoder(n_actions=5, num_nodes=10000, sdr_size=512)
        currents = enc.encode(2)
        assert (currents > 0).sum().item() == 512

    def test_non_overlapping_zones(self):
        enc = MotorEncoder(n_actions=5, num_nodes=10000, sdr_size=512)
        active_sets = []
        for a in range(5):
            currents = enc.encode(a)
            active = (currents > 0).nonzero(as_tuple=False).flatten().tolist()
            active_sets.append(set(active))
        # Check no overlap between any pair
        for i in range(5):
            for j in range(i + 1, 5):
                assert len(active_sets[i] & active_sets[j]) == 0

    def test_motor_zone_at_end(self):
        enc = MotorEncoder(n_actions=5, num_nodes=10000, sdr_size=512)
        assert enc.motor_zone_start == 10000 - 5 * 512
        currents = enc.encode(0)
        # All active bits should be >= motor_zone_start
        active = (currents > 0).nonzero(as_tuple=False).flatten()
        assert active.min().item() >= enc.motor_zone_start

    def test_decode_roundtrip(self):
        enc = MotorEncoder(n_actions=5, num_nodes=10000, sdr_size=512)
        for action in range(5):
            currents = enc.encode(action)
            decoded = enc.decode(currents)
            assert decoded == action

    def test_current_strength(self):
        enc = MotorEncoder(n_actions=5, num_nodes=10000, sdr_size=512, current_strength=2.5)
        currents = enc.encode(0)
        assert currents.max().item() == pytest.approx(2.5)

    def test_too_many_nodes_raises(self):
        with pytest.raises(ValueError, match="exceeds"):
            MotorEncoder(n_actions=5, num_nodes=100, sdr_size=512)

    def test_encode_with_device(self):
        enc = MotorEncoder(n_actions=5, num_nodes=10000, sdr_size=512)
        currents = enc.encode(0, device=torch.device("cpu"))
        assert currents.device.type == "cpu"
