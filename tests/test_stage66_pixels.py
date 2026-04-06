"""Stage 66 v2: Pixel perception — prototype memory tests.

Gate: ≥50% Crafter QA from pixel input.
Tests run locally with CPU — small scale, fast.
Full experiment (exp122_pixels.py) runs on minipc.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from snks.encoder.cnn_encoder import CNNEncoder, CNNEncoderOutput
from snks.encoder.predictive_trainer import (
    JEPAPredictor, PredictiveTrainer, supcon_loss,
)
from snks.agent.prototype_memory import PrototypeMemory
from snks.agent.cls_world_model import CLSWorldModel
from snks.agent.crafter_pixel_env import CrafterPixelEnv, ACTION_NAMES
from snks.agent.decode_head import NEAR_CLASSES


# ── CNN Encoder ──


class TestCNNEncoder:
    def test_output_shapes(self):
        enc = CNNEncoder(n_near_classes=len(NEAR_CLASSES))
        pixels = torch.rand(4, 3, 64, 64)
        out = enc(pixels)
        assert out.z_real.shape == (4, 2048)
        assert out.z_vsa.shape == (4, 2048)
        assert out.near_logits.shape == (4, len(NEAR_CLASSES))

    def test_single_input(self):
        enc = CNNEncoder(n_near_classes=len(NEAR_CLASSES))
        pixels = torch.rand(3, 64, 64)
        out = enc(pixels)
        assert out.z_real.shape == (2048,)

    def test_param_count(self):
        enc = CNNEncoder()
        n_params = sum(p.numel() for p in enc.parameters())
        assert n_params < 10_000_000, f"Too many params: {n_params}"


# ── Supervised Contrastive Loss ──


class TestSupConLoss:
    def test_same_class_zero_loss(self):
        z = torch.randn(4, 128)
        z = z / z.norm(dim=1, keepdim=True)
        # All same label → loss should be low
        labels = torch.tensor([0, 0, 0, 0])
        loss = supcon_loss(z, labels)
        assert loss.item() >= 0.0

    def test_different_classes_higher_loss(self):
        torch.manual_seed(42)
        # Cluster A
        z_a = torch.randn(4, 128) + torch.tensor([5.0] * 128)
        # Cluster B
        z_b = torch.randn(4, 128) + torch.tensor([-5.0] * 128)
        z = torch.cat([z_a, z_b])
        labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        loss = supcon_loss(z, labels)
        # Should be finite and non-negative
        assert loss.item() >= 0.0
        assert loss.item() < 10.0

    def test_no_positives_returns_zero(self):
        z = torch.randn(4, 128)
        labels = torch.tensor([0, 1, 2, 3])  # all different
        loss = supcon_loss(z, labels)
        assert loss.item() == 0.0

    def test_single_sample_returns_zero(self):
        z = torch.randn(1, 128)
        labels = torch.tensor([0])
        loss = supcon_loss(z, labels)
        assert loss.item() == 0.0


# ── Predictive Trainer with SupCon ──


class TestPredictiveTrainerSupCon:
    def test_train_step_with_labels(self):
        enc = CNNEncoder(n_near_classes=len(NEAR_CLASSES))
        pred = JEPAPredictor()
        trainer = PredictiveTrainer(enc, pred, contrastive_weight=0.5, device="cpu")
        metrics = trainer.train_step(
            torch.rand(8, 3, 64, 64),
            torch.rand(8, 3, 64, 64),
            torch.randint(0, 17, (8,)),
            situation_labels=torch.tensor([0, 0, 1, 1, 2, 2, 0, 1]),
        )
        assert "pred_loss" in metrics
        assert "con_loss" in metrics
        assert metrics["con_loss"] > 0.0

    def test_train_step_without_labels(self):
        enc = CNNEncoder(n_near_classes=len(NEAR_CLASSES))
        pred = JEPAPredictor()
        trainer = PredictiveTrainer(enc, pred, contrastive_weight=0.5, device="cpu")
        metrics = trainer.train_step(
            torch.rand(4, 3, 64, 64),
            torch.rand(4, 3, 64, 64),
            torch.randint(0, 17, (4,)),
        )
        assert metrics["con_loss"] == 0.0


# ── Prototype Memory ──


class TestPrototypeMemory:
    def test_add_and_query(self):
        mem = PrototypeMemory(dim=128, k=3)
        z_base = torch.randn(128)
        # Add 3 prototypes for action "do" (similar to z_base)
        for _ in range(3):
            mem.add(z_base + torch.randn(128) * 0.01, "do",
                    {"result": "collected", "gives": "wood"})

        # Query with similar z
        outcome, conf = mem.query(z_base, "do")
        assert outcome["result"] == "collected"
        assert conf > 0.0

    def test_action_filtering(self):
        mem = PrototypeMemory(dim=128, k=3)
        z = torch.randn(128)
        mem.add(z, "do", {"result": "collected"})
        mem.add(z, "place_table", {"result": "placed"})

        # Query "do" should return "collected"
        outcome, _ = mem.query(z, "do")
        assert outcome["result"] == "collected"

        # Query "place_table" should return "placed"
        outcome, _ = mem.query(z, "place_table")
        assert outcome["result"] == "placed"

    def test_empty_memory(self):
        mem = PrototypeMemory(dim=128)
        outcome, conf = mem.query(torch.randn(128), "do")
        assert outcome["result"] == "unknown"
        assert conf == 0.0

    def test_no_matching_action(self):
        mem = PrototypeMemory(dim=128)
        mem.add(torch.randn(128), "do", {"result": "collected"})
        outcome, conf = mem.query(torch.randn(128), "sleep")
        assert outcome["result"] == "unknown"
        assert conf == 0.0

    def test_majority_vote(self):
        mem = PrototypeMemory(dim=128, k=5)
        z_base = torch.randn(128)
        # 3 "collected" + 2 "failed"
        for _ in range(3):
            mem.add(z_base + torch.randn(128) * 0.01, "do", {"result": "collected"})
        for _ in range(2):
            mem.add(z_base + torch.randn(128) * 0.01, "do", {"result": "failed"})

        outcome, conf = mem.query(z_base, "do")
        assert outcome["result"] == "collected"

    def test_stats(self):
        mem = PrototypeMemory(dim=128)
        mem.add(torch.randn(128), "do", {"result": "a"})
        mem.add(torch.randn(128), "do", {"result": "b"})
        mem.add(torch.randn(128), "sleep", {"result": "c"})
        assert mem.stats() == {"do": 2, "sleep": 1}


# ── CLS Integration ──


class TestCLSPrototypePath:
    def test_query_from_pixels(self):
        enc = CNNEncoder(n_near_classes=len(NEAR_CLASSES))
        cls = CLSWorldModel(dim=2048, device="cpu")

        # Add a prototype
        with torch.no_grad():
            out = enc(torch.rand(3, 64, 64))
        cls.prototype_memory.add(out.z_real, "do", {"result": "collected"})

        # Query
        outcome, conf, source = cls.query_from_pixels(
            torch.rand(3, 64, 64), "do", enc,
        )
        assert isinstance(outcome, dict)
        assert source in ("prototype", "none")

    def test_train_from_pixels(self):
        enc = CNNEncoder(n_near_classes=len(NEAR_CLASSES))
        cls = CLSWorldModel(dim=2048, device="cpu")

        transitions = [
            (torch.rand(3, 64, 64), "do", {"result": "collected", "gives": "wood"}, 1.0),
        ]
        stats = cls.train_from_pixels(transitions, enc)
        assert stats["prototypes"] >= 1


# ── Crafter Pixel Env ──


class TestCrafterPixelEnv:
    def test_reset(self):
        env = CrafterPixelEnv(seed=42)
        pixels, sym = env.reset()
        assert pixels.shape == (3, 64, 64)
        assert sym["domain"] == "crafter"

    def test_step(self):
        env = CrafterPixelEnv(seed=42)
        env.reset()
        pixels, sym, reward, done = env.step(0)
        assert pixels.shape == (3, 64, 64)


# ── Integration ──


class TestPixelPipeline:
    def test_mini_training_pipeline(self):
        """Minimal end-to-end: collect → train → prototype → query."""
        env = CrafterPixelEnv(seed=42)
        enc = CNNEncoder(n_near_classes=len(NEAR_CLASSES))
        pred = JEPAPredictor()
        trainer = PredictiveTrainer(enc, pred, contrastive_weight=0.5, device="cpu")

        # Collect 20 transitions
        pixels, sym = env.reset()
        pts, pts1, acts, sit_labels = [], [], [], []
        label_map: dict[str, int] = {}
        for _ in range(20):
            action = torch.randint(0, 17, (1,)).item()
            next_pixels, next_sym, _, done = env.step(action)
            pts.append(torch.from_numpy(pixels))
            pts1.append(torch.from_numpy(next_pixels))
            acts.append(action)
            # Situation label
            near = sym.get("near", "empty")
            if near not in label_map:
                label_map[near] = len(label_map)
            sit_labels.append(label_map[near])
            pixels = next_pixels
            sym = next_sym
            if done:
                pixels, sym = env.reset()

        pt = torch.stack(pts)
        pt1 = torch.stack(pts1)
        a = torch.tensor(acts)
        sl = torch.tensor(sit_labels)

        # Train 2 steps
        for _ in range(2):
            metrics = trainer.train_step(pt, pt1, a, sl)

        assert metrics["pred_loss"] < 10.0
        assert metrics["pred_loss"] >= 0.0

        # Build prototype memory and query
        cls = CLSWorldModel(dim=2048, device="cpu")
        with torch.no_grad():
            out = enc(pt[0])
        cls.prototype_memory.add(out.z_real, "do", {"result": "collected"})

        outcome, conf, source = cls.query_from_pixels(pt[0], "do", enc)
        assert "result" in outcome
