"""Stage 66: Pixel perception gate test.

Gate: ≥50% Crafter QA from pixel input.

Tests run locally with CPU — small scale, fast.
Full experiment (exp122_pixels.py) runs on minipc with GPU.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from snks.encoder.vq_patch_encoder import VQPatchEncoder, EncoderOutput
from snks.encoder.predictive_trainer import JEPAPredictor, PredictiveTrainer
from snks.agent.decode_head import (
    DecodeHead, symbolic_to_gt_tensors, NEAR_CLASSES, INVENTORY_ITEMS,
)
from snks.agent.cls_world_model import CLSWorldModel
from snks.agent.crafter_pixel_env import CrafterPixelEnv, ACTION_NAMES


# ── Unit Tests ──


class TestVQPatchEncoder:
    def test_output_shapes(self):
        enc = VQPatchEncoder(device="cpu")
        pixels = torch.rand(4, 3, 64, 64)
        out = enc(pixels)
        assert out.z_real.shape == (4, 2048)
        assert out.z_vsa.shape == (4, 2048)
        assert out.indices.shape == (4, 64)

    def test_single_input(self):
        enc = VQPatchEncoder(device="cpu")
        pixels = torch.rand(3, 64, 64)
        out = enc(pixels)
        assert out.z_real.shape == (2048,)
        assert out.z_vsa.shape == (2048,)
        assert out.indices.shape == (64,)

    def test_z_vsa_is_binary(self):
        enc = VQPatchEncoder(device="cpu")
        out = enc(torch.rand(2, 3, 64, 64))
        unique = torch.unique(out.z_vsa)
        assert set(unique.tolist()).issubset({0.0, 1.0})

    def test_z_vsa_roughly_balanced(self):
        enc = VQPatchEncoder(device="cpu")
        out = enc(torch.rand(4, 3, 64, 64))
        mean = out.z_vsa.mean().item()
        assert 0.3 < mean < 0.7, f"z_vsa mean={mean}, too skewed"

    def test_kmeans_init(self):
        enc = VQPatchEncoder(device="cpu")
        patches = enc._patchify(torch.rand(8, 3, 64, 64))
        enc.init_codebook_kmeans(patches)
        assert enc._initialized

    def test_codebook_ema_update(self):
        enc = VQPatchEncoder(device="cpu")
        pixels = torch.rand(4, 3, 64, 64)
        out = enc(pixels)
        patches = enc._patchify(pixels)
        old_cb = enc.codebook.clone()
        enc.update_codebook_ema(patches, out.indices)
        # Codebook should have changed for used entries
        assert not torch.allclose(old_cb, enc.codebook)

    def test_dead_entry_reset(self):
        enc = VQPatchEncoder(device="cpu")
        pixels = torch.rand(4, 3, 64, 64)
        patches = enc._patchify(pixels)
        # Most entries will be dead (4*64 = 256 patches, 4096 entries)
        n_reset = enc.reset_dead_entries(patches, min_usage=1)
        assert n_reset > 0


class TestJEPAPredictor:
    def test_forward(self):
        pred = JEPAPredictor()
        z = torch.randn(4, 2048)
        actions = torch.randint(0, 17, (4,))
        out = pred(z, actions)
        assert out.shape == (4, 2048)

    def test_train_step(self):
        enc = VQPatchEncoder(device="cpu")
        pred = JEPAPredictor()
        trainer = PredictiveTrainer(enc, pred, device="cpu")
        metrics = trainer.train_step(
            torch.rand(4, 3, 64, 64),
            torch.rand(4, 3, 64, 64),
            torch.randint(0, 17, (4,)),
        )
        assert "pred_loss" in metrics
        assert "var_loss" in metrics
        assert all(v >= 0 for v in metrics.values())


class TestDecodeHead:
    def test_forward(self):
        head = DecodeHead()
        agent_idx = torch.randint(0, 4096, (4, 9))
        z_local = torch.randn(4, 2048)
        out = head(agent_idx, z_local)
        assert out["near_logits"].shape == (4, len(NEAR_CLASSES))
        assert out["inventory_logits"].shape == (4, len(INVENTORY_ITEMS))

    def test_decode_situation_key(self):
        head = DecodeHead()
        agent_idx = torch.randint(0, 4096, (9,))
        key, certainty = head.decode_situation_key(agent_idx)
        assert isinstance(key, str)
        assert key.startswith("crafter_")
        assert 0.0 <= certainty <= 1.0

    def test_symbolic_to_gt(self):
        sym = {"domain": "crafter", "near": "tree", "has_wood": "3"}
        near_idx, inv_vec = symbolic_to_gt_tensors(sym)
        assert NEAR_CLASSES[near_idx] == "tree"
        assert inv_vec[0] == 1.0  # wood
        assert inv_vec[1] == 0.0  # stone

    def test_train_step(self):
        head = DecodeHead()
        opt = torch.optim.Adam(head.parameters())
        metrics = head.train_step(
            torch.randint(0, 4096, (4, 9)),
            torch.randint(0, len(NEAR_CLASSES), (4,)),
            torch.randint(0, 2, (4, len(INVENTORY_ITEMS))).float(),
            opt,
            z_local=torch.randn(4, 2048),
        )
        assert metrics["near_acc"] >= 0.0


class TestCrafterPixelEnv:
    def test_reset(self):
        env = CrafterPixelEnv(seed=42)
        pixels, sym = env.reset()
        assert pixels.shape == (3, 64, 64)
        assert pixels.dtype == np.float32
        assert 0.0 <= pixels.min() and pixels.max() <= 1.0
        assert sym["domain"] == "crafter"

    def test_step(self):
        env = CrafterPixelEnv(seed=42)
        env.reset()
        pixels, sym, reward, done = env.step(0)  # noop
        assert pixels.shape == (3, 64, 64)
        assert isinstance(reward, float)
        assert isinstance(done, bool)

    def test_step_by_name(self):
        env = CrafterPixelEnv(seed=42)
        env.reset()
        pixels, sym, reward, done = env.step("do")
        assert pixels.shape == (3, 64, 64)


class TestCLSDualPath:
    def test_query_from_pixels(self):
        enc = VQPatchEncoder(device="cpu")
        head = DecodeHead()
        cls = CLSWorldModel(dim=2048, device="cpu")

        pixels = torch.rand(3, 64, 64)
        outcome, conf, source = cls.query_from_pixels(pixels, "do", enc, head)
        assert isinstance(outcome, dict)
        assert "result" in outcome
        assert 0.0 <= conf <= 1.0
        assert source in ("neocortex", "hippocampus", "none")

    def test_train_from_pixels(self):
        enc = VQPatchEncoder(device="cpu")
        head = DecodeHead()
        cls = CLSWorldModel(dim=2048, device="cpu")

        transitions = [
            (torch.rand(3, 64, 64), "do", {"result": "collected", "gives": "wood"}, 1.0),
        ]
        stats = cls.train_from_pixels(transitions, enc, head)
        assert stats["sdm_writes"] >= 1


# ── Integration Test (needs real Crafter) ──


class TestPixelPipeline:
    """Integration test: encoder → CLS → QA with real Crafter frames."""

    def test_encode_real_crafter_frame(self):
        env = CrafterPixelEnv(seed=42)
        pixels, sym = env.reset()
        enc = VQPatchEncoder(device="cpu")
        out = enc(torch.from_numpy(pixels))
        assert out.z_real.shape == (2048,)
        assert out.z_vsa.shape == (2048,)

    def test_mini_training_pipeline(self):
        """Minimal end-to-end: collect → train → query."""
        env = CrafterPixelEnv(seed=42)
        enc = VQPatchEncoder(device="cpu")
        pred = JEPAPredictor()
        trainer = PredictiveTrainer(enc, pred, device="cpu")

        # Collect 20 transitions
        pixels, sym = env.reset()
        pts, pts1, acts = [], [], []
        for _ in range(20):
            action = torch.randint(0, 17, (1,)).item()
            next_pixels, next_sym, _, done = env.step(action)
            pts.append(torch.from_numpy(pixels))
            pts1.append(torch.from_numpy(next_pixels))
            acts.append(action)
            pixels = next_pixels
            if done:
                pixels, sym = env.reset()

        pt = torch.stack(pts)
        pt1 = torch.stack(pts1)
        a = torch.tensor(acts)

        # Train 2 steps
        for _ in range(2):
            metrics = trainer.train_step(pt, pt1, a)

        assert metrics["pred_loss"] < 10.0  # sanity
        assert metrics["pred_loss"] >= 0.0

        # Query CLS
        head = DecodeHead()
        cls = CLSWorldModel(dim=2048, device="cpu")
        outcome, conf, source = cls.query_from_pixels(
            pt[0], "do", enc, head,
        )
        assert "result" in outcome
