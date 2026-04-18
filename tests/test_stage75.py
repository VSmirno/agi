"""Stage 75: Per-tile visual field tests.

Tests tile_head forward, classify_tiles, perceive_tile_field,
semantic_cell_label, and training loop basics.

All tests run locally without GPU.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from snks.encoder.cnn_encoder import CNNEncoder
from snks.agent.decode_head import NEAR_CLASSES
from snks.agent.perception import perceive_semantic_field, perceive_tile_field, VisualField


# ---------------------------------------------------------------------------
# CNNEncoder.tile_head basics
# ---------------------------------------------------------------------------


class TestTileHead:
    def test_tile_head_exists(self):
        encoder = CNNEncoder(feature_channels=256)
        assert hasattr(encoder, "tile_head")
        assert encoder.tile_head.in_features == 256
        assert encoder.tile_head.out_features == 12

    def test_tile_head_custom_classes(self):
        encoder = CNNEncoder(n_near_classes=8)
        assert encoder.tile_head.out_features == 8

    def test_tile_head_forward_single(self):
        encoder = CNNEncoder(feature_channels=256)
        feat = torch.randn(256)
        logits = encoder.tile_head(feat)
        assert logits.shape == (12,)

    def test_tile_head_forward_batch(self):
        encoder = CNNEncoder(feature_channels=256)
        feat = torch.randn(16, 256)
        logits = encoder.tile_head(feat)
        assert logits.shape == (16, 12)


# ---------------------------------------------------------------------------
# classify_tiles
# ---------------------------------------------------------------------------


class TestClassifyTiles:
    def test_single_frame(self):
        encoder = CNNEncoder(feature_channels=256)
        encoder.eval()
        pixels = torch.randn(3, 64, 64)
        class_ids, confidences = encoder.classify_tiles(pixels)
        assert class_ids.shape == (4, 4)
        assert confidences.shape == (4, 4)
        assert class_ids.dtype == torch.int64
        assert confidences.min() >= 0.0
        assert confidences.max() <= 1.0

    def test_batch_frame(self):
        encoder = CNNEncoder(feature_channels=256)
        encoder.eval()
        pixels = torch.randn(3, 3, 64, 64)
        class_ids, confidences = encoder.classify_tiles(pixels)
        assert class_ids.shape == (3, 4, 4)
        assert confidences.shape == (3, 4, 4)

    def test_class_range(self):
        encoder = CNNEncoder(feature_channels=256)
        encoder.eval()
        pixels = torch.randn(3, 64, 64)
        class_ids, _ = encoder.classify_tiles(pixels)
        assert class_ids.min() >= 0
        assert class_ids.max() < 12


# ---------------------------------------------------------------------------
# perceive_tile_field
# ---------------------------------------------------------------------------


class TestPerceiveTileField:
    def test_returns_visual_field(self):
        encoder = CNNEncoder(feature_channels=256)
        encoder.eval()
        pixels = torch.randn(3, 64, 64)
        vf = perceive_tile_field(pixels, encoder)
        assert isinstance(vf, VisualField)
        assert vf.near_concept is not None
        assert vf.raw_center_feature is not None
        assert vf.center_feature is not None

    def test_detections_format(self):
        encoder = CNNEncoder(feature_channels=256)
        encoder.eval()
        pixels = torch.randn(3, 64, 64)
        vf = perceive_tile_field(pixels, encoder, min_confidence=0.0)
        for cid, conf, gy, gx in vf.detections:
            assert isinstance(cid, str)
            assert 0.0 <= conf <= 1.0
            assert 0 <= gy < 4
            assert 0 <= gx < 4

    def test_near_concept_from_center(self):
        encoder = CNNEncoder(feature_channels=256)
        encoder.eval()
        pixels = torch.randn(3, 64, 64)
        vf = perceive_tile_field(pixels, encoder, min_confidence=0.0)
        # near_concept should be from center positions
        assert vf.near_concept in NEAR_CLASSES or vf.near_concept.startswith("class_")

    def test_visible_concepts(self):
        encoder = CNNEncoder(feature_channels=256)
        encoder.eval()
        pixels = torch.randn(3, 64, 64)
        vf = perceive_tile_field(pixels, encoder, min_confidence=0.0)
        visible = vf.visible_concepts()
        assert isinstance(visible, set)

    def test_find_method(self):
        encoder = CNNEncoder(feature_channels=256)
        encoder.eval()
        pixels = torch.randn(3, 64, 64)
        vf = perceive_tile_field(pixels, encoder, min_confidence=0.0)
        result = vf.find("empty")
        assert isinstance(result, list)


class TestPerceiveSemanticField:
    def test_returns_visual_field_from_semantic_info(self):
        semantic = np.full((64, 64), 2, dtype=np.int32)  # grass
        semantic[32, 32] = 15  # zombie near center
        info = {"semantic": semantic, "player_pos": (32, 32)}

        vf = perceive_semantic_field(info)

        assert isinstance(vf, VisualField)
        assert vf.near_similarity == 1.0

    def test_detects_arrow_and_center_empty_only(self):
        semantic = np.full((64, 64), 2, dtype=np.int32)  # grass
        semantic[32, 31] = 17  # arrow on center-ish viewport tile
        info = {"semantic": semantic, "player_pos": (32, 32)}

        vf = perceive_semantic_field(info)

        visible = vf.visible_concepts()
        assert "arrow" in visible
        empties = [(cid, gy, gx) for cid, _conf, gy, gx in vf.detections if cid == "empty"]
        assert len(empties) <= 4


# ---------------------------------------------------------------------------
# semantic_cell_label
# ---------------------------------------------------------------------------


class TestSemanticCellLabel:
    def test_all_grass_returns_empty(self):
        from snks.encoder.tile_head_trainer import semantic_cell_label
        semantic = np.full((64, 64), 2, dtype=np.int32)  # 2 = grass
        label = semantic_cell_label(semantic, 0, 0, 4)
        assert label == 0  # empty

    def test_tree_detected(self):
        from snks.encoder.tile_head_trainer import semantic_cell_label
        semantic = np.full((64, 64), 2, dtype=np.int32)  # grass
        # Put tree (6) in top-left cell
        semantic[0:16, 0:16] = 6  # tree
        label = semantic_cell_label(semantic, 0, 0, 4)
        # tree is at index 2 in NEAR_CLASSES (["empty", "water", "tree", ...])
        assert NEAR_CLASSES[label] == "tree"

    def test_zombie_detected(self):
        from snks.encoder.tile_head_trainer import semantic_cell_label
        semantic = np.full((64, 64), 2, dtype=np.int32)  # grass
        semantic[0:16, 0:16] = 15  # zombie
        label = semantic_cell_label(semantic, 0, 0, 4)
        assert NEAR_CLASSES[label] == "zombie"

    def test_minority_below_threshold_is_empty(self):
        from snks.encoder.tile_head_trainer import semantic_cell_label
        semantic = np.full((64, 64), 2, dtype=np.int32)  # grass
        # Only 9/256 pixels = 3.5% coal — below 20% threshold
        semantic[2:5, 2:5] = 8  # coal
        label = semantic_cell_label(semantic, 0, 0, 4)
        assert NEAR_CLASSES[label] == "empty"

    def test_majority_object_detected(self):
        from snks.encoder.tile_head_trainer import semantic_cell_label
        semantic = np.full((64, 64), 2, dtype=np.int32)  # grass
        # Coal fills 60% of cell → above threshold
        semantic[0:10, 0:16] = 8  # coal: 160/256 = 62.5%
        label = semantic_cell_label(semantic, 0, 0, 4)
        assert NEAR_CLASSES[label] == "coal"


# ---------------------------------------------------------------------------
# Training integration (minimal)
# ---------------------------------------------------------------------------


class TestTrainIntegration:
    def test_train_on_synthetic(self):
        from snks.encoder.tile_head_trainer import train_tile_head

        encoder = CNNEncoder(feature_channels=256)

        # Synthetic: 3 classes, 100 samples each
        features = torch.randn(300, 256)
        labels = torch.tensor([0] * 100 + [1] * 100 + [2] * 100)

        stats = train_tile_head(
            encoder, features, labels,
            epochs=10, lr=1e-2, batch_size=64,
        )

        assert "train_acc" in stats
        assert stats["train_acc"] > 0.0
