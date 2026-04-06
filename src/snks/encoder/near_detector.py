"""Stage 67: NearDetector — CNN-based near object detection.

Replaces CrafterPixelEnv._to_symbolic() for the "near" field.
Uses the CNN encoder trained in Stage 66 (JEPA + SupCon).
"""

from __future__ import annotations

import torch

from snks.agent.decode_head import NEAR_CLASSES
from snks.encoder.cnn_encoder import CNNEncoder


class NearDetector:
    """Detects the nearest object from pixel observations using CNN encoder.

    Wraps CNNEncoder.near_head output → argmax → NEAR_CLASSES string.
    Replaces the symbolic _to_symbolic()["near"] path (Stage 67).
    """

    def __init__(self, encoder: CNNEncoder) -> None:
        """
        Args:
            encoder: trained CNNEncoder (Stage 66 checkpoint).
                     Caller is responsible for loading weights.
        """
        n_classes = encoder.near_head[-1].out_features
        assert n_classes == len(NEAR_CLASSES), (
            f"Encoder n_near_classes={n_classes} != len(NEAR_CLASSES)={len(NEAR_CLASSES)}. "
            f"Ensure the encoder was trained with the same NEAR_CLASSES list."
        )
        self._encoder = encoder

    def detect(self, pixels: torch.Tensor) -> str:
        """Detect nearest object from pixel observation.

        Args:
            pixels: (3, 64, 64) float32 [0, 1]. CNNEncoder handles single input.

        Returns:
            Near object name from NEAR_CLASSES (e.g. "tree", "stone", "empty").
            Falls back to "empty" if index is out of range.
        """
        self._encoder.eval()
        device = next(self._encoder.parameters()).device
        with torch.no_grad():
            out = self._encoder(pixels.to(device))
        # single input → near_logits shape: (n_classes,)
        idx = int(out.near_logits.argmax().item())
        if idx < len(NEAR_CLASSES):
            return NEAR_CLASSES[idx]
        return "empty"
