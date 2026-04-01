"""HebbianEncoder: learnable visual encoder via Oja's Hebbian rule.

Extends VisualEncoder — initialized from Gabor filters, then learns
via local Hebbian updates modulated by prediction error.

Key properties:
- No backprop — only local learning rule (Oja 1982)
- PE modulation: high surprise → faster learning
- Diversity regularization: prevents filter collapse
- Drop-in replacement for VisualEncoder
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from snks.daf.types import EncoderConfig
from snks.encoder.encoder import VisualEncoder


class HebbianEncoder(VisualEncoder):
    """Learnable encoder using Oja's Hebbian rule on conv filters.

    Oja's rule per filter f:
        Δw_f = η_eff × <post_f> × (<pre_f> - <post_f> × w_f)

    where:
        pre_f = mean input patch under filter f
        post_f = mean pooled activation of filter f
        η_eff = η_base × clamp(PE / PE_baseline, 0.1, 2.0)

    PE modulation breaks Oja's self-normalization, so weight
    clamping to [w_min, w_max] is REQUIRED (not just safety).
    """

    def __init__(
        self,
        config: EncoderConfig,
        lr: float = 0.001,
        pe_baseline: float = 0.3,
        pe_ema_alpha: float = 0.05,
        diversity_interval: int = 50,
        diversity_threshold: float = 0.8,
        w_min: float = -2.0,
        w_max: float = 2.0,
    ) -> None:
        super().__init__(config)
        self.lr = lr
        self.pe_baseline_init = pe_baseline
        self.pe_ema_alpha = pe_ema_alpha
        self.diversity_interval = diversity_interval
        self.diversity_threshold = diversity_threshold
        self.w_min = w_min
        self.w_max = w_max

        # Adaptive PE baseline (EMA)
        self._pe_baseline = pe_baseline
        self._update_count = 0

    def hebbian_update(
        self,
        image: torch.Tensor,
        sdr: torch.Tensor,
        prediction_error: float,
    ) -> float:
        """Apply one Hebbian learning step to conv filters.

        Args:
            image: (H, W) grayscale float32 input image.
            sdr: (sdr_size,) binary SDR from encode().
            prediction_error: PE from pipeline (0 = expected, 1 = surprising).

        Returns:
            Mean absolute weight change (for monitoring convergence).
        """
        # Effective learning rate modulated by PE
        pe_ratio = prediction_error / max(self._pe_baseline, 1e-6)
        pe_mod = max(0.1, min(pe_ratio, 2.0))
        eta = self.lr * pe_mod

        # Update PE baseline (EMA)
        self._pe_baseline = (
            (1 - self.pe_ema_alpha) * self._pe_baseline
            + self.pe_ema_alpha * prediction_error
        )

        weight = self.gabor.conv.weight.data  # (n_filters, 1, kH, kW)
        n_filters = weight.shape[0]
        kH, kW = weight.shape[2], weight.shape[3]

        # Pre-synaptic: extract mean input patches per filter via unfold
        # image: (H, W) → (1, 1, H, W)
        x = image.unsqueeze(0).unsqueeze(0)
        padding = kH // 2
        # Unfold: (1, 1*kH*kW, L) where L = number of spatial positions
        patches = F.unfold(x, kernel_size=(kH, kW), padding=padding)  # (1, kH*kW, L)
        # Mean across spatial positions → (kH*kW,) = mean input patch
        pre = patches.squeeze(0).mean(dim=1)  # (kH*kW,)

        # Post-synaptic: pooled activations per filter (before k-WTA)
        with torch.no_grad():
            features = self.gabor(x)  # (1, n_filters, H, W)
            pooled = self.pool(features)  # (1, n_filters, pool_h, pool_w)
            # Mean activation per filter across spatial dimensions
            post = pooled.squeeze(0).mean(dim=(1, 2))  # (n_filters,)

        # Oja's rule per filter:
        # Δw_f = η × post_f × (pre - post_f × w_f)
        # w_f shape: (1, kH, kW) → flatten to (kH*kW,)
        w_flat = weight.view(n_filters, -1)  # (n_filters, kH*kW)

        # pre: (kH*kW,) broadcast to (n_filters, kH*kW)
        # post: (n_filters,) → (n_filters, 1) for broadcasting
        post_col = post.unsqueeze(1)  # (n_filters, 1)

        # Δw = η × post × (pre - post × w)
        delta_w = eta * post_col * (pre.unsqueeze(0) - post_col * w_flat)

        # Apply update in-place
        w_flat.add_(delta_w)

        # Clamp weights (REQUIRED — PE modulation breaks self-normalization)
        weight.clamp_(self.w_min, self.w_max)

        # Track convergence
        mean_delta = delta_w.abs().mean().item()

        # Diversity regularization at intervals
        self._update_count += 1
        if self._update_count % self.diversity_interval == 0:
            self._apply_diversity_regularization()

        return mean_delta

    def _apply_diversity_regularization(self) -> None:
        """Decorrelate highly similar filters to prevent collapse.

        Computes cosine similarity matrix. For pairs with
        similarity > threshold, perturbs one filter with noise
        proportional to the difference, then re-normalizes.
        """
        weight = self.gabor.conv.weight.data  # (n_filters, 1, kH, kW)
        n_filters = weight.shape[0]
        w_flat = weight.view(n_filters, -1)  # (n_filters, D)

        # Cosine similarity matrix
        w_norm = F.normalize(w_flat, dim=1)  # (n_filters, D)
        sim = w_norm @ w_norm.T  # (n_filters, n_filters)

        # Find pairs above threshold (excluding diagonal)
        mask = (sim.abs() > self.diversity_threshold) & (
            ~torch.eye(n_filters, dtype=torch.bool, device=weight.device)
        )

        if not mask.any():
            return

        # For each correlated pair, perturb the second filter
        rows, cols = mask.nonzero(as_tuple=True)
        perturbed = set()
        for i, j in zip(rows.tolist(), cols.tolist()):
            if j in perturbed or i >= j:
                continue
            # Push j away from i
            noise = torch.randn_like(w_flat[j]) * 0.1
            direction = w_flat[j] - w_flat[i]
            w_flat[j] += noise + 0.05 * direction
            perturbed.add(j)

        # Re-normalize perturbed filters to prevent drift
        for j in perturbed:
            norm = w_flat[j].norm()
            if norm > 1e-8:
                w_flat[j] /= norm

        # Clamp after perturbation
        weight.clamp_(self.w_min, self.w_max)

    @property
    def stats(self) -> dict:
        """Return encoder statistics for monitoring."""
        weight = self.gabor.conv.weight.data
        n_filters = weight.shape[0]
        w_flat = weight.view(n_filters, -1)
        w_norm = F.normalize(w_flat, dim=1)
        sim = w_norm @ w_norm.T

        # Mean off-diagonal similarity
        mask = ~torch.eye(n_filters, dtype=torch.bool, device=weight.device)
        mean_sim = sim[mask].abs().mean().item()

        return {
            "update_count": self._update_count,
            "pe_baseline": self._pe_baseline,
            "mean_weight": weight.mean().item(),
            "weight_std": weight.std().item(),
            "mean_filter_similarity": mean_sim,
            "weight_max": weight.max().item(),
            "weight_min": weight.min().item(),
        }
