"""HebbianEncoder: learnable visual encoder via competitive Hebbian rule.

Extends VisualEncoder — initialized from Gabor filters, then learns
via local competitive Hebbian updates modulated by prediction error.

Key properties:
- No backprop — only local learning rule (Sanger 1989 / competitive Hebbian)
- Competitive: only top-K active filters updated per image (lateral inhibition)
- Sanger's triangular decorrelation: each filter extracts a different PC
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
    """Learnable encoder using competitive Hebbian rule on conv filters.

    Sanger's Generalized Hebbian Algorithm (GHA) per filter f:
        Δw_f = η_eff × post_f × (pre - Σ_{g≤f} post_g × w_g)

    Only the top-K most active filters are updated (competitive learning).
    This naturally leads to different filters extracting different features.

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
        competitive_k_ratio: float = 0.25,
    ) -> None:
        super().__init__(config)
        self.lr = lr
        self.pe_baseline_init = pe_baseline
        self.pe_ema_alpha = pe_ema_alpha
        self.diversity_interval = diversity_interval
        self.diversity_threshold = diversity_threshold
        self.w_min = w_min
        self.w_max = w_max
        self.competitive_k_ratio = competitive_k_ratio

        # Adaptive PE baseline (EMA)
        self._pe_baseline = pe_baseline
        self._update_count = 0

    def hebbian_update(
        self,
        image: torch.Tensor,
        sdr: torch.Tensor,
        prediction_error: float,
    ) -> float:
        """Apply one competitive Hebbian learning step.

        Uses Sanger's GHA with competitive selection:
        1. Compute post-activations per filter
        2. Select top-K active filters (winners)
        3. Apply Sanger's rule (triangular decorrelation) to winners only

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

        # Pre-synaptic: extract mean input patches via unfold
        x = image.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        padding = kH // 2
        patches = F.unfold(x, kernel_size=(kH, kW), padding=padding)  # (1, kH*kW, L)
        pre = patches.squeeze(0).mean(dim=1)  # (kH*kW,)

        # Post-synaptic: activations per filter
        with torch.no_grad():
            features = self.gabor(x)  # (1, n_filters, H, W)
            pooled = self.pool(features)  # (1, n_filters, pool_h, pool_w)
            post = pooled.squeeze(0).mean(dim=(1, 2))  # (n_filters,)

        # Competitive selection: only top-K active filters get updated
        competitive_k = max(1, int(n_filters * self.competitive_k_ratio))
        _, winner_idx = torch.topk(post, competitive_k)

        w_flat = weight.view(n_filters, -1)  # (n_filters, kH*kW)

        # Sanger's GHA for winners (triangular decorrelation):
        # For winner i (sorted by activation, highest first):
        #   residual_i = pre - Σ_{j<i among winners} post_j × w_j
        #   Δw_i = η × post_i × (residual_i - post_i × w_i)
        # This ensures each winner extracts a different component.

        # Sort winners by activation (descending) for proper triangular ordering
        winner_post = post[winner_idx]
        sort_order = torch.argsort(winner_post, descending=True)
        sorted_idx = winner_idx[sort_order]
        sorted_post = winner_post[sort_order]

        residual = pre.clone()  # Start with full input
        total_delta = torch.zeros(1, device=weight.device)

        for rank in range(competitive_k):
            f = sorted_idx[rank].item()
            y_f = sorted_post[rank].item()

            if y_f < 1e-8:
                continue  # Skip inactive filters

            # Sanger update: Δw = η × y_f × (residual - y_f × w_f)
            delta = eta * y_f * (residual - y_f * w_flat[f])
            w_flat[f] += delta
            total_delta += delta.abs().mean()

            # Subtract this filter's contribution from residual (triangular)
            residual = residual - y_f * w_flat[f]

        # Clamp weights (REQUIRED — PE modulation breaks self-normalization)
        weight.clamp_(self.w_min, self.w_max)

        mean_delta = total_delta.item() / max(competitive_k, 1)

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
            # Push j away from i with noise + orthogonal direction
            noise = torch.randn_like(w_flat[j]) * 0.1
            direction = w_flat[j] - w_flat[i]
            w_flat[j] += noise + 0.1 * direction
            perturbed.add(j)

        # Re-normalize perturbed filters
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
