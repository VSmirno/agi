"""Stage 66: Mini-JEPA predictive trainer.

Self-supervised training: predict next-frame latent from current latent + action.
Trains the VQ Patch Encoder embeddings via straight-through gradient
and the predictor MLP directly.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class JEPAPredictor(nn.Module):
    """Predict z_real[t+1] from z_real[t] + action embedding."""

    def __init__(
        self,
        z_dim: int = 2048,
        action_dim: int = 64,
        hidden: int = 1024,
        n_actions: int = 17,
    ):
        super().__init__()
        self.action_embed = nn.Embedding(n_actions, action_dim)
        self.mlp = nn.Sequential(
            nn.Linear(z_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, z_dim),
        )

    def forward(self, z: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Predict next latent.

        Args:
            z: (B, z_dim) current latent.
            actions: (B,) int action indices.

        Returns:
            (B, z_dim) predicted next latent.
        """
        a = self.action_embed(actions)  # (B, action_dim)
        return self.mlp(torch.cat([z, a], dim=1))


class PredictiveTrainer:
    """Train VQ encoder + JEPA predictor on transition pairs."""

    def __init__(
        self,
        encoder: nn.Module,
        predictor: JEPAPredictor,
        lr: float = 1e-3,
        vicreg_weight: float = 0.1,
        device: torch.device | str = "cpu",
    ):
        self.encoder = encoder
        self.predictor = predictor
        self.device = torch.device(device)
        self.vicreg_weight = vicreg_weight

        # Optimize all encoder params + predictor
        self.optimizer = torch.optim.Adam(
            list(encoder.parameters()) +
            list(predictor.parameters()),
            lr=lr,
        )

    def train_step(
        self,
        pixels_t: torch.Tensor,
        pixels_t1: torch.Tensor,
        actions: torch.Tensor,
    ) -> dict[str, float]:
        """One training step.

        Args:
            pixels_t: (B, 3, 64, 64) current frame.
            pixels_t1: (B, 3, 64, 64) next frame.
            actions: (B,) int action indices.

        Returns:
            Dict with pred_loss, var_loss, total_loss.
        """
        self.encoder.train()
        self.predictor.train()

        out_t = self.encoder(pixels_t)
        z_t = out_t.z_real  # (B, 2048)

        with torch.no_grad():
            out_t1 = self.encoder(pixels_t1)
            z_t1 = out_t1.z_real  # (B, 2048) — stop gradient target

        z_pred = self.predictor(z_t, actions)  # (B, 2048)

        # Predictive loss
        pred_loss = nn.functional.mse_loss(z_pred, z_t1)

        # VICReg variance: penalize collapsed dimensions
        std_z = z_t.std(dim=0)  # (2048,)
        var_loss = self.vicreg_weight * torch.relu(1.0 - std_z).mean()

        total = pred_loss + var_loss

        self.optimizer.zero_grad()
        total.backward()
        self.optimizer.step()

        return {
            "pred_loss": pred_loss.item(),
            "var_loss": var_loss.item(),
            "total_loss": total.item(),
        }

    def train_epoch(
        self,
        pixels_t: torch.Tensor,
        pixels_t1: torch.Tensor,
        actions: torch.Tensor,
        batch_size: int = 256,
    ) -> dict[str, float]:
        """Train one epoch over dataset.

        Args:
            pixels_t: (N, 3, 64, 64) all current frames.
            pixels_t1: (N, 3, 64, 64) all next frames.
            actions: (N,) all action indices.
            batch_size: batch size.

        Returns:
            Averaged metrics over epoch.
        """
        dataset = TensorDataset(pixels_t, pixels_t1, actions)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        totals: dict[str, float] = {}
        n_batches = 0

        for batch_pt, batch_pt1, batch_a in loader:
            batch_pt = batch_pt.to(self.device)
            batch_pt1 = batch_pt1.to(self.device)
            batch_a = batch_a.to(self.device)

            metrics = self.train_step(batch_pt, batch_pt1, batch_a)
            for k, v in metrics.items():
                totals[k] = totals.get(k, 0.0) + v
            n_batches += 1

        return {k: v / max(n_batches, 1) for k, v in totals.items()}

    def train_full(
        self,
        pixels_t: torch.Tensor,
        pixels_t1: torch.Tensor,
        actions: torch.Tensor,
        epochs: int = 100,
        batch_size: int = 256,
        log_every: int = 10,
    ) -> list[dict[str, float]]:
        """Full training loop.

        Returns:
            List of per-epoch metrics.
        """
        history = []
        for epoch in range(epochs):
            metrics = self.train_epoch(pixels_t, pixels_t1, actions, batch_size)
            metrics["epoch"] = epoch
            metrics["codebook_util"] = getattr(self.encoder, 'codebook_utilization', 1.0)

            history.append(metrics)

            if log_every and epoch % log_every == 0:
                print(
                    f"Epoch {epoch}: pred={metrics['pred_loss']:.4f} "
                    f"var={metrics['var_loss']:.4f} "
                    f"util={metrics['codebook_util']:.2f}"
                )

        return history
