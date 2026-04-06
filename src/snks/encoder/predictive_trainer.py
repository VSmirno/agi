"""Stage 66: Mini-JEPA predictive trainer + supervised contrastive.

Self-supervised JEPA: predict next-frame latent from current latent + action.
Supervised contrastive (SupCon): cluster z_real by situation label.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
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


def supcon_loss(
    z: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Supervised Contrastive Loss (SupCon).

    Args:
        z: (B, D) L2-normalized embeddings.
        labels: (B,) integer class labels.
        temperature: scaling temperature.

    Returns:
        Scalar loss.
    """
    B = z.shape[0]
    if B < 2:
        return torch.tensor(0.0, device=z.device)

    z = F.normalize(z, dim=1)
    sim = z @ z.T / temperature  # (B, B)

    # Mask: same label (excluding self)
    label_eq = labels.unsqueeze(0) == labels.unsqueeze(1)  # (B, B)
    self_mask = ~torch.eye(B, dtype=torch.bool, device=z.device)
    pos_mask = label_eq & self_mask

    # Check that at least some positives exist
    has_pos = pos_mask.any(dim=1)
    if not has_pos.any():
        return torch.tensor(0.0, device=z.device)

    # Log-softmax over non-self entries
    logits = sim - sim.max(dim=1, keepdim=True).values  # stability
    exp_logits = torch.exp(logits) * self_mask.float()
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

    # Mean log-prob over positives
    pos_log_prob = (log_prob * pos_mask.float()).sum(dim=1)
    n_pos = pos_mask.float().sum(dim=1).clamp(min=1)
    loss = -(pos_log_prob / n_pos)

    # Average only over samples that have positives
    return loss[has_pos].mean()


class PredictiveTrainer:
    """Train VQ encoder + JEPA predictor on transition pairs."""

    def __init__(
        self,
        encoder: nn.Module,
        predictor: JEPAPredictor,
        lr: float = 1e-3,
        vicreg_weight: float = 0.1,
        contrastive_weight: float = 0.0,
        near_weight: float = 0.0,
        device: torch.device | str = "cpu",
    ):
        self.encoder = encoder
        self.predictor = predictor
        self.device = torch.device(device)
        self.vicreg_weight = vicreg_weight
        self.contrastive_weight = contrastive_weight
        self.near_weight = near_weight

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
        situation_labels: torch.Tensor | None = None,
        near_labels: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """One training step.

        Args:
            pixels_t: (B, 3, 64, 64) current frame.
            pixels_t1: (B, 3, 64, 64) next frame.
            actions: (B,) int action indices.
            situation_labels: (B,) int labels for SupCon (optional).
            near_labels: (B,) int near class indices for cross-entropy (optional).

        Returns:
            Dict with pred_loss, var_loss, con_loss, near_loss, total_loss.
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

        # Supervised contrastive loss
        con_loss_val = 0.0
        if self.contrastive_weight > 0 and situation_labels is not None:
            con = supcon_loss(z_t, situation_labels)
            con_loss_val = con.item()
            total = total + self.contrastive_weight * con

        # Near classification loss (cross-entropy on near_head output)
        near_loss_val = 0.0
        if self.near_weight > 0 and near_labels is not None:
            near_ce = nn.functional.cross_entropy(out_t.near_logits, near_labels)
            near_loss_val = near_ce.item()
            total = total + self.near_weight * near_ce

        self.optimizer.zero_grad()
        total.backward()
        self.optimizer.step()

        return {
            "pred_loss": pred_loss.item(),
            "var_loss": var_loss.item(),
            "con_loss": con_loss_val,
            "near_loss": near_loss_val,
            "total_loss": total.item(),
        }

    def train_epoch(
        self,
        pixels_t: torch.Tensor,
        pixels_t1: torch.Tensor,
        actions: torch.Tensor,
        situation_labels: torch.Tensor | None = None,
        near_labels: torch.Tensor | None = None,
        batch_size: int = 256,
    ) -> dict[str, float]:
        """Train one epoch over dataset.

        Args:
            pixels_t: (N, 3, 64, 64) all current frames.
            pixels_t1: (N, 3, 64, 64) all next frames.
            actions: (N,) all action indices.
            situation_labels: (N,) int labels for SupCon (optional).
            near_labels: (N,) int near class indices for cross-entropy (optional).
            batch_size: batch size.

        Returns:
            Averaged metrics over epoch.
        """
        tensors = [pixels_t, pixels_t1, actions]
        if situation_labels is not None:
            tensors.append(situation_labels)
        if near_labels is not None:
            # Pad situation_labels slot if absent
            if situation_labels is None:
                tensors.append(torch.zeros(len(pixels_t), dtype=torch.long))
            tensors.append(near_labels)

        dataset = TensorDataset(*tensors)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        totals: dict[str, float] = {}
        n_batches = 0

        for batch in loader:
            batch_pt = batch[0].to(self.device)
            batch_pt1 = batch[1].to(self.device)
            batch_a = batch[2].to(self.device)
            batch_sl = batch[3].to(self.device) if len(batch) > 3 else None
            batch_nl = batch[4].to(self.device) if len(batch) > 4 else None

            metrics = self.train_step(batch_pt, batch_pt1, batch_a, batch_sl, batch_nl)
            for k, v in metrics.items():
                totals[k] = totals.get(k, 0.0) + v
            n_batches += 1

        return {k: v / max(n_batches, 1) for k, v in totals.items()}

    def train_full(
        self,
        pixels_t: torch.Tensor,
        pixels_t1: torch.Tensor,
        actions: torch.Tensor,
        situation_labels: torch.Tensor | None = None,
        near_labels: torch.Tensor | None = None,
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
            metrics = self.train_epoch(
                pixels_t, pixels_t1, actions, situation_labels, near_labels, batch_size,
            )
            metrics["epoch"] = epoch

            history.append(metrics)

            if log_every and epoch % log_every == 0:
                parts = [f"Epoch {epoch}: pred={metrics['pred_loss']:.4f}"]
                parts.append(f"var={metrics['var_loss']:.4f}")
                if metrics.get("con_loss", 0) > 0:
                    parts.append(f"con={metrics['con_loss']:.4f}")
                if metrics.get("near_loss", 0) > 0:
                    parts.append(f"near={metrics['near_loss']:.4f}")
                print(" ".join(parts))

        return history
