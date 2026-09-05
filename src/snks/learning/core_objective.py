"""Gaussian characteristic-function SIGReg and missing-value-safe regression.

Recipe: https://github.com/galilai-group/lejepa/blob/main/MINIMAL.md
Symmetric Gaussian ECF quadrature uses 17 points on [0,3] and twice the
half-axis integral. This profile uses CNN/GRU dynamics instead of the source's
ViT, with end-to-end targets, no EMA teacher, and no final normalization.
"""

import torch
from torch import Tensor


def sigreg(z: Tensor, directions: Tensor) -> Tensor:
    """Penalize deviation of real encoder outputs from a spherical Gaussian."""
    if z.ndim != 2 or len(z) < 2:
        raise ValueError("SIGReg requires at least two valid examples")
    if directions.ndim != 2 or directions.shape[0] != z.shape[1] or directions.shape[1] == 0:
        raise ValueError("directions must have shape D,K with K positive")
    # Disabling autocast also keeps the matrix product and trigonometry in fp32.
    with torch.autocast(device_type=z.device.type, enabled=False):
        points = torch.linspace(0, 3, 17, device=z.device, dtype=torch.float32)
        phases = (z.float() @ directions.float()).unsqueeze(-1) * points
        reference = torch.exp(-0.5 * points.square())
        discrepancy = ((phases.cos().mean(0) - reference).square()
                       + phases.sin().mean(0).square())
        return 2 * len(z) * torch.trapezoid(discrepancy * reference, points, dim=-1).mean()


def masked_mse(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """Average present values only; absent NaNs cannot contaminate gradients."""
    pred, target = torch.broadcast_tensors(pred, target)
    expanded = torch.broadcast_to(mask.bool(), pred.shape)
    present_pred = pred.masked_select(expanded)
    present_target = target.masked_select(expanded)
    if present_pred.numel() == 0:
        return present_pred.sum()
    return (present_pred - present_target).square().mean()
