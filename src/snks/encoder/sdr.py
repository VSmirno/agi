"""Sparse Distributed Representation (SDR) utilities."""

import torch


def kwta(x: torch.Tensor, k: int) -> torch.Tensor:
    """k-Winner-Take-All: keep top-k values as 1, rest as 0.

    Args:
        x: (..., D) activation vector(s).
        k: number of winners.

    Returns:
        (..., D) binary SDR, float32.
    """
    topk_vals, topk_idx = torch.topk(x, k, dim=-1)
    sdr = torch.zeros_like(x)
    sdr.scatter_(-1, topk_idx, 1.0)
    return sdr


def sdr_overlap(a: torch.Tensor, b: torch.Tensor, k: int) -> float:
    """Overlap between two SDRs: |A ∩ B| / k.

    Args:
        a, b: (D,) binary SDR vectors.
        k: number of active bits.

    Returns:
        Overlap fraction in [0, 1].
    """
    return (a * b).sum().item() / k


def batch_overlap_matrix(sdrs: torch.Tensor, k: int) -> torch.Tensor:
    """Compute pairwise overlap matrix for a batch of SDRs.

    Args:
        sdrs: (N, D) batch of binary SDR vectors.
        k: number of active bits per SDR.

    Returns:
        (N, N) symmetric overlap matrix.
    """
    # (N, D) @ (D, N) → (N, N) counts of shared active bits
    overlap_counts = sdrs @ sdrs.T
    return overlap_counts / k
