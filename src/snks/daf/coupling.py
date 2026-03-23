"""Coupling computation via scatter_add on sparse COO graph."""

import torch

from snks.daf.graph import SparseDafGraph


def compute_kuramoto_coupling(
    states: torch.Tensor,
    graph: SparseDafGraph,
    coupling_strength: float,
) -> torch.Tensor:
    """Kuramoto coupling: K * Σ_j w_ij * sin(θ_j - θ_i + phase_shift_ij).

    Sign is determined by edge type: excitatory (+1) or inhibitory (-1).

    Args:
        states: (N, 8) — states[:, 0] = θ (phase)
        graph: SparseDafGraph
        coupling_strength: global K

    Returns:
        coupling: (N,) float32 — net coupling input per node
    """
    src_idx = graph.edge_index[0]  # (E,)
    dst_idx = graph.edge_index[1]  # (E,)

    theta = states[:, 0]  # (N,)
    strength = graph.edge_attr[:, 0]  # (E,)
    phase_shift = graph.edge_attr[:, 1]  # (E,)
    # Excitatory=0 → +1, Inhibitory=1 → -1
    sign = 1.0 - 2.0 * graph.edge_attr[:, 3]  # (E,)

    delta = theta[src_idx] - theta[dst_idx] + phase_shift  # (E,)
    contributions = sign * strength * torch.sin(delta)  # (E,)

    coupling = torch.zeros(graph.num_nodes, device=states.device, dtype=states.dtype)
    coupling.scatter_add_(0, dst_idx, contributions)
    coupling.mul_(coupling_strength)

    return coupling


def compute_fhn_coupling(
    states: torch.Tensor,
    graph: SparseDafGraph,
    coupling_strength: float,
) -> torch.Tensor:
    """FHN coupling: K * Σ_j w_ij * (v_j - v_i).

    Args:
        states: (N, 8) — states[:, 0] = v (membrane potential)
        graph: SparseDafGraph
        coupling_strength: global K

    Returns:
        coupling: (N,) float32
    """
    src_idx = graph.edge_index[0]
    dst_idx = graph.edge_index[1]

    v = states[:, 0]  # (N,)
    strength = graph.edge_attr[:, 0]  # (E,)
    sign = 1.0 - 2.0 * graph.edge_attr[:, 3]  # (E,)

    delta = v[src_idx] - v[dst_idx]  # (E,)
    contributions = sign * strength * delta  # (E,)

    coupling = torch.zeros(graph.num_nodes, device=states.device, dtype=states.dtype)
    coupling.scatter_add_(0, dst_idx, contributions)
    coupling.mul_(coupling_strength)

    return coupling
