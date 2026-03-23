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


def compute_fhn_coupling_inplace(
    states: torch.Tensor,
    graph: SparseDafGraph,
    coupling_strength: float,
    out: torch.Tensor,
    contrib_buf: torch.Tensor,
    src_v_buf: torch.Tensor,
    dst_v_buf: torch.Tensor,
    edge_sign: torch.Tensor | None = None,
) -> None:
    """Zero-allocation FHN coupling using pre-allocated buffers.

    Args:
        states: (N, 8) — states[:, 0] = v
        graph: SparseDafGraph (sorted by dst for cache locality)
        coupling_strength: global K
        out: (N,) pre-allocated output buffer — receives coupling result
        contrib_buf: (E,) pre-allocated — edge contributions
        src_v_buf: (E,) pre-allocated — source voltages
        dst_v_buf: (E,) pre-allocated — destination voltages
        edge_sign: (E,) pre-computed sign (1 - 2*type), avoids per-step temp allocation
    """
    v = states[:, 0]
    torch.index_select(v, 0, graph.edge_index[0], out=src_v_buf)
    torch.index_select(v, 0, graph.edge_index[1], out=dst_v_buf)

    # contrib = sign * strength * (v_src - v_dst)
    torch.sub(src_v_buf, dst_v_buf, out=contrib_buf)
    contrib_buf.mul_(graph.edge_attr[:, 0])  # strength
    if edge_sign is not None:
        contrib_buf.mul_(edge_sign)
    else:
        contrib_buf.mul_(1.0 - 2.0 * graph.edge_attr[:, 3])

    out.zero_()
    out.scatter_add_(0, graph.edge_index[1], contrib_buf)
    out.mul_(coupling_strength)


def build_coupling_csr(
    graph: SparseDafGraph,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build CSR adjacency matrix and weighted in-degree for FHN coupling.

    The coupling formula: coupling[i] = K * Σ_{j→i} sign_j * w_j * (v[j] - v[i])
    = K * (A @ v - v * degree)

    Where A[i,j] = sign * strength for edge j→i, degree[i] = sum_j A[i,j].

    Args:
        graph: SparseDafGraph (edges sorted by dst recommended)

    Returns:
        (A_csr, degree) where A_csr is (N,N) sparse CSR, degree is (N,) dense.
    """
    device = graph.device
    N = graph.num_nodes
    E = graph.num_edges

    src_idx = graph.edge_index[0]
    dst_idx = graph.edge_index[1]
    signed_strength = (1.0 - 2.0 * graph.edge_attr[:, 3]) * graph.edge_attr[:, 0]

    # Sort by dst (row) if not already sorted
    order = dst_idx.argsort()
    row_sorted = dst_idx[order]
    col_sorted = src_idx[order]
    val_sorted = signed_strength[order]

    # Build crow_indices
    crow = torch.zeros(N + 1, dtype=torch.int64, device=device)
    ones = torch.ones(E, dtype=torch.int64, device=device)
    crow[1:].scatter_add_(0, row_sorted, ones)
    crow = crow.cumsum(0)

    A_csr = torch.sparse_csr_tensor(crow, col_sorted, val_sorted, size=(N, N))

    # Weighted in-degree: degree[i] = Σ_{j→i} sign_j * w_j
    degree = torch.zeros(N, device=device)
    degree.scatter_add_(0, dst_idx, signed_strength)

    return A_csr, degree


def update_coupling_csr_values(
    A_csr: torch.Tensor,
    degree: torch.Tensor,
    graph: SparseDafGraph,
) -> None:
    """Update CSR values and degree in-place after STDP weight changes.

    Args:
        A_csr: sparse CSR tensor — values updated in-place
        degree: (N,) tensor — updated in-place
        graph: SparseDafGraph with updated edge_attr[:, 0]
    """
    src_idx = graph.edge_index[0]
    dst_idx = graph.edge_index[1]
    signed_strength = (1.0 - 2.0 * graph.edge_attr[:, 3]) * graph.edge_attr[:, 0]

    # Re-sort values in the same order as CSR construction
    order = dst_idx.argsort()
    A_csr.values().copy_(signed_strength[order])

    # Update degree
    degree.zero_()
    degree.scatter_add_(0, dst_idx, signed_strength)


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
