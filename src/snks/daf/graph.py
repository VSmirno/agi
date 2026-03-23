"""Sparse graph representation for DAF engine using COO format."""

from __future__ import annotations

import torch


class SparseDafGraph:
    """Sparse directed graph in COO format for oscillator coupling.

    Attributes:
        edge_index: (2, E) int64 — row 0 = source, row 1 = destination
        edge_attr: (E, 4) float32 — [strength, phase_shift, delay, type]
        num_nodes: number of nodes in the graph
        device: torch.device
    """

    __slots__ = ("edge_index", "edge_attr", "num_nodes", "device")

    def __init__(
        self,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        num_nodes: int,
        device: torch.device,
    ) -> None:
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.num_nodes = num_nodes
        self.device = device

    @staticmethod
    def random_sparse(
        num_nodes: int,
        avg_degree: int,
        device: torch.device,
        seed: int | None = None,
    ) -> SparseDafGraph:
        """Create a random Erdős–Rényi directed graph.

        Args:
            num_nodes: N — number of oscillator nodes.
            avg_degree: average out-degree per node.
            device: target device.
            seed: optional RNG seed for reproducibility.
        """
        if seed is not None:
            gen = torch.Generator(device="cpu").manual_seed(seed)
        else:
            gen = None

        total_edges = num_nodes * avg_degree

        # Generate on CPU (Cauchy / some distributions may not work on all backends)
        src = torch.randint(0, num_nodes, (total_edges,), dtype=torch.int64, generator=gen)
        dst = torch.randint(0, num_nodes, (total_edges,), dtype=torch.int64, generator=gen)

        # Remove self-loops
        mask = src != dst
        src = src[mask]
        dst = dst[mask]

        edge_index = torch.stack([src, dst], dim=0).to(device)
        num_edges = edge_index.shape[1]

        # Edge attributes: [strength, phase_shift, delay, type]
        strength = torch.rand(num_edges, generator=gen) * 0.5  # moderate initial weights
        phase_shift = torch.zeros(num_edges)
        delay = torch.zeros(num_edges)
        # 80% excitatory (0.0), 20% inhibitory (1.0)
        edge_type = (torch.rand(num_edges, generator=gen) < 0.2).float()

        edge_attr = torch.stack([strength, phase_shift, delay, edge_type], dim=1).to(device)

        return SparseDafGraph(edge_index, edge_attr, num_nodes, device)

    @property
    def num_edges(self) -> int:
        return self.edge_index.shape[1]

    def get_strength(self) -> torch.Tensor:
        """Return (E,) view of edge strengths."""
        return self.edge_attr[:, 0]

    def set_strength(self, values: torch.Tensor) -> None:
        """Set edge strengths in-place. values: (E,) float32."""
        self.edge_attr[:, 0].copy_(values)

    def add_edges(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        attr: torch.Tensor,
    ) -> None:
        """Append edges. src/dst: (K,) int64, attr: (K, 4) float32."""
        new_index = torch.stack([src, dst], dim=0).to(self.device)
        self.edge_index = torch.cat([self.edge_index, new_index], dim=1)
        self.edge_attr = torch.cat([self.edge_attr, attr.to(self.device)], dim=0)

    def remove_edges(self, mask: torch.Tensor) -> None:
        """Remove edges where mask is True. mask: (E,) bool."""
        keep = ~mask
        self.edge_index = self.edge_index[:, keep].contiguous()
        self.edge_attr = self.edge_attr[keep].contiguous()

    def to(self, device: torch.device) -> SparseDafGraph:
        """Move graph to another device."""
        return SparseDafGraph(
            self.edge_index.to(device),
            self.edge_attr.to(device),
            self.num_nodes,
            device,
        )
