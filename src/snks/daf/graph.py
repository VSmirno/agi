"""Sparse graph representation for DAF engine using COO format."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from snks.daf.types import ZoneConfig


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

        graph = SparseDafGraph(edge_index, edge_attr, num_nodes, device)
        graph.sort_by_dst()
        return graph

    @staticmethod
    def random_sparse_zonal(
        num_nodes: int,
        zones: dict[str, ZoneConfig],
        intra_degree: int,
        inter_degree: int,
        device: torch.device,
        seed: int | None = None,
    ) -> SparseDafGraph:
        """Create a random directed graph with zone-aware connectivity.

        Intra-zone edges are dense (intra_degree per node), inter-zone edges
        are sparse (inter_degree per node) to allow STDP-driven cross-modal
        grounding.

        Args:
            num_nodes: total number of oscillator nodes.
            zones: mapping of zone names to ZoneConfig (start, size).
            intra_degree: average out-degree within a zone.
            inter_degree: average out-degree to other zones (per zone pair).
            device: target device.
            seed: optional RNG seed.
        """
        if seed is not None:
            gen = torch.Generator(device="cpu").manual_seed(seed)
        else:
            gen = None

        all_src: list[torch.Tensor] = []
        all_dst: list[torch.Tensor] = []

        zone_list = list(zones.values())

        # Intra-zone edges
        for z in zone_list:
            n_edges = z.size * intra_degree
            src = torch.randint(0, z.size, (n_edges,), dtype=torch.int64, generator=gen) + z.start
            dst = torch.randint(0, z.size, (n_edges,), dtype=torch.int64, generator=gen) + z.start
            all_src.append(src)
            all_dst.append(dst)

        # Inter-zone edges (both directions for each pair)
        for i, za in enumerate(zone_list):
            for j, zb in enumerate(zone_list):
                if i >= j:
                    continue
                n_edges = min(za.size, zb.size) * inter_degree
                # A -> B
                src_ab = torch.randint(0, za.size, (n_edges,), dtype=torch.int64, generator=gen) + za.start
                dst_ab = torch.randint(0, zb.size, (n_edges,), dtype=torch.int64, generator=gen) + zb.start
                all_src.append(src_ab)
                all_dst.append(dst_ab)
                # B -> A
                src_ba = torch.randint(0, zb.size, (n_edges,), dtype=torch.int64, generator=gen) + zb.start
                dst_ba = torch.randint(0, za.size, (n_edges,), dtype=torch.int64, generator=gen) + za.start
                all_src.append(src_ba)
                all_dst.append(dst_ba)

        src = torch.cat(all_src)
        dst = torch.cat(all_dst)

        # Remove self-loops
        mask = src != dst
        src = src[mask]
        dst = dst[mask]

        edge_index = torch.stack([src, dst], dim=0).to(device)
        num_edges = edge_index.shape[1]

        # Edge attributes: [strength, phase_shift, delay, type]
        strength = torch.rand(num_edges, generator=gen) * 0.5
        phase_shift = torch.zeros(num_edges)
        delay = torch.zeros(num_edges)
        edge_type = (torch.rand(num_edges, generator=gen) < 0.2).float()

        edge_attr = torch.stack([strength, phase_shift, delay, edge_type], dim=1).to(device)

        graph = SparseDafGraph(edge_index, edge_attr, num_nodes, device)
        graph.sort_by_dst()
        return graph

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

    def sort_by_dst(self) -> None:
        """Sort edges by destination index for better scatter_add cache locality."""
        order = self.edge_index[1].argsort()
        self.edge_index = self.edge_index[:, order].contiguous()
        self.edge_attr = self.edge_attr[order].contiguous()

    def to(self, device: torch.device) -> SparseDafGraph:
        """Move graph to another device."""
        return SparseDafGraph(
            self.edge_index.to(device),
            self.edge_attr.to(device),
            self.num_nodes,
            device,
        )
