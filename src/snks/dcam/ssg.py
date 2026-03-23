"""SSG (Structured Sparse Graph) — multi-layer graph for world model structure.

Four layers: structural, causal, temporal, modulatory.
Pure Python dicts — no PyTorch tensors needed.
"""

from __future__ import annotations

from collections import defaultdict

VALID_LAYERS = ("structural", "causal", "temporal", "modulatory")


class StructuredSparseGraph:
    """Multi-layer sparse directed graph."""

    def __init__(self) -> None:
        # layer_name → src → dst → weight
        self._layers: dict[str, dict[int, dict[int, float]]] = {
            layer: defaultdict(dict) for layer in VALID_LAYERS
        }

    def _check_layer(self, layer: str) -> None:
        if layer not in VALID_LAYERS:
            raise ValueError(
                f"Unknown layer '{layer}'. Valid: {VALID_LAYERS}"
            )

    def add_edge(self, src: int, dst: int, layer: str, weight: float = 1.0) -> None:
        """Add or overwrite an edge in the specified layer."""
        self._check_layer(layer)
        self._layers[layer][src][dst] = weight

    def get_neighbors(self, node: int, layer: str) -> list[tuple[int, float]]:
        """Return [(dst, weight)] for outgoing edges from node in layer."""
        self._check_layer(layer)
        return list(self._layers[layer].get(node, {}).items())

    def update_edge(self, src: int, dst: int, layer: str, delta: float) -> None:
        """Add delta to existing edge weight, or create with weight=delta."""
        self._check_layer(layer)
        current = self._layers[layer][src].get(dst, 0.0)
        self._layers[layer][src][dst] = current + delta

    def prune(self, threshold: float, layer: str | None = None) -> int:
        """Remove edges with |weight| < threshold. Returns count of removed edges."""
        layers = [layer] if layer else list(VALID_LAYERS)
        removed = 0
        for lname in layers:
            self._check_layer(lname)
            to_remove: list[tuple[int, int]] = []
            for src, dsts in self._layers[lname].items():
                for dst, w in dsts.items():
                    if abs(w) < threshold:
                        to_remove.append((src, dst))
            for src, dst in to_remove:
                del self._layers[lname][src][dst]
                removed += 1
        return removed

    def get_all_edges(self, layer: str) -> list[tuple[int, int, float]]:
        """Return all edges [(src, dst, weight)] in a layer."""
        self._check_layer(layer)
        edges = []
        for src, dsts in self._layers[layer].items():
            for dst, w in dsts.items():
                edges.append((src, dst, w))
        return edges
