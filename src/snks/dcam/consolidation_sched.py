"""ConsolidationScheduler: episodic → SSG consolidation (Stage 16).

Converts AgentTransitionBuffer entries into causal SSG edges, enabling
cross-session persistence of learned world model structure.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


class SKSIDEmbedder:
    """Deterministic, session-stable HAC embedding for integer SKS IDs.

    Each ID → fixed unit vector via seeded torch.Generator.
    Same ID always produces the same vector, regardless of session.
    No persistence needed.
    """

    def __init__(self, hac_dim: int, device: torch.device) -> None:
        self._dim = hac_dim
        self._device = device
        self._cache: dict[int, Tensor] = {}

    def embed_id(self, sks_id: int) -> Tensor:
        """Return deterministic unit vector for integer SKS ID."""
        if sks_id not in self._cache:
            g = torch.Generator()
            g.manual_seed(int(sks_id) % (2 ** 32))
            vec = torch.randn(self._dim, generator=g, device=self._device)
            self._cache[sks_id] = vec / vec.norm().clamp(min=1e-8)
        return self._cache[sks_id]

    def encode_sks_set(self, sks_ids: set[int], hac) -> Tensor | None:
        """Bundle embeddings of all IDs in the set into one HAC vector.

        Returns None if sks_ids is empty.
        """
        if not sks_ids:
            return None
        vecs = [self.embed_id(s) for s in sks_ids]
        return hac.bundle(vecs) if len(vecs) > 1 else vecs[0]


@dataclass
class ConsolidationSummary:
    """Summary of one consolidation run."""
    n_episodes_processed: int
    n_edges_added: int
    total_causal_edges: int
    total_nodes: int


class ConsolidationScheduler:
    """Runs periodic consolidation: AgentTransitionBuffer → SSG causal layer.

    Every `every_n` episodes, takes the top-k transitions by importance,
    encodes pre/post SKS sets as HAC vectors, maps them to SSG nodes, and
    updates edge weights in the "causal" layer.
    """

    def __init__(
        self,
        agent_buffer,           # AgentTransitionBuffer
        dcam,                   # DcamWorldModel — for SSG + HACEngine
        every_n: int = 10,
        top_k: int = 50,
        node_threshold: float = 0.7,
        save_path: str | None = None,
    ) -> None:
        self.agent_buffer = agent_buffer
        self.dcam = dcam
        self.every_n = every_n
        self.top_k = top_k
        self.save_path = save_path

        self.embedder = SKSIDEmbedder(dcam.hac.dim, dcam.device)
        self._node_registry: dict[int, Tensor] = {}
        self._next_node_id: int = 0
        self._node_threshold: float = node_threshold
        self._edge_actions: dict[tuple[int, int], int] = {}

    def maybe_consolidate(self, episode: int) -> ConsolidationSummary | None:
        """Run consolidation if episode is a multiple of every_n (and > 0)."""
        if episode > 0 and episode % self.every_n == 0:
            return self._run()
        return None

    def query(
        self,
        context_sks: set[int],
        threshold: float = 0.3,
    ) -> tuple[int | None, float]:
        """Encode context_sks → HAC vec, find nearest node, return (action, weight).

        Returns (None, 0.0) if registry is empty or best similarity < threshold.
        """
        vec = self.embedder.encode_sks_set(context_sks, self.dcam.hac)
        if vec is None or not self._node_registry:
            return None, 0.0
        best_nid, best_sim = self._nearest_node(vec)
        if best_sim < threshold:
            return None, 0.0
        neighbors = self.dcam.graph.get_neighbors(best_nid, layer="causal")
        if not neighbors:
            return None, 0.0
        dst_id, weight = max(neighbors, key=lambda x: x[1])
        action = self._edge_actions.get((best_nid, dst_id))
        return action, float(weight)

    def save_state(self, path: str) -> None:
        """Save node registry and edge actions to disk."""
        torch.save(
            {
                "node_registry": self._node_registry,
                "edge_actions": self._edge_actions,
                "next_node_id": self._next_node_id,
            },
            path + "_sched.pt",
        )
        self.dcam.save(path)

    def load_state(self, path: str) -> None:
        """Restore node registry and edge actions from disk."""
        state = torch.load(path + "_sched.pt", map_location=self.dcam.device,
                           weights_only=False)
        self._node_registry = state["node_registry"]
        self._edge_actions = state["edge_actions"]
        self._next_node_id = state["next_node_id"]
        self.dcam.load(path)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(self) -> ConsolidationSummary:
        episodes = self.agent_buffer.get_top_k(k=self.top_k, by="importance")
        n_added = 0
        for ep in episodes:
            ctx_vec = self.embedder.encode_sks_set(ep.pre_sks, self.dcam.hac)
            next_vec = self.embedder.encode_sks_set(ep.post_sks, self.dcam.hac)
            if ctx_vec is None or next_vec is None:
                continue
            src_id = self._get_or_create_node(ctx_vec)
            dst_id = self._get_or_create_node(next_vec)
            self.dcam.graph.update_edge(src_id, dst_id, layer="causal",
                                        delta=ep.importance)
            self._edge_actions[(src_id, dst_id)] = ep.action
            n_added += 1
        if self.save_path:
            self.save_state(self.save_path)
        total = sum(
            len(dsts)
            for dsts in self.dcam.graph._layers["causal"].values()
        )
        return ConsolidationSummary(
            n_episodes_processed=len(episodes),
            n_edges_added=n_added,
            total_causal_edges=total,
            total_nodes=len(self._node_registry),
        )

    def _get_or_create_node(self, vec: Tensor) -> int:
        if self._node_registry:
            nid, sim = self._nearest_node(vec)
            if sim > self._node_threshold:
                return nid
        nid = self._next_node_id
        self._next_node_id += 1
        self._node_registry[nid] = vec.detach()
        return nid

    def _nearest_node(self, vec: Tensor) -> tuple[int, float]:
        best_nid, best_sim = -1, -1.0
        for nid, nvec in self._node_registry.items():
            sim = self.dcam.hac.similarity(vec, nvec)
            if sim > best_sim:
                best_sim, best_nid = sim, nid
        return best_nid, float(best_sim)
