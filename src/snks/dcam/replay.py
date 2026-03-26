"""ReplayEngine: replays top-k transitions through DAF + STDP (Stage 16).

Injects pre_sks node activations into DAF as external currents,
runs integration, and applies STDP on the resulting spike history.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ReplayReport:
    """Summary of one replay pass."""
    n_replayed: int
    stdp_updates: int


class ReplayEngine:
    """Replays important past transitions to strengthen DAF weights via STDP."""

    def __init__(
        self,
        daf_engine,     # DafEngine
        stdp,           # STDP
        top_k: int = 10,
        n_steps: int = 50,
    ) -> None:
        self.daf_engine = daf_engine
        self.stdp = stdp
        self.top_k = top_k
        self.n_steps = n_steps

    def replay(self, agent_buffer) -> ReplayReport:
        """Replay top_k transitions by importance through DAF + STDP.

        Args:
            agent_buffer: AgentTransitionBuffer — source of transitions.

        Returns:
            ReplayReport with counts of replayed episodes and STDP updates.
        """
        episodes = agent_buffer.get_top_k(k=self.top_k, by="importance")
        print(f"[REPLAY_DBG] buf_len={len(agent_buffer)} episodes_top_k={len(episodes)}")
        if episodes:
            sample = episodes[0]
            print(f"[REPLAY_DBG] sample ep: pre_nodes={len(sample.pre_nodes)} pre_sks={len(sample.pre_sks)} importance={sample.importance:.3f}")
        stdp_updates = 0
        for i, ep in enumerate(episodes):
            node_ids = [n for n in ep.pre_nodes
                        if n < self.daf_engine.num_nodes]
            print(f"[REPLAY_DBG] ep[{i}]: pre_nodes={len(ep.pre_nodes)} node_ids={len(node_ids)} num_nodes={self.daf_engine.num_nodes}")
            if not node_ids:
                continue
            self.daf_engine.inject_external_currents(node_ids, value=1.0)
            result = self.daf_engine.step(n_steps=self.n_steps)
            fh = result.fired_history
            print(f"[REPLAY_DBG] ep[{i}]: fired_history={fh is not None} shape={fh.shape if fh is not None else 'None'} any_spikes={fh.any().item() if fh is not None else 'N/A'}")
            if fh is not None:
                self.stdp.apply(self.daf_engine.graph, fh)
                stdp_updates += 1
        return ReplayReport(n_replayed=len(episodes), stdp_updates=stdp_updates)
