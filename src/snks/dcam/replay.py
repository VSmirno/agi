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
        stdp_updates = 0
        for ep in episodes:
            node_ids = [s for s in ep.pre_sks
                        if s < self.daf_engine.num_nodes]
            if not node_ids:
                continue
            self.daf_engine.inject_external_currents(node_ids, value=1.0)
            result = self.daf_engine.step(n_steps=self.n_steps)
            if result.fired_history is not None:
                self.stdp.apply(self.daf_engine.graph, result.fired_history)
                stdp_updates += 1
        return ReplayReport(n_replayed=len(episodes), stdp_updates=stdp_updates)
