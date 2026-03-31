"""MultiAgentEnv: coordinated multi-agent environment (Stage 33).

Simulates N agents in a shared environment with concept-level communication.
Each agent has its own CausalWorldModel, SkillLibrary, and AgentCommunicator.
Agents cooperate by exchanging concept messages — NOT natural language.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from snks.agent.causal_model import CausalLink, CausalWorldModel
from snks.daf.types import CausalAgentConfig
from snks.language.agent_communicator import AgentCommunicator, CommunicationStats
from snks.language.concept_message import ConceptMessage, MessageType
from snks.language.skill import Skill
from snks.language.skill_library import SkillLibrary


@dataclass
class AgentState:
    """State of one agent in the multi-agent environment."""
    agent_id: str
    causal_model: CausalWorldModel
    skill_library: SkillLibrary
    communicator: AgentCommunicator
    position: tuple[int, int] = (0, 0)
    role: str = ""                      # e.g. "explorer", "solver"
    episode_success: bool = False
    steps_taken: int = 0


@dataclass
class MultiAgentResult:
    """Result of a multi-agent cooperative episode."""
    success: bool = False
    total_steps: int = 0
    messages_exchanged: int = 0
    links_transferred: int = 0
    skills_transferred: int = 0
    agent_results: dict[str, bool] = field(default_factory=dict)
    cooperation_score: float = 0.0      # 0.0 .. 1.0


class MultiAgentEnv:
    """Coordinated multi-agent environment with concept-level communication.

    Manages N agents, each with their own knowledge base, that communicate
    by exchanging ConceptMessages through a shared channel.

    The environment supports two cooperation modes:
    1. Knowledge sharing: agents share causal links and skills
    2. Role-based cooperation: agents have different roles (explorer/solver)
    """

    def __init__(
        self,
        n_agents: int = 2,
        agent_ids: list[str] | None = None,
    ) -> None:
        if agent_ids is None:
            agent_ids = [f"agent_{i}" for i in range(n_agents)]
        if len(agent_ids) != n_agents:
            raise ValueError(f"Expected {n_agents} agent IDs, got {len(agent_ids)}")

        self._n_agents = n_agents
        self._agents: dict[str, AgentState] = {}
        self._message_channel: list[ConceptMessage] = []
        self._tick: int = 0

        for aid in agent_ids:
            config = CausalAgentConfig(causal_min_observations=1)
            model = CausalWorldModel(config)
            library = SkillLibrary()
            comm = AgentCommunicator(aid, model, library)
            self._agents[aid] = AgentState(
                agent_id=aid,
                causal_model=model,
                skill_library=library,
                communicator=comm,
            )

    @property
    def n_agents(self) -> int:
        return self._n_agents

    @property
    def agent_ids(self) -> list[str]:
        return list(self._agents.keys())

    def get_agent(self, agent_id: str) -> AgentState:
        return self._agents[agent_id]

    def set_role(self, agent_id: str, role: str) -> None:
        """Assign a role to an agent."""
        self._agents[agent_id].role = role

    def inject_knowledge(
        self,
        agent_id: str,
        links: list[CausalLink],
    ) -> int:
        """Inject causal knowledge directly into an agent's model."""
        agent = self._agents[agent_id]
        count = 0
        for link in links:
            for _ in range(max(link.count, 1)):
                agent.causal_model.observe_transition(
                    pre_sks=set(link.context_sks),
                    action=link.action,
                    post_sks=set(link.context_sks | link.effect_sks),
                )
            count += 1
        return count

    def inject_skill(self, agent_id: str, skill: Skill) -> None:
        """Inject a skill directly into an agent's library."""
        self._agents[agent_id].skill_library.register(skill)

    # ── Communication ────────────────────────────────────────────

    def broadcast(self, sender_id: str, msg: ConceptMessage) -> int:
        """Deliver a message to all agents except sender.

        Returns number of agents that received the message.
        """
        delivered = 0
        for aid, agent in self._agents.items():
            if aid == sender_id:
                continue
            if msg.receiver_id is not None and msg.receiver_id != aid:
                continue
            agent.communicator.receive(msg)
            delivered += 1
        self._message_channel.append(msg)
        return delivered

    def exchange_round(self) -> int:
        """Execute one round of communication.

        Each agent flushes their outbox and messages are delivered.
        Then each agent processes their inbox.

        Returns total messages exchanged.
        """
        self._tick += 1
        total = 0

        # 1. Collect all outgoing messages.
        all_msgs: list[tuple[str, ConceptMessage]] = []
        for aid, agent in self._agents.items():
            agent.communicator.tick()
            msgs = agent.communicator.flush_outbox()
            for msg in msgs:
                all_msgs.append((aid, msg))

        # 2. Deliver messages.
        for sender_id, msg in all_msgs:
            self.broadcast(sender_id, msg)
            total += 1

        # 3. Each agent processes inbox.
        for agent in self._agents.values():
            agent.communicator.process_inbox()

        return total

    def run_cooperative_episode(
        self,
        task_links: dict[str, list[CausalLink]],
        max_rounds: int = 10,
    ) -> MultiAgentResult:
        """Run a cooperative episode where agents share knowledge.

        Args:
            task_links: per-agent initial causal knowledge (simulates
                each agent exploring different parts of the environment).
            max_rounds: maximum communication rounds.

        Returns:
            MultiAgentResult with cooperation metrics.
        """
        result = MultiAgentResult()

        # 1. Inject initial knowledge (each agent knows different things).
        for aid, links in task_links.items():
            if aid in self._agents:
                self.inject_knowledge(aid, links)

        total_messages = 0

        # 2. Communication rounds: agents share what they know.
        for round_idx in range(max_rounds):
            # Each agent shares their causal links.
            for aid, agent in self._agents.items():
                agent.communicator.share_causal_links()

            # Exchange and integrate.
            n_msgs = self.exchange_round()
            total_messages += n_msgs

            if n_msgs == 0:
                break  # convergence

        # 3. Collect results.
        result.messages_exchanged = total_messages
        result.total_steps = max_rounds

        total_links = 0
        total_skills = 0
        for agent in self._agents.values():
            stats = agent.communicator.stats
            total_links += stats.links_integrated
            total_skills += stats.skills_integrated

        result.links_transferred = total_links
        result.skills_transferred = total_skills

        # Cooperation score: fraction of knowledge shared successfully.
        all_links_count = sum(
            a.causal_model.n_links for a in self._agents.values()
        )
        if all_links_count > 0:
            # Max possible = each agent has all links.
            max_possible = all_links_count * self._n_agents
            actual = sum(
                a.causal_model.n_links for a in self._agents.values()
            )
            result.cooperation_score = actual / max_possible
        else:
            result.cooperation_score = 0.0

        result.success = total_links > 0
        return result

    def get_aggregate_stats(self) -> dict[str, CommunicationStats]:
        """Get communication stats for all agents."""
        return {
            aid: agent.communicator.stats
            for aid, agent in self._agents.items()
        }

    def knowledge_overlap(self) -> float:
        """Compute pairwise knowledge overlap between agents.

        Returns average Jaccard similarity of causal link sets.
        """
        if self._n_agents < 2:
            return 1.0

        agents = list(self._agents.values())
        link_sets: list[set] = []
        for agent in agents:
            links = agent.causal_model.get_causal_links(0.0)
            link_set = {
                (l.action, l.context_sks, l.effect_sks)
                for l in links
            }
            link_sets.append(link_set)

        # Average pairwise Jaccard.
        total_jaccard = 0.0
        n_pairs = 0
        for i in range(len(link_sets)):
            for j in range(i + 1, len(link_sets)):
                intersection = len(link_sets[i] & link_sets[j])
                union = len(link_sets[i] | link_sets[j])
                if union > 0:
                    total_jaccard += intersection / union
                n_pairs += 1

        return total_jaccard / max(n_pairs, 1)
