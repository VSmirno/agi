"""AgentCommunicator: send, receive, and integrate concept messages (Stage 33).

Each agent has a communicator that manages:
- Outbox: messages to send
- Inbox: received messages
- Integration: merging received knowledge into the agent's causal model / skill library
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import torch
from torch import Tensor

from snks.agent.causal_model import CausalLink, CausalWorldModel
from snks.dcam.hac import HACEngine
from snks.language.concept_message import ConceptMessage, MessageType
from snks.language.skill import Skill
from snks.language.skill_library import SkillLibrary


@dataclass
class CommunicationStats:
    """Track communication metrics."""
    messages_sent: int = 0
    messages_received: int = 0
    links_integrated: int = 0
    skills_integrated: int = 0
    warnings_received: int = 0
    requests_received: int = 0


class AgentCommunicator:
    """Manages concept-level communication for one agent.

    Responsible for:
    1. Creating and sending ConceptMessages (causal links, skills, warnings)
    2. Receiving and filtering incoming messages
    3. Integrating received knowledge into CausalWorldModel / SkillLibrary
    """

    def __init__(
        self,
        agent_id: str,
        causal_model: CausalWorldModel,
        skill_library: SkillLibrary,
        hac_engine: HACEngine | None = None,
    ) -> None:
        self._agent_id = agent_id
        self._causal_model = causal_model
        self._skill_library = skill_library
        self._hac = hac_engine or HACEngine()
        self._inbox: deque[ConceptMessage] = deque()
        self._outbox: deque[ConceptMessage] = deque()
        self._stats = CommunicationStats()
        self._warning_sks: set[frozenset[int]] = set()  # dangerous contexts
        self._tick: int = 0

    @property
    def agent_id(self) -> str:
        return self._agent_id

    @property
    def stats(self) -> CommunicationStats:
        return self._stats

    @property
    def inbox_size(self) -> int:
        return len(self._inbox)

    @property
    def warning_contexts(self) -> set[frozenset[int]]:
        """SKS contexts flagged as dangerous by other agents."""
        return self._warning_sks

    def tick(self) -> None:
        """Advance logical clock."""
        self._tick += 1

    # ── Sending ──────────────────────────────────────────────────

    def share_causal_links(
        self,
        receiver_id: str | None = None,
        min_confidence: float = 0.5,
    ) -> ConceptMessage | None:
        """Create a message sharing high-confidence causal links."""
        links = self._causal_model.get_causal_links(min_confidence)
        if not links:
            return None

        # Gather context SKS from all links.
        all_ctx: set[int] = set()
        for link in links:
            all_ctx.update(link.context_sks)

        msg = ConceptMessage(
            sender_id=self._agent_id,
            receiver_id=receiver_id,
            content_type=MessageType.CAUSAL_LINKS,
            sks_context=frozenset(all_ctx),
            causal_links=list(links),
            timestamp=self._tick,
        )
        self._outbox.append(msg)
        self._stats.messages_sent += 1
        return msg

    def share_skill(
        self,
        skill: Skill,
        receiver_id: str | None = None,
    ) -> ConceptMessage:
        """Create a message sharing a learned skill."""
        msg = ConceptMessage(
            sender_id=self._agent_id,
            receiver_id=receiver_id,
            content_type=MessageType.SKILL,
            sks_context=skill.preconditions | skill.effects,
            skill=skill,
            timestamp=self._tick,
        )
        self._outbox.append(msg)
        self._stats.messages_sent += 1
        return msg

    def send_warning(
        self,
        danger_context: frozenset[int],
        hac_embedding: Tensor | None = None,
        receiver_id: str | None = None,
    ) -> ConceptMessage:
        """Warn other agents about a dangerous context."""
        msg = ConceptMessage(
            sender_id=self._agent_id,
            receiver_id=receiver_id,
            content_type=MessageType.WARNING,
            sks_context=danger_context,
            hac_embedding=hac_embedding,
            urgency=1.0,
            timestamp=self._tick,
        )
        self._outbox.append(msg)
        self._stats.messages_sent += 1
        return msg

    def request_knowledge(
        self,
        about_sks: frozenset[int],
        receiver_id: str | None = None,
    ) -> ConceptMessage:
        """Request knowledge about specific concepts from other agents."""
        msg = ConceptMessage(
            sender_id=self._agent_id,
            receiver_id=receiver_id,
            content_type=MessageType.REQUEST,
            sks_context=about_sks,
            timestamp=self._tick,
        )
        self._outbox.append(msg)
        self._stats.messages_sent += 1
        return msg

    def flush_outbox(self) -> list[ConceptMessage]:
        """Drain outbox and return all pending messages."""
        msgs = list(self._outbox)
        self._outbox.clear()
        return msgs

    # ── Receiving ────────────────────────────────────────────────

    def receive(self, msg: ConceptMessage) -> None:
        """Accept an incoming message into inbox."""
        if msg.receiver_id is not None and msg.receiver_id != self._agent_id:
            return  # not for us
        self._inbox.append(msg)
        self._stats.messages_received += 1

    def process_inbox(self) -> int:
        """Process all messages in inbox, integrating knowledge.

        Returns total number of knowledge items integrated.
        """
        total = 0
        while self._inbox:
            msg = self._inbox.popleft()
            total += self._integrate(msg)
        return total

    def _integrate(self, msg: ConceptMessage) -> int:
        """Integrate one message into agent's knowledge base."""
        count = 0

        if msg.content_type == MessageType.CAUSAL_LINKS:
            count = self._integrate_causal_links(msg.causal_links)

        elif msg.content_type == MessageType.SKILL:
            if msg.skill is not None:
                count = self._integrate_skill(msg.skill)

        elif msg.content_type == MessageType.WARNING:
            self._warning_sks.add(msg.sks_context)
            self._stats.warnings_received += 1
            count = 1

        elif msg.content_type == MessageType.REQUEST:
            self._stats.requests_received += 1
            # Respond with relevant links if we have them.
            count = self._handle_request(msg)

        return count

    def _integrate_causal_links(self, links: list[CausalLink]) -> int:
        """Merge received causal links into our model.

        Uses confidence-weighted integration: only accept links that are
        stronger than what we already know, or that are novel.
        """
        integrated = 0
        existing_links = {
            (l.action, l.context_sks, l.effect_sks): l
            for l in self._causal_model.get_causal_links(0.0)
        }

        for link in links:
            key = (link.action, link.context_sks, link.effect_sks)
            existing = existing_links.get(key)

            if existing is None or link.strength > existing.strength:
                # Inject via observe_transition for proper integration.
                for _ in range(link.count):
                    self._causal_model.observe_transition(
                        pre_sks=set(link.context_sks),
                        action=link.action,
                        post_sks=set(link.context_sks | link.effect_sks),
                    )
                integrated += 1

        self._stats.links_integrated += integrated
        return integrated

    def _integrate_skill(self, skill: Skill) -> int:
        """Register a received skill if we don't have it."""
        if self._skill_library.get(skill.name) is not None:
            return 0
        self._skill_library.register(skill)
        self._stats.skills_integrated += 1
        return 1

    def _handle_request(self, msg: ConceptMessage) -> int:
        """Respond to knowledge request by preparing relevant links."""
        requested_sks = msg.sks_context
        if not requested_sks:
            return 0

        # Find causal links that involve the requested SKS.
        relevant_links = []
        for link in self._causal_model.get_causal_links(0.3):
            if link.context_sks & requested_sks or link.effect_sks & requested_sks:
                relevant_links.append(link)

        if relevant_links:
            response = ConceptMessage(
                sender_id=self._agent_id,
                receiver_id=msg.sender_id,
                content_type=MessageType.CAUSAL_LINKS,
                sks_context=requested_sks,
                causal_links=relevant_links,
                timestamp=self._tick,
            )
            self._outbox.append(response)
            self._stats.messages_sent += 1
            return len(relevant_links)

        return 0
