"""Tests for Stage 33: Multi-Agent Communication."""

from __future__ import annotations

import pytest
import torch

from snks.agent.causal_model import CausalLink, CausalWorldModel
from snks.daf.types import CausalAgentConfig
from snks.dcam.hac import HACEngine
from snks.language.agent_communicator import AgentCommunicator
from snks.language.concept_message import ConceptMessage, MessageType
from snks.language.multi_agent_env import MultiAgentEnv
from snks.language.skill import Skill
from snks.language.skill_library import SkillLibrary


# ── Helpers ──────────────────────────────────────────────────────

def _make_causal_model(links: list[tuple[int, set[int], set[int]]]) -> CausalWorldModel:
    """Create a CausalWorldModel with pre-injected links.

    Each tuple: (action, context_sks, post_sks).
    """
    config = CausalAgentConfig(causal_min_observations=1)
    model = CausalWorldModel(config)
    for action, ctx, post in links:
        for _ in range(3):  # observe 3 times for confidence
            model.observe_transition(pre_sks=ctx, action=action, post_sks=post)
    return model


def _make_skill(name: str = "pickup_key") -> Skill:
    return Skill(
        name=name,
        preconditions=frozenset({50}),
        effects=frozenset({51}),
        terminal_action=3,
        target_word="key",
        success_count=5,
        attempt_count=5,
    )


def _make_communicator(
    agent_id: str = "agent_0",
    links: list[tuple[int, set[int], set[int]]] | None = None,
) -> AgentCommunicator:
    model = _make_causal_model(links or [])
    library = SkillLibrary()
    return AgentCommunicator(agent_id, model, library)


# ── ConceptMessage Tests ─────────────────────────────────────────

class TestConceptMessage:
    def test_create_causal_message(self):
        msg = ConceptMessage(
            sender_id="a",
            receiver_id="b",
            content_type=MessageType.CAUSAL_LINKS,
            sks_context=frozenset({50, 51}),
            causal_links=[],
            timestamp=1,
        )
        assert msg.sender_id == "a"
        assert msg.receiver_id == "b"
        assert not msg.is_broadcast

    def test_broadcast_message(self):
        msg = ConceptMessage(
            sender_id="a",
            receiver_id=None,
            content_type=MessageType.WARNING,
            sks_context=frozenset({52}),
            urgency=1.0,
        )
        assert msg.is_broadcast
        assert msg.urgency == 1.0

    def test_invalid_urgency(self):
        with pytest.raises(ValueError, match="urgency"):
            ConceptMessage(
                sender_id="a",
                receiver_id=None,
                content_type=MessageType.WARNING,
                urgency=1.5,
            )

    def test_content_summary(self):
        link = CausalLink(
            action=3,
            context_sks=frozenset({50}),
            effect_sks=frozenset({51}),
            strength=0.8,
            count=3,
        )
        msg = ConceptMessage(
            sender_id="a",
            receiver_id="b",
            content_type=MessageType.CAUSAL_LINKS,
            causal_links=[link],
        )
        summary = msg.content_summary()
        assert "causal_links" in summary
        assert "links=1" in summary

    def test_skill_message_summary(self):
        skill = _make_skill()
        msg = ConceptMessage(
            sender_id="a",
            receiver_id=None,
            content_type=MessageType.SKILL,
            skill=skill,
        )
        assert "pickup_key" in msg.content_summary()

    def test_message_types_enum(self):
        assert MessageType.CAUSAL_LINKS.value == "causal_links"
        assert MessageType.SKILL.value == "skill"
        assert MessageType.WARNING.value == "warning"
        assert MessageType.REQUEST.value == "request"


# ── AgentCommunicator Tests ──────────────────────────────────────

class TestAgentCommunicator:
    def test_share_causal_links(self):
        links = [(3, {50}, {50, 51})]
        comm = _make_communicator("explorer", links)
        msg = comm.share_causal_links()
        assert msg is not None
        assert msg.content_type == MessageType.CAUSAL_LINKS
        assert len(msg.causal_links) > 0
        assert msg.sender_id == "explorer"

    def test_share_causal_links_empty_model(self):
        comm = _make_communicator("empty")
        msg = comm.share_causal_links(min_confidence=0.5)
        assert msg is None

    def test_share_skill(self):
        comm = _make_communicator("teacher")
        skill = _make_skill()
        msg = comm.share_skill(skill, receiver_id="student")
        assert msg.content_type == MessageType.SKILL
        assert msg.skill == skill
        assert msg.receiver_id == "student"

    def test_send_warning(self):
        comm = _make_communicator("scout")
        ctx = frozenset({52, 57})
        msg = comm.send_warning(ctx)
        assert msg.content_type == MessageType.WARNING
        assert msg.urgency == 1.0
        assert msg.sks_context == ctx

    def test_request_knowledge(self):
        comm = _make_communicator("learner")
        msg = comm.request_knowledge(frozenset({50, 51}))
        assert msg.content_type == MessageType.REQUEST

    def test_receive_and_process(self):
        # Agent A has knowledge, Agent B receives it.
        links_a = [(3, {50}, {50, 51})]
        comm_a = _make_communicator("agent_a", links_a)
        comm_b = _make_communicator("agent_b")

        # A shares knowledge.
        msg = comm_a.share_causal_links(receiver_id="agent_b")
        assert msg is not None

        # B receives and integrates.
        comm_b.receive(msg)
        assert comm_b.inbox_size == 1
        n_integrated = comm_b.process_inbox()
        assert n_integrated > 0
        assert comm_b.stats.links_integrated > 0

    def test_receive_filters_wrong_receiver(self):
        comm = _make_communicator("agent_c")
        msg = ConceptMessage(
            sender_id="agent_a",
            receiver_id="agent_b",  # not for agent_c
            content_type=MessageType.CAUSAL_LINKS,
        )
        comm.receive(msg)
        assert comm.inbox_size == 0  # filtered out

    def test_warning_integration(self):
        comm = _make_communicator("receiver")
        danger = frozenset({52, 57})
        msg = ConceptMessage(
            sender_id="scout",
            receiver_id="receiver",
            content_type=MessageType.WARNING,
            sks_context=danger,
            urgency=1.0,
        )
        comm.receive(msg)
        comm.process_inbox()
        assert danger in comm.warning_contexts
        assert comm.stats.warnings_received == 1

    def test_skill_integration(self):
        comm = _make_communicator("student")
        skill = _make_skill("toggle_door")
        msg = ConceptMessage(
            sender_id="teacher",
            receiver_id="student",
            content_type=MessageType.SKILL,
            skill=skill,
        )
        comm.receive(msg)
        n = comm.process_inbox()
        assert n == 1
        assert comm.stats.skills_integrated == 1

    def test_duplicate_skill_not_integrated(self):
        comm = _make_communicator("student")
        skill = _make_skill("toggle_door")
        # Register first.
        comm._skill_library.register(skill)
        msg = ConceptMessage(
            sender_id="teacher",
            receiver_id="student",
            content_type=MessageType.SKILL,
            skill=skill,
        )
        comm.receive(msg)
        n = comm.process_inbox()
        assert n == 0  # already known

    def test_flush_outbox(self):
        comm = _make_communicator("agent")
        comm.request_knowledge(frozenset({50}))
        comm.request_knowledge(frozenset({51}))
        msgs = comm.flush_outbox()
        assert len(msgs) == 2
        assert comm.flush_outbox() == []  # drained

    def test_stats_tracking(self):
        links = [(3, {50}, {50, 51})]
        comm = _make_communicator("agent", links)
        comm.share_causal_links()
        assert comm.stats.messages_sent == 1

    def test_request_triggers_response(self):
        links = [(3, {50}, {50, 51})]
        comm = _make_communicator("knowledgeable", links)
        request = ConceptMessage(
            sender_id="curious",
            receiver_id="knowledgeable",
            content_type=MessageType.REQUEST,
            sks_context=frozenset({50}),
        )
        comm.receive(request)
        comm.process_inbox()
        # Should have prepared a response in outbox.
        responses = comm.flush_outbox()
        assert len(responses) >= 1
        assert responses[0].content_type == MessageType.CAUSAL_LINKS
        assert responses[0].receiver_id == "curious"


# ── MultiAgentEnv Tests ──────────────────────────────────────────

class TestMultiAgentEnv:
    def test_create_env(self):
        env = MultiAgentEnv(n_agents=3)
        assert env.n_agents == 3
        assert len(env.agent_ids) == 3

    def test_custom_agent_ids(self):
        env = MultiAgentEnv(n_agents=2, agent_ids=["alice", "bob"])
        assert "alice" in env.agent_ids
        assert "bob" in env.agent_ids

    def test_invalid_agent_count(self):
        with pytest.raises(ValueError):
            MultiAgentEnv(n_agents=2, agent_ids=["only_one"])

    def test_set_role(self):
        env = MultiAgentEnv(n_agents=2)
        env.set_role("agent_0", "explorer")
        assert env.get_agent("agent_0").role == "explorer"

    def test_inject_knowledge(self):
        env = MultiAgentEnv(n_agents=2)
        link = CausalLink(
            action=3,
            context_sks=frozenset({50}),
            effect_sks=frozenset({51}),
            strength=0.9,
            count=3,
        )
        n = env.inject_knowledge("agent_0", [link])
        assert n == 1
        agent = env.get_agent("agent_0")
        assert agent.causal_model.n_links > 0

    def test_inject_skill(self):
        env = MultiAgentEnv(n_agents=2)
        skill = _make_skill()
        env.inject_skill("agent_0", skill)
        assert env.get_agent("agent_0").skill_library.get("pickup_key") is not None

    def test_broadcast(self):
        env = MultiAgentEnv(n_agents=3)
        msg = ConceptMessage(
            sender_id="agent_0",
            receiver_id=None,
            content_type=MessageType.WARNING,
            sks_context=frozenset({52}),
            urgency=1.0,
        )
        delivered = env.broadcast("agent_0", msg)
        assert delivered == 2  # all except sender

    def test_directed_message(self):
        env = MultiAgentEnv(n_agents=3)
        msg = ConceptMessage(
            sender_id="agent_0",
            receiver_id="agent_1",
            content_type=MessageType.WARNING,
            sks_context=frozenset({52}),
            urgency=1.0,
        )
        delivered = env.broadcast("agent_0", msg)
        assert delivered == 1  # only agent_1

    def test_exchange_round(self):
        env = MultiAgentEnv(n_agents=2)
        # Give agent_0 some knowledge.
        link = CausalLink(
            action=3,
            context_sks=frozenset({50}),
            effect_sks=frozenset({51}),
            strength=0.9,
            count=3,
        )
        env.inject_knowledge("agent_0", [link])
        # Agent 0 prepares to share.
        env.get_agent("agent_0").communicator.share_causal_links()
        # Exchange.
        n_msgs = env.exchange_round()
        assert n_msgs >= 1
        # Agent 1 should now have knowledge.
        agent1 = env.get_agent("agent_1")
        assert agent1.communicator.stats.links_integrated > 0

    def test_cooperative_episode(self):
        env = MultiAgentEnv(n_agents=2, agent_ids=["explorer", "solver"])
        env.set_role("explorer", "explorer")
        env.set_role("solver", "solver")

        # Explorer knows about keys, solver knows about doors.
        key_link = CausalLink(
            action=3,
            context_sks=frozenset({50}),
            effect_sks=frozenset({51}),
            strength=0.9,
            count=3,
        )
        door_link = CausalLink(
            action=5,
            context_sks=frozenset({51, 52}),
            effect_sks=frozenset({53}),
            strength=0.8,
            count=3,
        )

        result = env.run_cooperative_episode(
            task_links={
                "explorer": [key_link],
                "solver": [door_link],
            },
            max_rounds=5,
        )

        assert result.success
        assert result.messages_exchanged > 0
        assert result.links_transferred > 0

    def test_knowledge_overlap_initial(self):
        env = MultiAgentEnv(n_agents=2)
        # No knowledge → no overlap (both empty → edge case).
        overlap = env.knowledge_overlap()
        # Both empty sets → Jaccard = 0/0 → 0.
        assert overlap == 0.0

    def test_knowledge_overlap_after_sharing(self):
        env = MultiAgentEnv(n_agents=2)
        link = CausalLink(
            action=3,
            context_sks=frozenset({50}),
            effect_sks=frozenset({51}),
            strength=0.9,
            count=3,
        )
        # Give same knowledge to both.
        env.inject_knowledge("agent_0", [link])
        env.inject_knowledge("agent_1", [link])
        overlap = env.knowledge_overlap()
        assert overlap > 0.5  # should be high overlap

    def test_no_text_exchange(self):
        """Verify that no ConceptMessage contains natural language text."""
        env = MultiAgentEnv(n_agents=2)
        link = CausalLink(
            action=3,
            context_sks=frozenset({50}),
            effect_sks=frozenset({51}),
            strength=0.9,
            count=3,
        )
        env.inject_knowledge("agent_0", [link])
        env.get_agent("agent_0").communicator.share_causal_links()
        msgs = env.get_agent("agent_0").communicator.flush_outbox()
        for msg in msgs:
            # Messages contain concepts (SKS IDs, CausalLinks), not text.
            assert isinstance(msg.sks_context, frozenset)
            for sks_id in msg.sks_context:
                assert isinstance(sks_id, int)
            for cl in msg.causal_links:
                assert isinstance(cl.action, int)
                assert isinstance(cl.context_sks, frozenset)
                assert isinstance(cl.effect_sks, frozenset)
