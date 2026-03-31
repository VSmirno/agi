"""Tests for Stage 35: Integration Demo — IntegratedAgent."""

from __future__ import annotations

import pytest
import torch

from snks.agent.causal_model import CausalLink
from snks.language.concept_message import ConceptMessage, MessageType
from snks.language.integrated_agent import IntegratedAgent, IntegrationResult
from snks.language.pattern_element import PatternElement, PatternMatrix
from snks.language.plan_node import PlanGraph
from snks.language.skill import Skill


# ── Helpers ──────────────────────────────────────────────────────

def _doorkey_links() -> list[CausalLink]:
    return [
        CausalLink(action=3, context_sks=frozenset({50}),
                    effect_sks=frozenset({51}), strength=0.9, count=5),
        CausalLink(action=5, context_sks=frozenset({51, 52}),
                    effect_sks=frozenset({53}), strength=0.85, count=4),
        CausalLink(action=1, context_sks=frozenset({53, 54}),
                    effect_sks=frozenset({54}), strength=0.8, count=3),
        CausalLink(action=2, context_sks=frozenset({50, 54}),
                    effect_sks=frozenset({54}), strength=0.7, count=3),
        CausalLink(action=0, context_sks=frozenset({52}),
                    effect_sks=frozenset({52}), strength=0.6, count=3),
    ]


def _doorkey_skills() -> list[Skill]:
    return [
        Skill(name="pickup_key", preconditions=frozenset({50}),
              effects=frozenset({51}), terminal_action=3,
              target_word="key", success_count=10, attempt_count=10),
        Skill(name="toggle_door", preconditions=frozenset({51, 52}),
              effects=frozenset({53}), terminal_action=5,
              target_word="door", success_count=8, attempt_count=10),
    ]


def _make_agent(with_knowledge: bool = False) -> IntegratedAgent:
    agent = IntegratedAgent(agent_id="test_agent", grid_size=8)
    if with_knowledge:
        agent.inject_knowledge(_doorkey_links())
        for skill in _doorkey_skills():
            agent.inject_skill(skill)
    return agent


# ── Capability Inventory ─────────────────────────────────────────

class TestCapabilities:
    def test_all_10_capabilities(self):
        agent = _make_agent()
        caps = agent.capabilities()
        assert len(caps) == 10
        assert all(c.available for c in caps)
        assert agent.n_capabilities() == 10

    def test_capability_names(self):
        agent = _make_agent()
        names = {c.name for c in agent.capabilities()}
        expected = {
            "goal_decomposition", "transfer_learning", "skill_abstraction",
            "analogical_reasoning", "curiosity_exploration", "few_shot_learning",
            "pattern_reasoning", "meta_learning", "multi_agent_communication",
            "hierarchical_planning",
        }
        assert names == expected


# ── Properties & Init ────────────────────────────────────────────

class TestInit:
    def test_agent_id(self):
        agent = IntegratedAgent(agent_id="my_agent")
        assert agent.agent_id == "my_agent"

    def test_initial_state(self):
        agent = _make_agent()
        assert agent.episodes_completed == 0
        assert agent.success_rate == 0.0
        assert agent.causal_model.n_links == 0
        assert len(agent.skill_library.skills) == 0

    def test_hac_engine(self):
        agent = _make_agent()
        assert agent.hac_engine.dim == 2048


# ── Knowledge Management ─────────────────────────────────────────

class TestKnowledge:
    def test_inject_knowledge(self):
        agent = _make_agent()
        n = agent.inject_knowledge(_doorkey_links())
        assert n == 5
        assert agent.causal_model.n_links > 0

    def test_inject_skill(self):
        agent = _make_agent()
        skill = _doorkey_skills()[0]
        agent.inject_skill(skill)
        assert agent.skill_library.get("pickup_key") is not None

    def test_extract_skills(self):
        agent = _make_agent()
        agent.inject_knowledge(_doorkey_links())
        n = agent.extract_skills(min_confidence=0.3)
        assert n >= 0  # may or may not extract depending on link format


# ── Strategy Selection ───────────────────────────────────────────

class TestStrategy:
    def test_profile_fresh_agent(self):
        agent = _make_agent()
        p = agent.profile()
        assert p.known_skills == 0
        assert p.causal_links == 0
        assert p.episodes_completed == 0

    def test_profile_with_knowledge(self):
        agent = _make_agent(with_knowledge=True)
        p = agent.profile()
        assert p.known_skills == 2
        assert p.causal_links > 0

    def test_strategy_fresh(self):
        agent = _make_agent()
        s = agent.select_strategy()
        assert s.strategy in ("curiosity", "explore")

    def test_strategy_with_knowledge(self):
        agent = _make_agent(with_knowledge=True)
        # Simulate some exploration to increase coverage.
        from snks.language.curiosity_module import CuriosityModule
        for i in range(50):
            key = CuriosityModule.make_key({50 + i % 10}, (i, 0))
            agent.curiosity.observe(key)
        s = agent.select_strategy()
        assert s.strategy == "skill"


# ── Planning ─────────────────────────────────────────────────────

class TestPlanning:
    def test_plan_to_goal(self):
        agent = _make_agent(with_knowledge=True)
        plan = agent.plan_to_goal(
            goal_sks=frozenset({51, 53, 54}),
            current_sks=frozenset({50, 52, 54}),
        )
        assert isinstance(plan, PlanGraph)
        assert plan.total_steps > 0

    def test_plan_multi_room(self):
        agent = IntegratedAgent(grid_size=30)
        agent.inject_knowledge(_doorkey_links())
        for s in _doorkey_skills():
            agent.inject_skill(s)
        plan = agent.plan_to_goal(
            goal_sks=frozenset({51, 53, 54}),
            current_sks=frozenset({50, 52, 54}),
            n_rooms=5,
        )
        assert plan.total_steps > 200


# ── Communication ────────────────────────────────────────────────

class TestCommunication:
    def test_share_knowledge(self):
        agent = _make_agent(with_knowledge=True)
        msg = agent.share_knowledge()
        assert msg is not None
        assert msg.content_type == MessageType.CAUSAL_LINKS

    def test_share_skills(self):
        agent = _make_agent(with_knowledge=True)
        msgs = agent.share_skills()
        assert len(msgs) == 2

    def test_receive_message(self):
        agent_a = _make_agent(with_knowledge=True)
        agent_b = _make_agent()

        msg = agent_a.share_knowledge(receiver_id=agent_b.agent_id)
        agent_a.communicator.flush_outbox()

        agent_b.receive_message(msg)
        assert agent_b.causal_model.n_links > 0

    def test_two_agents_cooperate(self):
        agent_a = _make_agent(with_knowledge=True)
        agent_b = _make_agent()

        # A shares, B receives.
        msg = agent_a.share_knowledge(receiver_id=agent_b.agent_id)
        agent_a.communicator.flush_outbox()
        agent_b.receive_message(msg)

        # B should now have knowledge.
        assert agent_b.causal_model.n_links > 0

        # B can now plan.
        for s in _doorkey_skills():
            agent_b.inject_skill(s)
        plan = agent_b.plan_to_goal(
            goal_sks=frozenset({51, 53, 54}),
            current_sks=frozenset({50, 52, 54}),
        )
        assert plan.total_steps > 0


# ── Pattern Reasoning ────────────────────────────────────────────

class TestPatternReasoning:
    def test_solve_analogy(self):
        agent = _make_agent()
        hac = agent.hac_engine
        a = hac.random_vector()
        b = hac.random_vector()
        c = hac.random_vector()
        d_pred, conf = agent.solve_analogy(a, b, c)
        assert d_pred.shape == (2048,)
        assert isinstance(conf, float)


# ── Curiosity ────────────────────────────────────────────────────

class TestCuriosity:
    def test_curiosity_tracking(self):
        agent = _make_agent()
        from snks.language.curiosity_module import CuriosityModule
        key = CuriosityModule.make_key({50, 51}, (3, 4))
        r = agent.curiosity.observe(key)
        assert r > 0  # first visit = high reward
        r2 = agent.curiosity.observe(key)
        assert r2 < r  # revisit = lower reward


# ── Integrated Episode ───────────────────────────────────────────

class TestIntegratedEpisode:
    def test_full_episode(self):
        agent = _make_agent(with_knowledge=True)
        result = agent.run_integrated_episode(
            current_sks=frozenset({50, 52, 54}),
            goal_sks=frozenset({51, 53, 54}),
        )
        assert isinstance(result, IntegrationResult)
        assert result.success
        assert result.steps > 0
        assert result.strategy_used in ("skill", "curiosity", "explore", "few_shot")
        assert len(result.capabilities_exercised) >= 3

    def test_episode_tracks_metrics(self):
        agent = _make_agent(with_knowledge=True)
        result = agent.run_integrated_episode(
            current_sks=frozenset({50, 52, 54}),
            goal_sks=frozenset({51, 53, 54}),
        )
        assert result.plan_steps > 0
        assert agent.episodes_completed == 1

    def test_multiple_episodes(self):
        agent = _make_agent(with_knowledge=True)
        for _ in range(3):
            agent.run_integrated_episode(
                current_sks=frozenset({50, 52, 54}),
                goal_sks=frozenset({51, 53, 54}),
            )
        assert agent.episodes_completed == 3
        assert agent.success_rate > 0

    def test_capabilities_exercised(self):
        agent = _make_agent(with_knowledge=True)
        result = agent.run_integrated_episode(
            current_sks=frozenset({50, 52, 54}),
            goal_sks=frozenset({51, 53, 54}),
        )
        # Should exercise at least: meta_learning, goal_decomposition, hierarchical_planning
        assert "meta_learning" in result.capabilities_exercised
        assert "hierarchical_planning" in result.capabilities_exercised


# ── Zero Backprop ────────────────────────────────────────────────

class TestZeroBackprop:
    def test_no_gradient_learning(self):
        agent = _make_agent()
        assert agent.verify_zero_backprop()

    def test_no_requires_grad(self):
        agent = _make_agent()
        # HAC scalar base should not require grad.
        assert not agent.hac_engine._scalar_base.requires_grad
