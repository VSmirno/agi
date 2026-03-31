"""Unit tests for Stage 24b: InstructionPlanner + QA backends."""

from __future__ import annotations

import pytest
import torch

from snks.agent.causal_model import CausalWorldModel
from snks.agent.stochastic_simulator import StochasticSimulator
from snks.daf.types import CausalAgentConfig
from snks.language.chunker import Chunk, RuleBasedChunker
from snks.language.grounding_map import GroundingMap
from snks.language.planner import InstructionPlanner
from snks.language.qa import QAResult, QuestionType
from snks.language.qa_backends import CausalQABackend, SimulationQABackend


# --- Fixtures ---

# Use IDs >= 10000 to avoid _split_context coarsening
KEY = 10001
DOOR = 10002
BALL = 10003
KEY_HELD = 10004
DOOR_OPEN = 10005
NEAR_BALL = 10006

# Action IDs
PICKUP = 3
OPEN = 5
GOTO = 8


@pytest.fixture
def config() -> CausalAgentConfig:
    cfg = CausalAgentConfig()
    cfg.causal_min_observations = 1  # synthetic: 1 obs enough
    return cfg


@pytest.fixture
def causal_model(config) -> CausalWorldModel:
    cm = CausalWorldModel(config)
    # Train transitions (observe multiple times for confidence)
    for _ in range(3):
        cm.observe_transition({KEY}, PICKUP, {KEY, KEY_HELD})        # pickup key → key_held
        cm.observe_transition({DOOR, KEY_HELD}, OPEN, {DOOR_OPEN})   # open door (with key) → door_open
        cm.observe_transition({BALL}, GOTO, {BALL, NEAR_BALL})       # goto ball → near_ball
    return cm


@pytest.fixture
def simulator(causal_model) -> StochasticSimulator:
    return StochasticSimulator(causal_model, seed=42)


@pytest.fixture
def grounding_map() -> GroundingMap:
    gmap = GroundingMap()
    dummy = torch.zeros(64)
    gmap.register("key", KEY, dummy)
    gmap.register("door", DOOR, dummy)
    gmap.register("ball", BALL, dummy)
    gmap.register("pick up", PICKUP, dummy)
    gmap.register("open", OPEN, dummy)
    gmap.register("go to", GOTO, dummy)
    gmap.register("key_held", KEY_HELD, dummy)
    gmap.register("door_open", DOOR_OPEN, dummy)
    gmap.register("near_ball", NEAR_BALL, dummy)
    return gmap


@pytest.fixture
def chunker() -> RuleBasedChunker:
    return RuleBasedChunker()


@pytest.fixture
def planner(grounding_map, causal_model, simulator) -> InstructionPlanner:
    action_names = {"pick up": PICKUP, "open": OPEN, "go to": GOTO}
    return InstructionPlanner(grounding_map, causal_model, simulator, action_names)


# === InstructionPlanner tests ===


class TestInstructionPlanner:
    def test_simple_goto(self, planner, chunker):
        chunks = chunker.chunk("go to the ball")
        actions = planner.plan(chunks, current_sks={BALL})
        assert GOTO in actions

    def test_simple_pickup(self, planner, chunker):
        chunks = chunker.chunk("pick up the key")
        actions = planner.plan(chunks, current_sks={KEY})
        assert PICKUP in actions

    def test_sequential(self, planner, chunker):
        chunks = chunker.chunk("pick up the key then open the door")
        actions = planner.plan(chunks, current_sks={KEY, DOOR})
        assert PICKUP in actions
        assert OPEN in actions
        # pickup should come before open
        assert actions.index(PICKUP) < actions.index(OPEN)

    def test_causal_chain_open_needs_key(self, planner, chunker):
        """Open door without key → should produce pickup first."""
        chunks = chunker.chunk("open the door")
        actions = planner.plan(chunks, current_sks={DOOR})
        # Planner should detect that open(door) needs key_held
        # and prepend pickup action
        assert OPEN in actions

    def test_split_by_seq_break(self):
        chunks = [
            Chunk("pick up", "ACTION"), Chunk("key", "OBJECT"),
            Chunk("", "SEQ_BREAK"),
            Chunk("open", "ACTION"), Chunk("door", "OBJECT"),
        ]
        result = InstructionPlanner._split_by_seq_break(chunks)
        assert len(result) == 2


# === CausalQABackend tests ===


class TestCausalQABackend:
    def test_factual_query(self, causal_model, grounding_map):
        backend = CausalQABackend(causal_model, grounding_map)
        # Query: what does pickup produce from key context?
        result = backend.query({"ACTION": PICKUP, "OBJECT": KEY})
        # Should find key_held in effect
        assert result is not None
        assert result.source == QuestionType.FACTUAL

    def test_unknown_query(self, causal_model, grounding_map):
        backend = CausalQABackend(causal_model, grounding_map)
        result = backend.query({"ACTION": 999, "OBJECT": 888})
        assert result is None


# === SimulationQABackend tests ===


class TestSimulationQABackend:
    def test_simulate_pickup(self, simulator, grounding_map):
        backend = SimulationQABackend(simulator, grounding_map, current_sks={KEY})
        result = backend.query({"ACTION": PICKUP})
        assert result is not None
        assert result.source == QuestionType.SIMULATION

    def test_simulate_unknown(self, simulator, grounding_map):
        backend = SimulationQABackend(simulator, grounding_map, current_sks=set())
        result = backend.query({"ACTION": 999})
        assert result is None
