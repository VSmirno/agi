"""Stage 62: Unit tests for BossLevel components.

Tests MissionModel, CausalWorldModel extensions, BossLevelAgent.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from snks.agent.boss_level_agent import BossLevelAgent, OBJ_NAME_TO_ID
from snks.agent.causal_world_model import (
    CausalWorldModel,
    RULE_GO_TO_COMPLETE,
    RULE_PUT_NEXT_TO,
)
from snks.agent.mission_model import (
    COLORS,
    MissionEncoder,
    MissionModel,
    OBJ_TYPES,
    SG_DROP,
    SG_GO_TO,
    SG_OPEN,
    SG_PICK_UP,
    SG_PUT_NEXT_TO,
    Subgoal,
    SubgoalEncoder,
)
from snks.agent.spatial_map import OBJ_BALL, OBJ_BOX
from snks.agent.vsa_world_model import VSACodebook


# ── Test MissionEncoding ──

class TestMissionEncoding:
    def setup_method(self):
        self.cb = VSACodebook(dim=512, seed=42)
        self.enc = MissionEncoder(self.cb)

    def test_tokenize_strips_punctuation(self):
        tokens = self.enc.tokenize("pick up a blue key, then open the door")
        assert "key," not in tokens
        assert "key" in tokens

    def test_tokenize_removes_stopwords(self):
        tokens = self.enc.tokenize("pick up a red ball")
        assert "a" not in tokens
        assert "pick" in tokens
        assert "red" in tokens

    def test_encode_produces_vector(self):
        vec = self.enc.encode_mission("pick up a red ball")
        assert vec.shape == (512,)
        assert vec.sum() > 0

    def test_similar_missions_similar_vectors(self):
        v1 = self.enc.encode_mission("pick up a red ball")
        v2 = self.enc.encode_mission("pick up a blue ball")
        v3 = self.enc.encode_mission("open a grey door")
        sim_same = VSACodebook.similarity(v1, v2)
        sim_diff = VSACodebook.similarity(v1, v3)
        # Missions sharing "pick up" should be more similar
        assert sim_same > sim_diff


# ── Test MissionModel ──

class TestMissionModel:
    def setup_method(self):
        self.mm = MissionModel(dim=512, seed=42)

    def test_learn_and_retrieve(self):
        sgs = [Subgoal(type=SG_GO_TO, obj="ball", color="red"),
               Subgoal(type=SG_PICK_UP, obj="ball", color="red")]
        self.mm.learn("pick up a red ball", sgs)
        result = self.mm.retrieve("pick up a red ball")
        assert len(result) == 2
        assert result[0].type == SG_GO_TO
        assert result[1].type == SG_PICK_UP

    def test_extract_from_text_pick_up(self):
        result = self.mm._extract_from_text("pick up a red ball")
        types = [s.type for s in result]
        assert types == [SG_GO_TO, SG_PICK_UP]
        assert result[1].obj == "ball"
        assert result[1].color == "red"

    def test_extract_from_text_open(self):
        result = self.mm._extract_from_text("open the blue door")
        types = [s.type for s in result]
        assert types == [SG_GO_TO, SG_OPEN]
        assert result[1].obj == "door"
        assert result[1].color == "blue"

    def test_extract_from_text_compound(self):
        result = self.mm._extract_from_text(
            "pick up a grey key and open the red door"
        )
        types = [s.type for s in result]
        assert SG_PICK_UP in types
        assert SG_OPEN in types

    def test_extract_put_next_to(self):
        result = self.mm._extract_from_text(
            "put the red box next to a grey door"
        )
        types = [s.type for s in result]
        assert SG_PUT_NEXT_TO in types
        assert SG_PICK_UP in types  # must pick up first

    def test_train_from_demos(self):
        demos = [{
            "success": True,
            "mission": "pick up a ball",
            "subgoals_extracted": [
                {"type": "GO_TO", "obj": "ball", "color": "red"},
                {"type": "PICK_UP", "obj": "ball", "color": "red"},
            ],
        }]
        n = self.mm.train_from_demos(demos)
        assert n == 1
        assert self.mm.n_trained == 1


# ── Test SubgoalExtraction ──

class TestSubgoalExtraction:
    def test_go_to_mission(self):
        mm = MissionModel()
        result = mm.retrieve("go to a red box")
        assert len(result) >= 1
        assert result[0].type == SG_GO_TO
        assert result[0].obj == "box"

    def test_compound_and(self):
        mm = MissionModel()
        result = mm.retrieve("pick up a key and open a door")
        types = [s.type for s in result]
        pick_count = types.count(SG_PICK_UP)
        open_count = types.count(SG_OPEN)
        assert pick_count >= 1
        assert open_count >= 1

    def test_after_reversal(self):
        """'X after Y' means do Y first, then X.
        But text extraction gives text order — the demo training
        should capture execution order. For now, text extraction
        preserves text order (this is a known limitation)."""
        mm = MissionModel()
        result = mm.retrieve(
            "pick up a ball after you go to the red door"
        )
        # Should have both go_to and pick_up subgoals
        types = [s.type for s in result]
        assert SG_GO_TO in types
        assert SG_PICK_UP in types


# ── Test NewRuleTypes ──

class TestNewRuleTypes:
    def setup_method(self):
        self.cwm = CausalWorldModel(dim=512, seed=42)
        self.cwm.learn_all_rules(["red", "blue", "green"])

    def test_put_next_to_preconditions(self):
        assert self.cwm.query_can_act(
            "put_next_to", carrying=True, adjacent=True
        )
        assert not self.cwm.query_can_act(
            "put_next_to", carrying=False, adjacent=True
        )

    def test_go_to_complete(self):
        assert self.cwm.query_can_act("go_to", adjacent=True)
        assert not self.cwm.query_can_act("go_to", adjacent=False)

    def test_seven_sdms_total(self):
        stats = self.cwm.get_stats()
        assert len(stats) == 7
        assert RULE_PUT_NEXT_TO in stats
        assert RULE_GO_TO_COMPLETE in stats


# ── Test ExtendedPlanner ──

class TestExtendedPlanner:
    def test_mission_to_plan(self):
        agent = BossLevelAgent()
        agent.train([{
            "success": True,
            "mission": "pick up a red ball",
            "subgoals_extracted": [
                {"type": "GO_TO", "obj": "ball", "color": "red"},
                {"type": "PICK_UP", "obj": "ball", "color": "red"},
            ],
        }])
        agent.reset("pick up a red ball")
        assert len(agent._plan) >= 2
        names = [sg.name for sg in agent._plan]
        assert "go_to" in names
        assert "pick_up" in names

    def test_open_mission_plan(self):
        agent = BossLevelAgent()
        agent.train([{
            "success": True,
            "mission": "open a blue door",
            "subgoals_extracted": [
                {"type": "GO_TO", "obj": "door", "color": "blue"},
                {"type": "OPEN", "obj": "door", "color": "blue"},
            ],
        }])
        agent.reset("open a blue door")
        names = [sg.name for sg in agent._plan]
        assert "open" in names

    def test_put_next_to_plan(self):
        agent = BossLevelAgent()
        agent.train([{
            "success": True,
            "mission": "put a red box next to a blue door",
            "subgoals_extracted": [
                {"type": "GO_TO", "obj": "box", "color": "red"},
                {"type": "PICK_UP", "obj": "box", "color": "red"},
                {"type": "GO_TO", "obj": "door", "color": "blue"},
                {"type": "PUT_NEXT_TO", "obj": "box", "color": "red",
                 "obj2": "door", "color2": "blue"},
            ],
        }])
        agent.reset("put a red box next to a blue door")
        names = [sg.name for sg in agent._plan]
        assert "pick_up" in names
        assert "put_next_to" in names

    def test_empty_plan_for_empty_mission(self):
        agent = BossLevelAgent()
        agent.reset("")
        assert len(agent._plan) == 0


# ── Test ExtendedExecutor ──

class TestExtendedExecutor:
    def test_inventory_tracking(self):
        agent = BossLevelAgent()
        agent._carrying = ("key", "red")
        assert agent._carrying == ("key", "red")
        agent._carrying = None
        assert agent._carrying is None

    def test_drop_subgoal_achievement(self):
        from snks.agent.demo_guided_agent import ExecutableSubgoal, ACT_DROP
        agent = BossLevelAgent()
        sg = ExecutableSubgoal(
            name="drop", target_pos=(0, 0),
            action_at_target=ACT_DROP, precondition=None,
        )
        agent._carrying = ("key", "red")
        assert not agent._is_subgoal_achieved(sg, 5, 5)
        agent._carrying = None
        assert agent._is_subgoal_achieved(sg, 5, 5)

    def test_go_to_subgoal_achievement(self):
        from snks.agent.demo_guided_agent import ExecutableSubgoal
        agent = BossLevelAgent()
        sg = ExecutableSubgoal(
            name="go_to", target_pos=(5, 5),
            action_at_target=None, precondition="adjacent",
        )
        # Adjacent = distance 1
        assert agent._is_subgoal_achieved(sg, 5, 6)
        assert agent._is_subgoal_achieved(sg, 4, 5)
        # Too far
        assert not agent._is_subgoal_achieved(sg, 3, 5)


# ── Test BossLevelAgentUnit ──

class TestBossLevelAgentUnit:
    def test_create_agent(self):
        agent = BossLevelAgent(grid_width=22, grid_height=22)
        assert agent.spatial_map.width == 22
        assert agent.spatial_map.height == 22
        assert not agent._trained

    def test_train_sets_trained(self):
        agent = BossLevelAgent()
        agent.train([{
            "success": True,
            "mission": "pick up a ball",
            "subgoals_extracted": [
                {"type": "GO_TO", "obj": "ball", "color": "red"},
                {"type": "PICK_UP", "obj": "ball", "color": "red"},
            ],
        }])
        assert agent._trained

    def test_reset_clears_state(self):
        agent = BossLevelAgent()
        agent._carrying = ("key", "red")
        agent._current_sg_idx = 3
        agent.explore_steps = 100
        agent.reset("pick up a ball")
        assert agent._carrying is None
        assert agent._current_sg_idx == 0
        assert agent.explore_steps == 0

    def test_stats(self):
        agent = BossLevelAgent()
        agent.reset("pick up a ball")
        stats = agent.get_stats()
        assert "explore_steps" in stats
        assert "mission" in stats
        assert stats["mission"] == "pick up a ball"


# ── Test SpatialMap Constants ──

class TestSpatialMapConstants:
    def test_ball_box_constants(self):
        assert OBJ_BALL == 6
        assert OBJ_BOX == 7

    def test_obj_name_to_id_mapping(self):
        assert OBJ_NAME_TO_ID["ball"] == OBJ_BALL
        assert OBJ_NAME_TO_ID["box"] == OBJ_BOX
