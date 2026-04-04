"""Stage 64: Curiosity exploration unit tests."""

import pytest

from snks.agent.cls_world_model import CLSWorldModel
from snks.agent.crafter_env_symbolic import CrafterSymbolicEnv
from snks.agent.crafter_trainer import (
    generate_taught_transitions,
    CRAFTER_TAUGHT,
    CRAFTER_RULES,
)
from snks.agent.curiosity_explorer import CuriosityExplorer
from snks.agent.world_model_trainer import extract_demo_transitions


class TestCrafterSymbolicEnv:
    def test_reset_clears_inventory(self):
        env = CrafterSymbolicEnv()
        env.inventory["wood"] = 5
        env.reset()
        assert env.inventory == {}

    def test_chop_tree_gives_wood(self):
        env = CrafterSymbolicEnv()
        # Find tree
        while env.current_nearby != "tree":
            env.next_target()
        outcome, reward = env.step("do")
        assert outcome["result"] == "collected"
        assert outcome["gives"] == "wood"
        assert env.inventory["wood"] == 1
        assert reward == 1.0

    def test_mine_stone_without_pickaxe_fails(self):
        env = CrafterSymbolicEnv()
        while env.current_nearby != "stone":
            env.next_target()
        outcome, reward = env.step("do")
        assert outcome["result"] == "failed_no_tool"
        assert reward == -1.0

    def test_mine_stone_with_pickaxe_succeeds(self):
        env = CrafterSymbolicEnv()
        env.inventory["wood_pickaxe"] = 1
        while env.current_nearby != "stone":
            env.next_target()
        outcome, reward = env.step("do")
        assert outcome["result"] == "collected"
        assert outcome["gives"] == "stone"

    def test_craft_consumes_resources(self):
        env = CrafterSymbolicEnv()
        env.inventory["wood"] = 1
        while env.current_nearby != "table":
            env.next_target()
        outcome, reward = env.step("make_wood_pickaxe")
        assert outcome["result"] == "crafted"
        assert "wood" not in env.inventory  # consumed
        assert env.inventory["wood_pickaxe"] == 1

    def test_unknown_action_returns_nothing(self):
        env = CrafterSymbolicEnv()
        while env.current_nearby != "tree":
            env.next_target()
        outcome, reward = env.step("make_iron_pickaxe")
        assert outcome["result"] == "nothing_happened"


class TestTaughtTransitions:
    def test_taught_count(self):
        taught = generate_taught_transitions()
        assert len(taught) == len(CRAFTER_TAUGHT)
        assert len(taught) <= 10

    def test_taught_is_subset_of_rules(self):
        taught_actions = {(r["action"], r["near"]) for r in CRAFTER_TAUGHT}
        all_actions = {(r["action"], r["near"]) for r in CRAFTER_RULES}
        assert taught_actions.issubset(all_actions)


class TestCuriosityExplorer:
    @pytest.fixture(scope="class")
    def setup(self):
        """Train WM with taught demos only, then explore."""
        taught = generate_taught_transitions()
        wm = CLSWorldModel(dim=256, n_locations=200)
        wm.train(taught)

        env = CrafterSymbolicEnv(seed=123)
        explorer = CuriosityExplorer(wm, explore_threshold=0.3, seed=123)
        discovered = explorer.explore(env, n_episodes=10, steps_per_episode=30)
        return wm, discovered

    def test_discovers_new_rules(self, setup):
        _, discovered = setup
        # Must discover at least some rules not in taught set
        assert len(discovered) > 0

    def test_discovers_untaught_recipes(self, setup):
        _, discovered = setup
        # Check for rules the teacher didn't show
        taught_actions = {(r["action"], r["near"]) for r in CRAFTER_TAUGHT}
        novel = [
            d for d in discovered
            if (d.action, d.situation.get("near", "")) not in taught_actions
            and d.outcome.get("result") not in ("nothing_happened",)
        ]
        # Should discover at least a few new successful/failure interactions
        assert len(novel) >= 3

    def test_wm_grows_from_exploration(self, setup):
        wm, _ = setup
        # After exploration, neocortex should have more rules than just taught
        taught_count = len(generate_taught_transitions())
        total_rules = len(wm.neocortex)
        assert total_rules > taught_count, (
            f"WM should learn beyond taught demos: "
            f"{total_rules} rules vs {taught_count} taught"
        )

    def test_select_action_random_tiebreak(self):
        """When all confidences are zero, should not always pick first action."""
        wm = CLSWorldModel(dim=512, n_locations=500)
        explorer = CuriosityExplorer(wm, seed=42)
        situation = {"domain": "crafter", "near": "tree"}
        actions = ["do", "place_table", "make_wood_pickaxe"]
        # Run multiple selections — should vary
        selected = set()
        for i in range(20):
            explorer._rng = __import__("random").Random(i)
            selected.add(explorer.select_action(situation, actions))
        assert len(selected) > 1  # not always the same
