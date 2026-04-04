"""Stage 64: Curiosity-driven exploration via WM confidence.

Selects actions where the world model is least confident.
Discovers new rules by trying unknown interactions and learning from outcomes.
Supports both Crafter and MiniGrid symbolic environments.
"""

from __future__ import annotations

import random
from typing import Protocol

from snks.agent.cls_world_model import CLSWorldModel
from snks.agent.world_model_trainer import Transition


class SymbolicEnv(Protocol):
    """Protocol for symbolic environments (Crafter or MiniGrid)."""
    def reset(self) -> dict[str, str]: ...
    def observe(self) -> dict[str, str]: ...
    def available_actions(self) -> list[str]: ...
    def step(self, action: str) -> tuple[dict[str, str], float]: ...


class CuriosityExplorer:
    """Explores environment by preferring actions with lowest WM confidence."""

    def __init__(self, wm: CLSWorldModel, explore_threshold: float = 0.3,
                 seed: int = 42):
        self.wm = wm
        self.explore_threshold = explore_threshold
        self._rng = random.Random(seed)

    def select_action(self, situation: dict[str, str],
                      available_actions: list[str]) -> str:
        """Pick action with lowest WM confidence. Random tiebreaking."""
        candidates: list[str] = []
        worst_conf = 1.0

        for action in available_actions:
            _, conf, _ = self.wm.query(situation, action)
            if conf < worst_conf:
                worst_conf = conf
                candidates = [action]
            elif conf == worst_conf:
                candidates.append(action)

        return self._rng.choice(candidates)

    def explore_episode(self, env: SymbolicEnv,
                        max_steps: int = 50) -> list[Transition]:
        """Run one exploration episode. Returns discovered transitions."""
        env.reset()
        discovered: list[Transition] = []

        for _ in range(max_steps):
            situation = env.observe()
            action = self.select_action(situation, env.available_actions())
            outcome, reward = env.step(action)

            t = Transition(
                situation=situation,
                action=action,
                outcome=outcome,
                reward=reward,
            )

            # Check if this was surprising
            known_outcome, conf, _ = self.wm.query(situation, action)
            if (conf < self.explore_threshold
                    or known_outcome.get("result") != outcome.get("result")):
                discovered.append(t)

            # Train incrementally
            self.wm.train([t])

            # Cycle target for Crafter-style envs
            if hasattr(env, "next_target"):
                env.next_target()

        return discovered

    def explore(self, env: SymbolicEnv,
                n_episodes: int = 30,
                steps_per_episode: int = 50) -> list[Transition]:
        """Run multiple exploration episodes. Returns all discoveries."""
        all_discovered: list[Transition] = []
        for _ in range(n_episodes):
            discovered = self.explore_episode(env, steps_per_episode)
            all_discovered.extend(discovered)
        return all_discovered


class DirectedMiniGridExplorer(CuriosityExplorer):
    """MiniGrid explorer that systematically tests key scenarios.

    Random exploration misses rare combos like carrying key + locked door.
    This explorer sets up specific scenarios and tries all actions.
    """

    def explore_directed(self, env: SymbolicEnv,
                         n_episodes: int = 30,
                         steps_per_episode: int = 30) -> list[Transition]:
        """Run random exploration + directed scenarios."""
        from snks.agent.minigrid_env_symbolic import MiniGridSymbolicEnv
        assert isinstance(env, MiniGridSymbolicEnv)

        discovered: list[Transition] = []

        # Phase 1: Random exploration
        random_disc = self.explore(env, n_episodes=n_episodes,
                                   steps_per_episode=steps_per_episode)
        discovered.extend(random_disc)

        # Phase 2: Directed scenarios — systematically test combos
        from snks.agent.world_model_trainer import (
            CARRYABLE, COLORS, OBJ_TYPES, ACTIONS, DOOR_STATES,
        )
        colors = env.colors

        scenarios = []
        # Locked door + carrying key (same and different colors)
        for door_color in colors:
            for key_color in colors:
                scenarios.append({
                    "facing_obj": "door", "obj_color": door_color,
                    "obj_state": "locked",
                    "carrying": "key", "carrying_color": key_color,
                })
        # Carrying something + facing carryable (failed_carrying)
        for obj in CARRYABLE:
            for carried in CARRYABLE:
                scenarios.append({
                    "facing_obj": obj, "obj_color": colors[0],
                    "obj_state": "none",
                    "carrying": carried, "carrying_color": colors[0],
                })
        # Wall/solid blocking
        for obj in ("wall", "door", "key", "ball", "box"):
            state = "closed" if obj == "door" else "none"
            scenarios.append({
                "facing_obj": obj, "obj_color": colors[0],
                "obj_state": state,
                "carrying": "nothing", "carrying_color": "",
            })

        for scenario in scenarios:
            env.set_scenario(**scenario)
            situation = env.observe()
            for action in ACTIONS:
                outcome, reward = env.step(action)
                t = Transition(situation=situation, action=action,
                               outcome=outcome, reward=reward)
                known, conf, _ = self.wm.query(situation, action)
                if (conf < self.explore_threshold
                        or known.get("result") != outcome.get("result")):
                    discovered.append(t)
                self.wm.train([t])
                # Reset to same scenario for next action
                env.set_scenario(**scenario)

        return discovered


class DirectedCrafterExplorer(CuriosityExplorer):
    """Crafter-specific explorer that builds inventory before crafting.

    Strategy: first collect resources, then try crafting recipes.
    This ensures the agent has materials to discover advanced recipes.
    """

    def explore_episode(self, env: SymbolicEnv,
                        max_steps: int = 80) -> list[Transition]:
        """Directed exploration: collect phase → craft phase."""
        env.reset()
        discovered: list[Transition] = []

        # Phase 1: Collect resources (first half of episode)
        collect_steps = max_steps // 2
        for _ in range(collect_steps):
            # Prefer "do" action near resources
            situation = env.observe()
            near = situation.get("near", "empty")

            if near in ("tree", "stone", "coal", "iron", "diamond",
                        "water", "cow"):
                action = "do"
            else:
                action = self.select_action(situation, env.available_actions())

            outcome, reward = env.step(action)
            t = Transition(situation=situation, action=action,
                           outcome=outcome, reward=reward)

            known_outcome, conf, _ = self.wm.query(situation, action)
            if (conf < self.explore_threshold
                    or known_outcome.get("result") != outcome.get("result")):
                discovered.append(t)
            self.wm.train([t])

            if hasattr(env, "next_target"):
                env.next_target()

        # Phase 2: Try crafting/placing with accumulated inventory
        for _ in range(max_steps - collect_steps):
            situation = env.observe()
            near = situation.get("near", "empty")

            if near in ("table", "furnace", "empty"):
                # Try all craft/place actions — curiosity selects least known
                craft_actions = [a for a in env.available_actions()
                                 if a.startswith("make_") or a.startswith("place_")]
                if craft_actions:
                    action = self.select_action(situation, craft_actions)
                else:
                    action = self.select_action(situation, env.available_actions())
            else:
                action = self.select_action(situation, env.available_actions())

            outcome, reward = env.step(action)
            t = Transition(situation=situation, action=action,
                           outcome=outcome, reward=reward)

            known_outcome, conf, _ = self.wm.query(situation, action)
            if (conf < self.explore_threshold
                    or known_outcome.get("result") != outcome.get("result")):
                discovered.append(t)
            self.wm.train([t])

            if hasattr(env, "next_target"):
                env.next_target()

        return discovered
