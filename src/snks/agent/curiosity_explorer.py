"""Stage 64: Curiosity-driven exploration via WM confidence.

Selects actions where the world model is least confident.
Discovers new rules by trying unknown interactions and learning from outcomes.
"""

from __future__ import annotations

import random

from snks.agent.cls_world_model import CLSWorldModel
from snks.agent.crafter_env_symbolic import CrafterSymbolicEnv
from snks.agent.world_model_trainer import Transition


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

    def explore_episode(self, env: CrafterSymbolicEnv,
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

            # Move to next target after each action
            env.next_target()

        return discovered

    def explore(self, env: CrafterSymbolicEnv,
                n_episodes: int = 30,
                steps_per_episode: int = 50) -> list[Transition]:
        """Run multiple exploration episodes. Returns all discoveries."""
        all_discovered: list[Transition] = []
        for ep in range(n_episodes):
            discovered = self.explore_episode(env, steps_per_episode)
            all_discovered.extend(discovered)
        return all_discovered
