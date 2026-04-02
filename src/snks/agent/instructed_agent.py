"""Stage 51: InstructedAgent — follows natural language instructions in MiniGrid.

Pipeline: instruction → LanguageGrounder → subgoals → SubgoalNavigator → actions.
Thin wrapper over SubgoalPlanningAgent (Stage 46-47) + LanguageGrounder (Stage 50).
"""

from __future__ import annotations

import numpy as np
import torch

from snks.agent.pathfinding import GridPathfinder
from snks.agent.subgoal_planning import (
    PlanGraph,
    Subgoal,
    SubgoalNavigator,
    _extract_symbolic,
)
from snks.agent.vsa_world_model import SDMMemory, VSACodebook, VSAEncoder
from snks.language.language_grounder import LanguageGrounder

# Object type constants
OBJ_KEY = 5
OBJ_DOOR = 4
OBJ_GOAL = 8


class InstructedAgent:
    """Agent that follows natural language instructions in MiniGrid environments.

    Takes a text instruction, parses it into subgoals via LanguageGrounder,
    builds a navigation plan from the observation, and executes it using
    BFS-based SubgoalNavigator.
    """

    def __init__(self, codebook: VSACodebook | None = None, epsilon: float = 0.05):
        if codebook is None:
            codebook = VSACodebook(dim=512, seed=42)
        self.cb = codebook
        self.encoder = VSAEncoder(codebook)
        self.grounder = LanguageGrounder(codebook)
        self.navigator = SubgoalNavigator(
            sdm=SDMMemory(n_locations=100, dim=codebook.dim),
            codebook=codebook,
            encoder=self.encoder,
            n_actions=7,
            epsilon=epsilon,
        )
        self.plan: PlanGraph | None = None
        self._subgoal_names: list[str] = []

    def set_instruction(self, instruction: str) -> list[str]:
        """Parse instruction into subgoal names.

        Returns list of subgoal names (e.g. ["pickup_key", "open_door", "reach_goal"]).
        """
        self._subgoal_names = self.grounder.to_subgoals(instruction)
        self.plan = None
        return self._subgoal_names

    def build_plan(self, obs: np.ndarray) -> bool:
        """Build navigation plan from observation for the current subgoals.

        Scans observation for object positions and creates target positions
        for each subgoal in the instruction. Returns True if plan built.
        """
        if not self._subgoal_names:
            return False

        key_pos = None
        door_pos = None
        goal_pos = None

        for r in range(obs.shape[0]):
            for c in range(obs.shape[1]):
                obj = int(obs[r, c, 0])
                if obj == OBJ_KEY:
                    key_pos = (r, c)
                elif obj == OBJ_DOOR:
                    door_pos = (r, c)
                elif obj == OBJ_GOAL:
                    goal_pos = (r, c)

        # Build subgoals and targets only for requested subgoal names
        dummy = torch.zeros(self.cb.dim)
        subgoals: list[Subgoal] = []
        targets: dict[str, tuple[int, int, int | None]] = {}

        for name in self._subgoal_names:
            if name == "pickup_key" and key_pos is not None:
                subgoals.append(Subgoal(name, dummy, dummy, "symbolic"))
                targets[name] = (key_pos[0], key_pos[1], 3)  # action 3 = pickup
            elif name == "open_door" and door_pos is not None:
                subgoals.append(Subgoal(name, dummy, dummy, "symbolic"))
                # Find door-adjacent cell
                adj = self._find_door_adjacent(obs, door_pos)
                targets[name] = (adj[0], adj[1], 5)  # action 5 = toggle
            elif name == "reach_goal" and goal_pos is not None:
                subgoals.append(Subgoal(name, dummy, dummy, "symbolic"))
                targets[name] = (goal_pos[0], goal_pos[1], None)
            elif name in ("goto_key",) and key_pos is not None:
                subgoals.append(Subgoal(name, dummy, dummy, "symbolic"))
                targets[name] = (key_pos[0], key_pos[1], None)
            elif name in ("goto_door",) and door_pos is not None:
                subgoals.append(Subgoal(name, dummy, dummy, "symbolic"))
                adj = self._find_door_adjacent(obs, door_pos)
                targets[name] = (adj[0], adj[1], None)

        if not subgoals:
            return False

        self.plan = PlanGraph(subgoals)
        self.navigator.set_target_positions(targets)
        return True

    def step(self, obs: np.ndarray) -> int:
        """Select action based on current plan and observation."""
        if self.plan is None:
            return 0  # noop

        current_sg = self.plan.current_subgoal()
        if current_sg is None:
            return 0  # plan complete

        state = self.encoder.encode(obs)
        return self.navigator.select(state, current_sg, current_obs=obs)

    def run_episode(
        self, env, instruction: str, max_steps: int = 200
    ) -> tuple[bool, int]:
        """Run one episode with given instruction.

        Returns (success, steps).
        success = True if all subgoals achieved (or env terminated with reward).
        """
        subgoals = self.set_instruction(instruction)
        if not subgoals:
            return False, 0

        obs = env.reset()
        if not self.build_plan(obs):
            return False, 0

        for step_i in range(max_steps):
            # Check subgoal advancement
            current_sg = self.plan.current_subgoal()
            if current_sg is not None:
                if self._is_subgoal_achieved(obs, current_sg):
                    complete = self.plan.advance()
                    if complete:
                        return True, step_i + 1

            action = self.step(obs)
            obs, reward, terminated, truncated, info = env.step(action)

            # Check after action
            current_sg = self.plan.current_subgoal()
            if current_sg is not None:
                if self._is_subgoal_achieved(obs, current_sg):
                    complete = self.plan.advance()
                    if complete:
                        return True, step_i + 1

            if terminated:
                # Check if we finished all subgoals
                if reward > 0 and self.plan.current_subgoal() is None:
                    return True, step_i + 1
                # reach_goal is detected by termination
                if reward > 0 and self.plan.current_subgoal() is not None:
                    sg = self.plan.current_subgoal()
                    if sg.name == "reach_goal":
                        return True, step_i + 1
                return reward > 0, step_i + 1
            if truncated:
                return False, step_i + 1

        return False, max_steps

    def _is_subgoal_achieved(self, obs: np.ndarray, subgoal: Subgoal) -> bool:
        """Check if subgoal is achieved."""
        return self.navigator.is_achieved(obs, subgoal)

    def _find_door_adjacent(
        self, obs: np.ndarray, door_pos: tuple[int, int]
    ) -> tuple[int, int]:
        """Find best adjacent cell to door for toggle action."""
        pf = GridPathfinder()
        sym = _extract_symbolic(obs)
        best_adj = None
        best_dist = float("inf")

        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            ar, ac = door_pos[0] + dr, door_pos[1] + dc
            if 0 <= ar < obs.shape[0] and 0 <= ac < obs.shape[1]:
                if int(obs[ar, ac, 0]) not in (2,):  # not wall
                    path = pf.find_path(
                        obs, (sym.agent_row, sym.agent_col), (ar, ac)
                    )
                    if path and len(path) < best_dist:
                        best_dist = len(path)
                        best_adj = (ar, ac)

        if best_adj is None:
            # Fallback: cell above door
            return (door_pos[0] - 1, door_pos[1])
        return best_adj
