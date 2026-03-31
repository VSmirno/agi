"""BabyAIExecutor: end-to-end instruction execution in MiniGrid (Stage 24c).

Full cycle: text instruction → parse → plan → navigate → execute → verify.
Uses GridPerception (not DAF) and GridNavigator (BFS, not learned).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from snks.language.chunker import Chunk, RuleBasedChunker
from snks.language.grid_navigator import GridNavigator
from snks.language.grid_perception import GridPerception


# MiniGrid action IDs.
ACT_PICKUP = 3
ACT_DROP = 4
ACT_TOGGLE = 5
ACT_DONE = 6

# Instruction action → MiniGrid terminal action (after navigation).
TERMINAL_ACTIONS: dict[str, int | None] = {
    "go to": None,        # just navigate, no terminal action
    "pick up": ACT_PICKUP,
    "open": ACT_TOGGLE,
    "toggle": ACT_TOGGLE,
    "put": ACT_DROP,
    "drop": ACT_DROP,
}


@dataclass
class ExecutionResult:
    """Result of executing an instruction in MiniGrid."""

    success: bool = False
    reward: float = 0.0
    steps_taken: int = 0
    trajectory: list[int] = field(default_factory=list)
    instruction: str = ""
    parsed_action: str = ""
    parsed_object: str = ""
    error: str = ""


class BabyAIExecutor:
    """End-to-end executor: instruction → actions → MiniGrid.

    Components:
    - RuleBasedChunker: text → chunks
    - GridPerception: grid state → SKS concepts + object locations
    - GridNavigator: BFS pathfinding → action sequences
    - MiniGrid env: execute actions, get reward
    """

    def __init__(self, env, perception: GridPerception) -> None:
        self._env = env
        self._perception = perception
        self._chunker = RuleBasedChunker()
        self._navigator = GridNavigator()

    def execute(self, instruction: str, max_steps: int = 100) -> ExecutionResult:
        """Execute a text instruction in the environment.

        Steps:
        1. Parse instruction with chunker
        2. Perceive grid to find target object
        3. Navigate to target (BFS)
        4. Execute terminal action (pickup/toggle/etc.)
        5. Check env reward for success

        Args:
            instruction: BabyAI text instruction.
            max_steps: maximum environment steps.

        Returns:
            ExecutionResult with success flag, reward, trajectory.
        """
        # 1. Parse instruction.
        chunks = self._chunker.chunk(instruction)
        action_text, object_text, attr_text = self._extract_roles(chunks)

        result = ExecutionResult(
            instruction=instruction,
            parsed_action=action_text,
            parsed_object=object_text,
        )

        if not action_text:
            result.error = "no action parsed"
            return result

        # 2. Perceive grid.
        env_unwrapped = self._env.unwrapped
        grid = env_unwrapped.grid
        agent_pos = tuple(env_unwrapped.agent_pos)
        agent_dir = int(env_unwrapped.agent_dir)

        carrying = getattr(env_unwrapped, 'carrying', None)
        self._perception.perceive(grid, agent_pos, agent_dir, carrying=carrying)

        # 3. Find target object.
        target_word = f"{attr_text} {object_text}" if attr_text else object_text
        target_obj = self._perception.find_object(target_word)

        if target_obj is None and attr_text:
            # Fallback: try without attribute.
            target_obj = self._perception.find_object(object_text)

        if target_obj is None:
            result.error = f"object not found: {target_word}"
            return result

        # 4. Navigate to target.
        terminal_action = TERMINAL_ACTIONS.get(action_text)
        # Always stop adjacent — MiniGrid objects are not passable,
        # and BabyAI "go to" succeeds when agent faces the object.
        stop_adjacent = True

        nav_actions = self._navigator.plan_path(
            grid, agent_pos, agent_dir, target_obj.pos,
            stop_adjacent=stop_adjacent,
        )

        # 5. Execute actions in env.
        trajectory: list[int] = []
        total_reward = 0.0
        terminated = False
        truncated = False

        for action in nav_actions:
            if len(trajectory) >= max_steps:
                break
            obs, reward, terminated, truncated, info = self._env.step(action)
            trajectory.append(action)
            total_reward += reward
            if terminated or truncated:
                break

        # 6. Execute terminal action if needed and not already done.
        if terminal_action is not None and not terminated and not truncated:
            if len(trajectory) < max_steps:
                obs, reward, terminated, truncated, info = self._env.step(terminal_action)
                trajectory.append(terminal_action)
                total_reward += reward

        result.success = total_reward > 0 or terminated and total_reward >= 0
        result.reward = total_reward
        result.steps_taken = len(trajectory)
        result.trajectory = trajectory

        # MiniGrid: success = positive reward.
        # Some envs give reward only on done action, others on reaching goal.
        if total_reward > 0:
            result.success = True
        else:
            result.success = False

        return result

    def execute_sequential(self, instruction: str, max_steps: int = 150) -> ExecutionResult:
        """Execute a sequential instruction (e.g. 'pick up X then open Y').

        Splits at SEQ_BREAK and executes each sub-instruction.
        """
        chunks = self._chunker.chunk(instruction)
        sub_instructions = self._split_by_seq_break(chunks)

        trajectory: list[int] = []
        total_reward = 0.0
        steps_used = 0

        for sub_chunks in sub_instructions:
            # Reconstruct text from chunks for single execution.
            sub_text = self._reconstruct_instruction(sub_chunks)
            sub_result = self.execute(sub_text, max_steps=max_steps - steps_used)
            trajectory.extend(sub_result.trajectory)
            total_reward += sub_result.reward
            steps_used += sub_result.steps_taken

            if sub_result.error:
                return ExecutionResult(
                    success=False,
                    reward=total_reward,
                    steps_taken=steps_used,
                    trajectory=trajectory,
                    instruction=instruction,
                    error=f"sub-instruction failed: {sub_result.error}",
                )
            if steps_used >= max_steps:
                break

        return ExecutionResult(
            success=total_reward > 0,
            reward=total_reward,
            steps_taken=steps_used,
            trajectory=trajectory,
            instruction=instruction,
        )

    @staticmethod
    def _extract_roles(chunks: list[Chunk]) -> tuple[str, str, str]:
        """Extract ACTION, OBJECT, ATTR from chunks."""
        action = ""
        obj = ""
        attr = ""
        for chunk in chunks:
            if chunk.role == "ACTION" and not action:
                action = chunk.text
            elif chunk.role == "OBJECT" and not obj:
                obj = chunk.text
            elif chunk.role == "ATTR" and not attr:
                attr = chunk.text
        return action, obj, attr

    @staticmethod
    def _split_by_seq_break(chunks: list[Chunk]) -> list[list[Chunk]]:
        """Split chunks at SEQ_BREAK markers."""
        result: list[list[Chunk]] = []
        current: list[Chunk] = []
        for chunk in chunks:
            if chunk.role == "SEQ_BREAK":
                if current:
                    result.append(current)
                    current = []
            else:
                current.append(chunk)
        if current:
            result.append(current)
        return result if result else [[]]

    @staticmethod
    def _reconstruct_instruction(chunks: list[Chunk]) -> str:
        """Reconstruct instruction text from chunks."""
        parts: list[str] = []
        for chunk in chunks:
            if chunk.role in ("ACTION", "OBJECT", "ATTR"):
                parts.append(chunk.text)
        return " ".join(parts)
