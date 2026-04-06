"""Stage 70: ScenarioRunner — FSM-based curriculum data collection.

Runs scenario chains in Crafter environments to collect labeled training data
for the outcome encoder. Guarantees coverage of rare objects (coal, iron)
by following tool-dependency chains.

No info["semantic"] used. Only:
  - info["inventory"]  — proprioception
  - info["player_pos"] — proprioception
  - NearDetector       — perception (for navigation)
  - OutcomeLabeler     — outcome detection (for labeling)

Usage:
    runner = ScenarioRunner()
    labeled = runner.run_chain(env, detector, labeler, CRAFTER_CHAIN, rng)
    # labeled: list of (pixels_tensor, near_idx)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import torch

from snks.agent.decode_head import NEAR_TO_IDX
from snks.agent.outcome_labeler import OutcomeLabeler
from snks.agent.crafter_spatial_map import CrafterSpatialMap, find_target_with_map

_ALL4 = ["move_up", "move_down", "move_left", "move_right"]
WINDOW_SIZE = 5     # retrospective labeling window (frames before success)
DO_RETRIES = 25     # directional probe cycles for "do" actions
ACTION_RETRIES = 15  # retry budget for craft/place actions


@dataclass
class ScenarioStep:
    """One step in a scenario chain.

    Defines navigation target, action to perform, success label, and constraints.

    Attributes:
        navigate_to: near_label to navigate to before acting. None = act in place.
        action: Crafter action name (e.g. "do", "make_wood_pickaxe", "place_table").
        near_label: label assigned to frames on successful execution.
        prerequisite_inv: minimum required inventory before running this step.
        repeat: how many successful executions to collect for this step.
        max_nav_steps: step budget for the navigation phase.
        do_retries: directional probe cycles for "do" actions.
    """
    navigate_to: str | None
    action: str
    near_label: str
    prerequisite_inv: dict[str, int] = field(default_factory=dict)
    repeat: int = 1
    max_nav_steps: int = 300
    do_retries: int = DO_RETRIES


# ---------------------------------------------------------------------------
# Standard Crafter scenario chain (S1→S7)
# ---------------------------------------------------------------------------

#: Full dependency chain from wood to iron.
#: Each step requires the previous steps to have been completed (inventory state).
CRAFTER_CHAIN: list[ScenarioStep] = [
    # S1: Harvest wood ×3 — no prerequisites
    ScenarioStep("tree", "do", "tree", repeat=3),
    # S2: Place crafting table — requires ≥2 wood in inventory
    ScenarioStep(None, "place_table", "empty", prerequisite_inv={"wood": 2}),
    # S3: Craft wood pickaxe — navigate to placed table
    ScenarioStep("table", "make_wood_pickaxe", "table"),
    # S4: Harvest stone ×4 — pickaxe not strictly required but part of the chain
    ScenarioStep("stone", "do", "stone", repeat=4),
    # S5: Craft stone pickaxe — requires ≥3 stone
    ScenarioStep("table", "make_stone_pickaxe", "table", prerequisite_inv={"stone": 3}),
    # S6: Harvest coal — requires wood_pickaxe
    ScenarioStep("coal", "do", "coal", prerequisite_inv={"wood_pickaxe": 1}, repeat=2),
    # S7: Harvest iron — requires stone_pickaxe
    ScenarioStep("iron", "do", "iron", prerequisite_inv={"stone_pickaxe": 1}, repeat=2),
]

#: Shorter chain for Phase 0 nav encoder bootstrap (tree+stone only, no tools needed).
BOOTSTRAP_CHAIN: list[ScenarioStep] = [
    ScenarioStep("tree", "do", "tree", repeat=3),
    ScenarioStep("stone", "do", "stone", repeat=3),
]


class ScenarioRunner:
    """FSM executor for scenario chains.

    Executes a sequence of ScenarioSteps in a single env episode, tracking
    inventory state across steps. On each success: retrospectively labels
    WINDOW_SIZE frames as near_label.

    Design decisions:
    - "do" actions: directional probing (face each of 4 directions then do).
      Crafter's "do" acts in the player's current facing direction — without
      this, stone/coal/iron yield 0 successes.
    - craft/place actions: simple retry loop with random repositioning.
    - The spatial map is reset per run_chain call (new episode = new map).
    """

    def run_chain(
        self,
        env: object,
        detector: object,
        labeler: OutcomeLabeler,
        chain: list[ScenarioStep],
        rng: np.random.RandomState,
    ) -> list[tuple[torch.Tensor, int]]:
        """Execute scenario chain in a running env.

        Args:
            env: CrafterPixelEnv instance, already reset.
            detector: NearDetector for navigation.
            labeler: OutcomeLabeler for success detection.
            chain: list of ScenarioStep to execute in order.
            rng: random state for reproducibility.

        Returns:
            List of (pixels_tensor, near_idx) training pairs.
            Stops early if a step's prerequisites are not met or navigation fails.
        """
        smap = CrafterSpatialMap()
        pixels_np, info = env.observe()
        window_buf: deque = deque(maxlen=WINDOW_SIZE + DO_RETRIES * 5)
        window_buf.append(pixels_np)
        labeled: list[tuple[torch.Tensor, int]] = []

        for step in chain:
            near_idx = NEAR_TO_IDX.get(step.near_label)
            if near_idx is None:
                continue

            for _ in range(step.repeat):
                inv = dict(info.get("inventory", {}))
                if not self._prereqs_met(inv, step.prerequisite_inv):
                    break  # prerequisites lost mid-repeat, abandon step

                # Navigation phase
                if step.navigate_to is not None:
                    pixels_t, info, found = find_target_with_map(
                        env, detector, smap, step.navigate_to,
                        max_steps=step.max_nav_steps, rng=rng,
                    )
                    if not found:
                        break
                    pixels_np = (
                        pixels_t.numpy() if isinstance(pixels_t, torch.Tensor) else pixels_t
                    )
                    window_buf.append(pixels_np)

                # Action phase
                if step.action == "do":
                    success, info = self._probe_do(
                        env, info, labeler, step.near_label, near_idx,
                        rng, window_buf, labeled, step.do_retries,
                    )
                else:
                    success, info, pixels_np = self._probe_action(
                        env, info, labeler, step.action, step.near_label,
                        near_idx, pixels_np, rng, labeled,
                    )

                if not success:
                    break  # step failed, stop chain

        return labeled

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prereqs_met(self, inv: dict[str, int], prereqs: dict[str, int]) -> bool:
        return all(inv.get(k, 0) >= v for k, v in prereqs.items())

    def _probe_do(
        self,
        env: object,
        info: dict,
        labeler: OutcomeLabeler,
        target_near: str,
        near_idx: int,
        rng: np.random.RandomState,
        window_buf: deque,
        labeled: list,
        do_retries: int,
    ) -> tuple[bool, dict]:
        """Directional probing for "do" actions.

        Strategy: face each of the 4 directions (move→do) repeatedly near target.
        Crafter's "do" only acts in the player's current facing direction.

        Returns:
            (success, info_after)
        """
        for _ in range(do_retries):
            for direction in _ALL4:
                pix_np, _, done, info_face = env.step(direction)
                window_buf.append(pix_np)
                if done:
                    return False, info_face

                inv_before = dict(info_face.get("inventory", {}))
                pix_np, _, done, info_after = env.step("do")
                inv_after = dict(info_after.get("inventory", {}))

                label = labeler.label("do", inv_before, inv_after)
                if label == target_near:
                    for frame in list(window_buf)[-WINDOW_SIZE:]:
                        labeled.append((torch.from_numpy(frame), near_idx))
                    return True, info_after

                info = info_after
                if done:
                    return False, info_after

            # Random reposition between direction cycles
            move = str(rng.choice(_ALL4))
            pix_np, _, done, info = env.step(move)
            window_buf.append(pix_np)
            if done:
                return False, info

        return False, info

    def _probe_action(
        self,
        env: object,
        info: dict,
        labeler: OutcomeLabeler,
        action: str,
        target_near: str,
        near_idx: int,
        pixels_np: np.ndarray,
        rng: np.random.RandomState,
        labeled: list,
    ) -> tuple[bool, dict, np.ndarray]:
        """Simple retry loop for craft/place actions.

        Returns:
            (success, info_after, pixels_np_after)
        """
        for _ in range(ACTION_RETRIES):
            inv_before = dict(info.get("inventory", {}))
            pix_np, _, done, info_after = env.step(action)
            inv_after = dict(info_after.get("inventory", {}))

            label = labeler.label(action, inv_before, inv_after)
            if label == target_near:
                labeled.append((torch.from_numpy(pixels_np), near_idx))
                return True, info_after, pix_np

            if done:
                return False, info_after, pix_np

            # Reposition and retry
            move = str(rng.choice(_ALL4))
            pix_np, _, done, info = env.step(move)
            if done:
                return False, info, pix_np
            pixels_np = pix_np

        return False, info, pixels_np
