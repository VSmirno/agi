"""Agent loop for Crafter Survival Demo.

Uses the REAL pipeline from experiments:
- ScenarioRunner.run_chain_with_concepts() for execution
- ChainGenerator for auto-generated chains
- IRON_CHAIN / hardcoded chains as fallback
- ReactiveCheck integrated via run_chain_with_concepts

The env is wrapped so every env.step() updates the UI snapshot,
tracks item stats, and respects play/pause/step/FPS controls.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from snks.agent.crafter_pixel_env import CrafterPixelEnv, SEMANTIC_NAMES, ACTION_TO_IDX
from snks.agent.outcome_labeler import OutcomeLabeler
from snks.agent.scenario_runner import (
    ScenarioRunner, ScenarioStep, IRON_CHAIN, TREE_CHAIN, COAL_CHAIN,
)
from snks.agent.decode_head import NEAR_CLASSES

if TYPE_CHECKING:
    from demos.crafter_demo.engine import DemoEngine

SURVIVAL_KEYS = {"health", "food", "drink", "energy"}
_MOVE_ACTIONS = {"move_up", "move_down", "move_left", "move_right"}

_CRAFT_ITEMS = {
    "wood_pickaxe", "stone_pickaxe", "iron_pickaxe",
    "wood_sword", "stone_sword", "iron_sword",
}


class DemoEnvWrapper:
    """Wraps CrafterPixelEnv to hook every step() for UI updates.

    On each step():
    - Updates engine.last_pixels / last_info / step_count
    - Tracks collected/crafted items
    - Infers current phase from action patterns
    - Updates snapshot for WS streaming
    - Waits for FPS timing
    - Respects pause (blocks until resumed)
    """

    def __init__(self, env: CrafterPixelEnv, engine: DemoEngine) -> None:
        self._env = env
        self._engine = engine
        # Phase tracking — inferred from action patterns
        self._phase = "idle"           # nav / probe / action / reactive / idle
        self._phase_target = ""        # what we're navigating to / probing
        self._move_streak = 0          # consecutive move actions
        self._last_was_do = False
        # Current chain info (set externally before run_chain)
        self.chain_steps: list[ScenarioStep] = []
        self.chain_name: str = ""
        self.current_step_idx: int = 0  # approximate — updated by tracking

    def step(self, action: int | str) -> tuple[Any, float, bool, dict]:
        eng = self._engine

        # Wait if paused
        while eng.state == "paused" and eng._running:
            _process_commands(eng)
            time.sleep(0.05)

        if not eng._running:
            return eng.last_pixels, 0.0, True, eng.last_info

        old_info = eng.last_info
        old_inv = _get_inv_items(old_info)

        pixels, reward, done, info = self._env.step(action)

        eng.last_pixels = pixels
        eng.last_info = info
        eng.step_count += 1

        # Track items
        new_inv = _get_inv_items(info)
        for k, v in new_inv.items():
            delta = v - old_inv.get(k, 0)
            if delta > 0:
                if k in _CRAFT_ITEMS:
                    eng.metrics.record_crafted(k)
                    eng.log_event(f"crafted {k}")
                else:
                    eng.metrics.record_collected(k)
                    eng.log_event(f"collected {k}")

        action_str = action if isinstance(action, str) else str(action)
        if isinstance(action, str) and action.startswith("place_"):
            placed = action.replace("place_", "")
            eng.metrics.record_crafted(placed)
            eng.log_event(f"placed {placed}")

        # Infer phase from action pattern
        self._infer_phase(action_str, old_inv, new_inv)

        # Track zombie
        near = "empty"
        with eng.model_lock:
            if eng.detector is not None:
                try:
                    near = eng.detector.detect(torch.from_numpy(pixels).float())
                except Exception:
                    pass
        if near == "zombie":
            eng.metrics.cur_encounters += 1
            old_health = old_info.get("inventory", {}).get("health", 9)
            new_health = info.get("inventory", {}).get("health", 9)
            if new_health >= old_health:
                eng.metrics.cur_survived += 1

        # Estimate which chain step we're on from inventory state
        self._estimate_step_idx(info)

        # Build plan UI from chain
        plan_ui = self._build_plan_ui()

        # Build snapshot
        reason = self._phase
        if self._phase_target:
            reason = f"{self._phase}: {self._phase_target}"

        snapshot = eng.build_snapshot(
            agent_action=action_str,
            agent_near=near,
            agent_reason=reason,
            plan_data=plan_ui,
            plan_step=self.current_step_idx,
            plan_total=len(self.chain_steps),
        )
        with eng.snapshot_lock:
            eng.snapshot = snapshot

        # FPS throttle
        time.sleep(1.0 / max(1, eng.target_fps))

        # Handle stepping mode
        if eng.state == "stepping":
            eng.state = "paused"

        return pixels, reward, done, info

    def _infer_phase(self, action: str, old_inv: dict, new_inv: dict) -> None:
        """Infer what ScenarioRunner is doing from the action."""
        is_move = action in _MOVE_ACTIONS
        is_do = action == "do"
        is_sleep = action == "sleep"
        is_craft = action.startswith("make_") or action.startswith("place_")

        if is_sleep:
            self._phase = "reactive"
            self._phase_target = "sleep"
            self._move_streak = 0
            self._last_was_do = False
            return

        if is_craft:
            self._phase = "action"
            self._phase_target = action
            self._move_streak = 0
            self._last_was_do = False
            return

        if is_do and self._last_was_do:
            # Consecutive do — shouldn't happen, probably reactive
            self._phase = "reactive"
            self._phase_target = "attack"
            self._move_streak = 0
            self._last_was_do = True
            return

        if is_do and self._move_streak >= 1:
            # move→do pattern = directional probing
            self._phase = "probe"
            self._phase_target = "do (facing)"
            self._move_streak = 0
            self._last_was_do = True
            # Check if do actually collected something
            for k, v in new_inv.items():
                if v > old_inv.get(k, 0):
                    self._phase = "action"
                    self._phase_target = f"collected {k}"
                    break
            return

        if is_do:
            self._phase = "probe"
            self._phase_target = "do"
            self._move_streak = 0
            self._last_was_do = True
            return

        if is_move:
            self._move_streak += 1
            self._last_was_do = False
            if self._move_streak >= 3:
                self._phase = "navigate"
                # Try to figure out where we're going from the chain
                if self.chain_steps and self.current_step_idx < len(self.chain_steps):
                    target = self.chain_steps[self.current_step_idx].navigate_to
                    if target:
                        self._phase_target = target
                    else:
                        self._phase_target = "repositioning"
                else:
                    self._phase_target = ""
            return

        self._move_streak = 0
        self._last_was_do = False

    def _estimate_step_idx(self, info: dict) -> None:
        """Estimate which chain step we're on based on inventory."""
        if not self.chain_steps:
            return
        inv = dict(info.get("inventory", {}))
        # Walk through chain: a step is "done" if its prereqs are met
        # and we've progressed past it
        for i, step in enumerate(self.chain_steps):
            if step.prerequisite_inv:
                if not all(inv.get(k, 0) >= v for k, v in step.prerequisite_inv.items()):
                    self.current_step_idx = max(0, i - 1)
                    return
        # If all prereqs met, we might be on the last step
        self.current_step_idx = min(self.current_step_idx, len(self.chain_steps) - 1)

    def _build_plan_ui(self) -> list[dict[str, str]]:
        """Build plan UI from current chain."""
        if not self.chain_steps:
            return []
        result = []
        for i, step in enumerate(self.chain_steps):
            label = f"{step.action}"
            if step.navigate_to:
                label = f"go to {step.navigate_to} → {step.action}"
            if step.prerequisite_inv:
                reqs = ", ".join(f"{k}:{v}" for k, v in step.prerequisite_inv.items())
                label += f" (need {reqs})"
            if step.repeat > 1:
                label += f" ×{step.repeat}"

            if i < self.current_step_idx:
                status = "done"
            elif i == self.current_step_idx:
                status = "active"
            else:
                status = "pending"
            result.append({"label": label, "status": status})
        return result

    def observe(self) -> tuple[Any, dict]:
        return self._env.observe()

    def reset(self) -> tuple[Any, dict]:
        pixels, info = self._env.reset()
        self._engine.last_pixels = pixels
        self._engine.last_info = info
        return pixels, info

    @property
    def n_actions(self) -> int:
        return self._env.n_actions

    @property
    def action_names(self) -> list[str]:
        return self._env.action_names


def _get_inv_items(info: dict) -> dict[str, int]:
    inv = dict(info.get("inventory", {}))
    for k in SURVIVAL_KEYS:
        inv.pop(k, None)
    return {k: v for k, v in inv.items() if v > 0}


def _process_commands(engine: DemoEngine) -> None:
    """Drain command queue."""
    while not engine.cmd_queue.empty():
        try:
            cmd = engine.cmd_queue.get_nowait()
        except Exception:
            break

        action = cmd.get("cmd", "")
        if action == "play":
            engine.state = "playing"
        elif action == "pause":
            engine.state = "paused"
        elif action == "step":
            engine.state = "stepping"
        elif action == "reset":
            engine.reset_env()
        elif action == "set_mode":
            engine.mode = cmd.get("mode", "survival")
        elif action == "set_goal":
            engine._pending_goal = cmd.get("goal", "")
        elif action == "set_speed":
            engine.target_fps = max(1, min(30, cmd.get("fps", 10)))
        elif action == "train":
            engine.start_training(epochs=cmd.get("epochs", 150))


def env_thread_loop(engine: DemoEngine) -> None:
    """Main loop — runs ScenarioRunner chains exactly like experiments."""
    runner = ScenarioRunner()
    labeler = OutcomeLabeler()
    rng = np.random.RandomState(42)
    chain_idx = 0

    while engine._running:
        # Process commands
        _process_commands(engine)

        # Wait if paused
        if engine.state == "paused":
            time.sleep(0.05)
            continue

        # Create wrapped env
        engine.reset_env()
        wrapped = DemoEnvWrapper(engine.env, engine)

        if engine.mode == "interactive" and hasattr(engine, "_pending_goal") and engine._pending_goal:
            goal = engine._pending_goal
            engine._pending_goal = ""
            with engine.model_lock:
                if engine.chain_gen:
                    chain = engine.chain_gen.generate(goal)
                    engine.log_event(f"goal: {goal} ({len(chain)} steps)")
                else:
                    chain = list(IRON_CHAIN)
                    engine.log_event(f"goal: {goal} (fallback IRON_CHAIN)")
        elif engine.mode == "survival":
            # Use ChainGenerator for resource goals, cycling through them
            chain = list(IRON_CHAIN)  # default fallback
            with engine.model_lock:
                if engine.chain_gen:
                    goals = engine.chain_gen.available_goals()
                    resource_goals = [g for g in goals if g not in (
                        "restore_drink", "restore_energy", "restore_food")]
                    if resource_goals:
                        goal = resource_goals[chain_idx % len(resource_goals)]
                        gen_chain = engine.chain_gen.generate(goal)
                        if gen_chain:
                            chain = gen_chain
                            engine.log_event(f"chain: {goal} ({len(chain)} steps)")
                        else:
                            engine.log_event(f"chain: IRON_CHAIN (fallback)")
                    else:
                        engine.log_event(f"chain: IRON_CHAIN (no goals)")
                else:
                    engine.log_event(f"chain: IRON_CHAIN (no chain_gen)")
            chain_idx += 1
        else:
            time.sleep(0.1)
            continue

        # Set chain info on wrapper for plan display
        wrapped.chain_steps = chain
        wrapped.chain_name = f"chain #{chain_idx}"
        wrapped.current_step_idx = 0

        # Run the chain using the REAL pipeline
        with engine.model_lock:
            detector = engine.detector
            store = engine.store
            reactive = engine.reactive

        try:
            labeled = runner.run_chain_with_concepts(
                env=wrapped,
                detector=detector,
                labeler=labeler,
                chain=chain,
                rng=rng,
                concept_store=store,
                reactive=reactive,
            )
            engine.log_event(f"chain done: {len(labeled)} labeled frames")
        except Exception as e:
            engine.log_event(f"chain error: {e}")

        # Episode stats
        engine.metrics.finish_episode(engine.step_count)
        engine.log_event(f"episode {engine.episode_count} done (step {engine.step_count})")
