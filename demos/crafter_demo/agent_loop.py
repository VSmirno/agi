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

# Chains to cycle through in survival mode
SURVIVAL_CHAINS = [
    ("wood harvest", TREE_CHAIN),
    ("iron chain", IRON_CHAIN),
]

_CRAFT_ITEMS = {
    "wood_pickaxe", "stone_pickaxe", "iron_pickaxe",
    "wood_sword", "stone_sword", "iron_sword",
}


class DemoEnvWrapper:
    """Wraps CrafterPixelEnv to hook every step() for UI updates.

    On each step():
    - Updates engine.last_pixels / last_info / step_count
    - Tracks collected/crafted items
    - Updates snapshot for WS streaming
    - Waits for FPS timing
    - Respects pause (blocks until resumed)
    """

    def __init__(self, env: CrafterPixelEnv, engine: DemoEngine) -> None:
        self._env = env
        self._engine = engine

    def step(self, action: int | str) -> tuple[Any, float, bool, dict]:
        eng = self._engine

        # Wait if paused
        while eng.state == "paused" and eng._running:
            # Process commands while paused
            _process_commands(eng)
            time.sleep(0.05)

        if not eng._running:
            # Shutting down — return dummy
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

        action_str = action if isinstance(action, str) else ""
        if isinstance(action, str) and action.startswith("place_"):
            placed = action.replace("place_", "")
            eng.metrics.record_crafted(placed)
            eng.log_event(f"placed {placed}")

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

        # Build snapshot
        snapshot = eng.build_snapshot(
            agent_action=action_str if isinstance(action, str) else str(action),
            agent_near=near,
            agent_reason="pipeline",
        )
        with eng.snapshot_lock:
            eng.snapshot = snapshot

        # FPS throttle
        time.sleep(1.0 / max(1, eng.target_fps))

        # Handle stepping mode
        if eng.state == "stepping":
            eng.state = "paused"

        return pixels, reward, done, info

    def observe(self) -> tuple[Any, dict]:
        return self._env.observe()

    def reset(self) -> tuple[Any, dict]:
        pixels, info = self._env.reset()
        self._engine.last_pixels = pixels
        self._engine.last_info = info
        return pixels, info

    # Pass through n_actions etc
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
            # Interactive: use ChainGenerator for user-specified goal
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
            # Survival: cycle through chains
            name, chain = SURVIVAL_CHAINS[chain_idx % len(SURVIVAL_CHAINS)]
            chain_idx += 1
            engine.log_event(f"chain: {name}")

            # Also try ChainGenerator chains
            with engine.model_lock:
                if engine.chain_gen:
                    goals = engine.chain_gen.available_goals()
                    resource_goals = [g for g in goals if g not in (
                        "restore_drink", "restore_energy", "restore_food")]
                    if resource_goals:
                        goal = resource_goals[(chain_idx - 1) % len(resource_goals)]
                        gen_chain = engine.chain_gen.generate(goal)
                        if gen_chain:
                            chain = gen_chain
                            engine.log_event(f"  (generated: {goal}, {len(chain)} steps)")
        else:
            # Interactive with no goal — idle
            time.sleep(0.1)
            continue

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
