"""Tick-based agent loop for Crafter Survival Demo.

One env.step() per tick. Handles reactive checks, plan execution,
and auto-planning. Runs inside EnvThread.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

from snks.agent.crafter_pixel_env import ACTION_TO_IDX, SEMANTIC_NAMES

if TYPE_CHECKING:
    from demos.crafter_demo.engine import DemoEngine

# Resource progression for auto-planning
_RESOURCE_GOALS = ["wood", "stone_item", "coal_item", "iron_item"]

# Need → goal mapping for survival
_NEED_GOALS = {
    "food": "cow",
    "drink": "water",
    "energy": "_sleep",
}


@dataclass
class AgentState:
    """Mutable agent state across ticks."""

    plan: list[dict] | None = None  # list of ScenarioStep-like dicts
    plan_index: int = 0
    nav_phase: bool = True
    retry_count: int = 0
    rng: np.random.RandomState = field(default_factory=lambda: np.random.RandomState(42))
    resource_index: int = 0  # which resource to pursue next


def _find_nearest_object(
    semantic: np.ndarray, player_pos: np.ndarray, target_name: str
) -> tuple[int, int] | None:
    """Find nearest cell with target in semantic map. Returns (row, col) or None."""
    target_id = None
    for sid, name in SEMANTIC_NAMES.items():
        if name == target_name:
            target_id = sid
            break
    if target_id is None:
        return None

    px, py = int(player_pos[0]), int(player_pos[1])
    h, w = semantic.shape
    best_pos = None
    best_dist = float("inf")

    for r in range(h):
        for c in range(w):
            if int(semantic[r, c]) == target_id:
                dist = abs(r - px) + abs(c - py)
                if dist < best_dist:
                    best_dist = dist
                    best_pos = (r, c)
    return best_pos


def _navigate_one_step(
    engine: DemoEngine,
    target_name: str,
    rng: np.random.RandomState,
) -> str:
    """One move toward target using semantic map. Returns action name."""
    semantic = engine.last_info.get("semantic")
    player_pos = engine.last_info.get("player_pos")
    if semantic is None or player_pos is None:
        return _random_move(rng)

    target_pos = _find_nearest_object(semantic, player_pos, target_name)
    if target_pos is None:
        return _random_move(rng)

    px, py = int(player_pos[0]), int(player_pos[1])
    tx, ty = target_pos

    dx = tx - px
    dy = ty - py

    if abs(dx) >= abs(dy):
        return "move_right" if dx > 0 else "move_left"
    else:
        return "move_down" if dy > 0 else "move_up"


def _is_adjacent(engine: DemoEngine, target_name: str) -> bool:
    """Check if target is within 2 cells."""
    semantic = engine.last_info.get("semantic")
    player_pos = engine.last_info.get("player_pos")
    if semantic is None or player_pos is None:
        return False

    target_pos = _find_nearest_object(semantic, player_pos, target_name)
    if target_pos is None:
        return False

    px, py = int(player_pos[0]), int(player_pos[1])
    tx, ty = target_pos
    return abs(px - tx) <= 2 and abs(py - ty) <= 2


def _random_move(rng: np.random.RandomState) -> str:
    moves = ["move_up", "move_down", "move_left", "move_right"]
    return moves[rng.randint(0, 4)]


def _get_inv_items(info: dict) -> dict[str, int]:
    """Extract item inventory (no survival stats)."""
    inv = dict(info.get("inventory", {}))
    for k in ("health", "food", "drink", "energy"):
        inv.pop(k, None)
    return {k: v for k, v in inv.items() if v > 0}


def _get_survival(info: dict) -> dict[str, int]:
    """Extract survival stats."""
    inv = info.get("inventory", {})
    return {
        "health": inv.get("health", 9),
        "food": inv.get("food", 9),
        "drink": inv.get("drink", 9),
        "energy": inv.get("energy", 9),
    }


def _plan_to_ui(plan: list[dict] | None, plan_index: int) -> list[dict[str, str]]:
    """Convert plan to UI-friendly list with status markers."""
    if not plan:
        return []
    result = []
    for i, step in enumerate(plan):
        if i < plan_index:
            status = "done"
        elif i == plan_index:
            status = "active"
        else:
            status = "pending"
        result.append({"label": step.get("label", step.get("action", "?")), "status": status})
    return result


def _make_plan_from_chain(engine: DemoEngine, goal: str) -> list[dict] | None:
    """Generate plan from ChainGenerator for a goal."""
    with engine.model_lock:
        if engine.chain_gen is None:
            return None
        try:
            steps = engine.chain_gen.generate(goal)
        except Exception:
            return None

    if not steps:
        return None

    plan = []
    for s in steps:
        plan.append({
            "navigate_to": s.navigate_to,
            "action": s.action,
            "near_label": s.near_label,
            "label": f"{s.action} → {s.near_label}",
            "prerequisite_inv": dict(s.prerequisite_inv) if s.prerequisite_inv else {},
        })
    return plan


def _auto_plan(engine: DemoEngine, agent: AgentState) -> None:
    """Auto-generate plan based on survival needs or resource progression."""
    survival = _get_survival(engine.last_info)

    # Check survival needs first
    with engine.model_lock:
        if engine.reactive:
            need = engine.reactive.check_needs(dict(engine.last_info.get("inventory", {})))
            if need:
                need_name, target = need
                if target == "_sleep":
                    # Simple sleep plan
                    agent.plan = [{"action": "sleep", "navigate_to": None, "near_label": "sleep",
                                   "label": "sleep (energy)", "prerequisite_inv": {}}]
                else:
                    agent.plan = [{"action": "do", "navigate_to": target, "near_label": target,
                                   "label": f"seek {target} ({need_name})", "prerequisite_inv": {}}]
                agent.plan_index = 0
                agent.nav_phase = True
                engine.log_event(f"auto-plan: {need_name} need → seek {target}")
                return

    # Resource progression
    goal = _RESOURCE_GOALS[agent.resource_index % len(_RESOURCE_GOALS)]
    plan = _make_plan_from_chain(engine, goal)
    if plan:
        agent.plan = plan
        agent.plan_index = 0
        agent.nav_phase = True
        engine.log_event(f"auto-plan: {goal}")
    else:
        # Fallback: random explore
        agent.plan = None


def tick(engine: DemoEngine, agent: AgentState) -> None:
    """One tick of the agent loop. Executes one env.step()."""
    if engine.env is None or engine.last_pixels is None:
        return

    pixels = engine.last_pixels
    info = engine.last_info
    inv = dict(info.get("inventory", {}))

    # --- Step 1: Detect near object ---
    near = "empty"
    with engine.model_lock:
        if engine.detector is not None:
            try:
                near = engine.detector.detect(torch.from_numpy(pixels).float())
            except Exception:
                near = "empty"

    agent_reason = "idle"
    action_name = "noop"
    reactive_data = None

    # --- Step 2: Reactive check ---
    with engine.model_lock:
        if engine.reactive is not None:
            result = engine.reactive.check_all(near, inv)
            if result["action"] is not None:
                reactive_data = result
                agent_reason = "reactive"

                if result["action"] == "flee":
                    action_name = _random_move(agent.rng)  # flee = random away
                    engine.log_event(f"reactive: flee from {near}")
                elif result["action"] == "do":
                    action_name = "do"
                    reason = result.get("reason", "")
                    engine.log_event(f"reactive: {reason} → do")
                elif result["action"] == "sleep":
                    action_name = "sleep"
                    engine.log_event("reactive: sleep (energy)")
                elif result["action"] == "seek":
                    target = result.get("target", "water")
                    action_name = _navigate_one_step(engine, target, agent.rng)
                    engine.log_event(f"reactive: seek {target}")

    # --- Step 3: Plan execution ---
    if agent_reason == "idle" and agent.plan and agent.plan_index < len(agent.plan):
        step = agent.plan[agent.plan_index]
        agent_reason = "plan"

        if step.get("navigate_to") and agent.nav_phase:
            if _is_adjacent(engine, step["navigate_to"]):
                agent.nav_phase = False
                action_name = step["action"]
            else:
                action_name = _navigate_one_step(engine, step["navigate_to"], agent.rng)
        else:
            action_name = step["action"]
            agent.nav_phase = False

        # If we just did the action (not navigating), check if we should advance
        if not agent.nav_phase:
            # Execute action and advance
            agent.nav_phase = True
            agent.plan_index += 1
            agent.retry_count = 0
            if agent.plan_index >= len(agent.plan):
                engine.log_event(f"plan complete")
                agent.resource_index += 1
                agent.plan = None

    # --- Step 4: Auto-plan if idle ---
    if agent_reason == "idle" and engine.mode == "survival":
        _auto_plan(engine, agent)
        if agent.plan:
            agent_reason = "plan"
            # Will execute on next tick
            action_name = "noop"
    elif agent_reason == "idle" and engine.mode == "interactive":
        action_name = "noop"

    # --- Step 5: Execute action ---
    if action_name == "noop":
        action_idx = 0
    elif isinstance(action_name, str):
        action_idx = ACTION_TO_IDX.get(action_name, 0)
    else:
        action_idx = int(action_name)

    new_pixels, reward, done, new_info = engine.env.step(action_idx)

    # Track resources
    old_inv = _get_inv_items(info)
    new_inv = _get_inv_items(new_info)
    for k, v in new_inv.items():
        if v > old_inv.get(k, 0):
            engine.metrics.cur_resources += 1
            engine.log_event(f"collected {k}")

    # Track zombie encounters
    if near == "zombie":
        engine.metrics.cur_encounters += 1
        new_health = new_info.get("inventory", {}).get("health", 9)
        old_health = inv.get("health", 9)
        if new_health >= old_health:
            engine.metrics.cur_survived += 1

    engine.last_pixels = new_pixels
    engine.last_info = new_info
    engine.step_count += 1

    # --- Step 6: Update snapshot ---
    plan_ui = _plan_to_ui(agent.plan, agent.plan_index)
    snapshot = engine.build_snapshot(
        agent_action=action_name,
        agent_near=near,
        agent_reason=agent_reason,
        plan_data=plan_ui,
        plan_step=agent.plan_index,
        plan_total=len(agent.plan) if agent.plan else 0,
        reactive_data=reactive_data,
    )
    with engine.snapshot_lock:
        engine.snapshot = snapshot

    # --- Step 7: Episode done ---
    if done:
        engine.metrics.finish_episode(engine.step_count)
        engine.log_event(f"episode {engine.episode_count} ended (step {engine.step_count})")
        agent.plan = None
        agent.plan_index = 0
        # Reset immediately — no dead-env ticks
        engine.reset_env()
        agent.rng = np.random.RandomState(engine.episode_count)
        # Rebuild snapshot with fresh env state
        snapshot = engine.build_snapshot(
            agent_action="reset",
            agent_near="empty",
            agent_reason="new episode",
        )
        with engine.snapshot_lock:
            engine.snapshot = snapshot


def env_thread_loop(engine: DemoEngine) -> None:
    """Main loop for EnvThread. Processes commands and ticks."""
    agent = AgentState()

    while engine._running:
        # Process commands
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
                agent = AgentState()
            elif action == "set_mode":
                engine.mode = cmd.get("mode", "survival")
                agent.plan = None
                agent.plan_index = 0
            elif action == "set_goal":
                goal = cmd.get("goal", "")
                plan = _make_plan_from_chain(engine, goal)
                if plan:
                    agent.plan = plan
                    agent.plan_index = 0
                    agent.nav_phase = True
                    engine.log_event(f"goal set: {goal}")
            elif action == "set_speed":
                engine.target_fps = max(1, min(30, cmd.get("fps", 10)))

        # Tick
        if engine.state == "playing":
            tick(engine, agent)
            time.sleep(1.0 / engine.target_fps)
        elif engine.state == "stepping":
            tick(engine, agent)
            engine.state = "paused"
        else:
            time.sleep(0.05)  # idle sleep when paused
