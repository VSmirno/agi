"""Stage 72: Autonomous agent loop for Crafter Survival Demo.

Perceive → Decide → Act → Learn. No chains, no ScenarioRunner.

Architecture:
  pixels → CNN encoder (frozen) → z_real → ConceptStore.query_visual_scored()
  → concept + similarity → drive-based goal selection → ConceptStore.plan()
  → CrafterSpatialMap navigation → action → outcome → experiential grounding

Design: docs/superpowers/specs/2026-04-07-stage72-perception-pivot-design.md
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from snks.agent.crafter_spatial_map import CrafterSpatialMap, _step_toward, MOVE_ACTIONS
from snks.agent.outcome_labeler import OutcomeLabeler
from snks.agent.perception import (
    perceive,
    on_action_outcome,
    select_goal,
    get_drive_strengths,
    explore_action,
    ground_empty_on_start,
    ground_zombie_on_damage,
    verify_outcome,
    outcome_to_verify,
    MIN_SIMILARITY,
)

if TYPE_CHECKING:
    from demos.crafter_demo.engine import DemoEngine

SURVIVAL_KEYS = {"health", "food", "drink", "energy"}
_MOVE_ACTIONS = {"move_up", "move_down", "move_left", "move_right"}
_DIRECTIONS = ["move_up", "move_down", "move_left", "move_right"]

# Probing: try all 4 directions when doing "do" action
DO_PROBE_DIRS = ["move_up", "do", "move_down", "do", "move_left", "do", "move_right", "do"]

# How often to re-evaluate goal (steps)
REPLAN_INTERVAL = 20
# Max nav steps before giving up on target
MAX_NAV_STEPS = 200
# Sleep duration (steps)
SLEEP_STEPS = 3


class DemoEnvWrapper:
    """Wraps CrafterPixelEnv to hook every step() for UI updates."""

    def __init__(self, env: Any, engine: DemoEngine) -> None:
        self._env = env
        self._engine = engine

    def step(self, action: int | str) -> tuple[Any, float, bool, dict]:
        eng = self._engine

        # Wait if paused
        while eng.state == "paused" and eng._running:
            _process_commands(eng)
            time.sleep(0.05)

        if not eng._running:
            return eng.last_pixels, 0.0, True, eng.last_info

        old_info = eng.last_info
        old_inv = _get_inv(old_info)

        pixels, reward, done, info = self._env.step(action)

        eng.last_pixels = pixels
        eng.last_info = info
        eng.step_count += 1

        # Track items
        new_inv = _get_inv(info)
        for k, v in new_inv.items():
            if k in SURVIVAL_KEYS:
                continue
            delta = v - old_inv.get(k, 0)
            if delta > 0:
                eng.metrics.record_collected(k)
                eng.log_event(f"collected {k}")

        action_str = action if isinstance(action, str) else str(action)
        if isinstance(action, str) and action.startswith("place_"):
            placed = action.replace("place_", "")
            eng.metrics.record_crafted(placed)
            eng.log_event(f"placed {placed}")

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

    @property
    def n_actions(self) -> int:
        return self._env.n_actions

    @property
    def action_names(self) -> list[str]:
        return self._env.action_names


def _get_inv(info: dict) -> dict[str, int]:
    return dict(info.get("inventory", {}))


def _get_inv_items(info: dict) -> dict[str, int]:
    """Inventory without survival keys."""
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


def _build_snapshot(
    engine: DemoEngine,
    action_str: str,
    near_str: str,
    reason: str,
    plan_ui: list[dict],
    plan_step: int,
    plan_total: int,
    drives: dict[str, float],
    perception_sim: float,
    grounding_events: list[str],
) -> None:
    """Build and set snapshot on engine."""
    snapshot = engine.build_snapshot(
        agent_action=action_str,
        agent_near=near_str,
        agent_reason=reason,
        plan_data=plan_ui,
        plan_step=plan_step,
        plan_total=plan_total,
        drives=drives,
        perception_sim=perception_sim,
        grounding_events=grounding_events,
    )
    with engine.snapshot_lock:
        engine.snapshot = snapshot


def _do_probe(
    env: DemoEnvWrapper,
    rng: np.random.RandomState,
    labeler: OutcomeLabeler,
    target_label: str,
) -> tuple[bool, Any, dict]:
    """Directional probe: face each direction + do, check if outcome matches.

    Returns (success, pixels, info).
    """
    for direction in _DIRECTIONS:
        pixels, _, done, info = env.step(direction)
        if done:
            return False, pixels, info
        old_inv = _get_inv(info)
        pixels, _, done, info = env.step("do")
        if done:
            return False, pixels, info
        new_inv = _get_inv(info)
        label = labeler.label("do", old_inv, new_inv)
        if label == target_label:
            return True, pixels, info
    return False, pixels, info


def _plan_to_ui(plan: list, current_step: int) -> list[dict[str, str]]:
    """Convert plan steps to UI format."""
    result = []
    for i, step in enumerate(plan):
        label = f"{step.action} → {step.expected_gain}"
        if step.target:
            label = f"go to {step.target} → {step.action} → {step.expected_gain}"
        if step.requires:
            reqs = ", ".join(f"{k}:{v}" for k, v in step.requires.items())
            label += f" (need {reqs})"

        if i < current_step:
            status = "done"
        elif i == current_step:
            status = "active"
        else:
            status = "pending"
        result.append({"label": label, "status": status})
    return result


def env_thread_loop(engine: DemoEngine) -> None:
    """Autonomous agent loop: perceive → decide → act → learn."""
    labeler = OutcomeLabeler()
    rng = np.random.RandomState(42)

    while engine._running:
        _process_commands(engine)

        if engine.state == "paused":
            time.sleep(0.05)
            continue

        # --- Episode start ---
        engine.reset_env()
        env = DemoEnvWrapper(engine.env, engine)
        spatial_map = engine.spatial_map
        pixels, info = env.observe()

        # State for this episode
        current_goal = ""
        current_plan: list = []
        plan_step_idx = 0
        nav_steps = 0
        grounding_log: list[str] = []
        last_perception_sim = 0.0
        steps_since_replan = 0
        episode_done = False

        engine.log_event(f"episode {engine.episode_count} start (autonomous)")

        # --- Bootstrap: ground "empty" from first frame ---
        with engine.model_lock:
            if engine.encoder is not None:
                pix_t = torch.from_numpy(pixels).float()
                if ground_empty_on_start(pix_t, engine.encoder, engine.store):
                    engine.log_event("BOOTSTRAP: grounded 'empty' from first frame")

        while engine._running and not episode_done:
            _process_commands(engine)
            if engine.state == "paused":
                time.sleep(0.05)
                continue

            inv = _get_inv(info)
            inv_items = {k: v for k, v in inv.items() if k not in SURVIVAL_KEYS}

            # ---- 1. PERCEIVE ----
            near_concept = None
            near_str = "empty"
            z_real = None

            with engine.model_lock:
                encoder = engine.encoder
                store = engine.store

            if encoder is not None:
                pix_tensor = torch.from_numpy(pixels).float()
                near_concept, z_real = perceive(pix_tensor, encoder, store)
                if near_concept is not None:
                    near_str = near_concept.id
                    _, sim = store.query_visual_scored(z_real)
                    last_perception_sim = sim
                else:
                    last_perception_sim = 0.0

            # Update spatial map
            player_pos = info.get("player_pos", (32, 32))
            spatial_map.update(player_pos, near_str)

            # ---- 1b. ZOMBIE GROUNDING (damage detection) ----
            if z_real is not None and hasattr(engine, '_prev_inv'):
                if ground_zombie_on_damage(engine._prev_inv, inv, z_real, store):
                    engine.log_event("DISCOVERY: grounded 'zombie' from damage")
                    grounding_log.append("damage→zombie")
            engine._prev_inv = dict(inv)

            # ---- 2. REACTIVE CHECK (danger only) ----
            reactive_action = None
            with engine.model_lock:
                reactive = engine.reactive
            if reactive is not None:
                danger = reactive.check(near_str, inv)
                if danger == "do":
                    reactive_action = "do"
                elif danger == "flee":
                    reactive_action = "flee"

            if reactive_action == "flee":
                engine.log_event("FLEE from danger")
                for _ in range(4):
                    direction = _DIRECTIONS[rng.randint(0, 4)]
                    pixels, _, done, info = env.step(direction)
                    if done:
                        episode_done = True
                        break
                _build_snapshot(engine, "flee", near_str, "danger: flee",
                               _plan_to_ui(current_plan, plan_step_idx),
                               plan_step_idx, len(current_plan),
                               get_drive_strengths(inv), last_perception_sim, grounding_log[-3:])
                continue

            if reactive_action == "do":
                engine.log_event(f"ATTACK {near_str}")
                old_inv = _get_inv(info)
                pixels, _, done, info = env.step("do")
                if done:
                    episode_done = True
                _build_snapshot(engine, "do", near_str, f"danger: attack {near_str}",
                               _plan_to_ui(current_plan, plan_step_idx),
                               plan_step_idx, len(current_plan),
                               get_drive_strengths(inv), last_perception_sim, grounding_log[-3:])
                continue

            # ---- 3. GOAL SELECTION (re-evaluate periodically) ----
            if not current_plan or steps_since_replan >= REPLAN_INTERVAL:
                old_goal = current_goal
                current_goal, current_plan = select_goal(inv, store)
                plan_step_idx = 0
                nav_steps = 0
                steps_since_replan = 0

                # Handle sleep directly
                if current_goal == "restore_energy":
                    engine.log_event("SLEEP (energy low)")
                    for _ in range(SLEEP_STEPS):
                        pixels, _, done, info = env.step("sleep")
                        if done:
                            episode_done = True
                            break
                    _build_snapshot(engine, "sleep", near_str, "sleep for energy",
                                   [], 0, 0,
                                   get_drive_strengths(inv), last_perception_sim, grounding_log[-3:])
                    current_plan = []
                    continue

                if current_goal != old_goal:
                    engine.log_event(f"goal: {current_goal} ({len(current_plan)} steps)")

            steps_since_replan += 1

            # ---- 4. EXECUTE PLAN ----
            if not current_plan:
                # No plan → curiosity-driven exploration (motor babbling + spatial)
                action_str = explore_action(rng, store, inventory=inv)

                if action_str == "babble_do":
                    # Motor babbling: face random direction + do
                    direction = _DIRECTIONS[rng.randint(0, 4)]
                    pixels, _, done, info = env.step(direction)
                    if done:
                        episode_done = True
                        continue
                    if encoder is not None:
                        pix_t = torch.from_numpy(pixels).float()
                        _, z_before = perceive(pix_t, encoder, store)
                    else:
                        z_before = None
                    old_inv_b = _get_inv(info)
                    pixels, _, done, info = env.step("do")
                    if done:
                        episode_done = True
                        continue
                    new_inv_b = _get_inv(info)
                    if z_before is not None:
                        grounded = on_action_outcome(
                            "do", old_inv_b, new_inv_b, z_before, store, labeler)
                        if grounded:
                            grounding_log.append(f"babble→{grounded}")
                            engine.log_event(f"DISCOVERY: babble→{grounded}")
                            spatial_map.update(
                                info.get("player_pos", player_pos), grounded)
                        # Universal verification
                        do_out = outcome_to_verify("do", old_inv_b, new_inv_b)
                        verify_outcome(grounded or near_str, "do", do_out, store)
                    _build_snapshot(engine, "babble", near_str, "curiosity: motor babbling",
                                   [], 0, 0,
                                   get_drive_strengths(inv), last_perception_sim, grounding_log[-3:])
                    continue

                elif action_str.startswith("babble_"):
                    # Craft babbling: place_table or make_*
                    craft_action = action_str.replace("babble_", "")
                    old_inv_b = _get_inv(info)
                    pixels, _, done, info = env.step(craft_action)
                    if done:
                        episode_done = True
                        continue
                    new_inv_b = _get_inv(info)
                    grounded = on_action_outcome(
                        craft_action, old_inv_b, new_inv_b,
                        z_real if z_real is not None else torch.zeros(1), store, labeler)
                    if grounded:
                        grounding_log.append(f"craft-babble→{grounded}")
                        engine.log_event(f"DISCOVERY: craft→{grounded}")
                        spatial_map.update(
                            info.get("player_pos", player_pos), grounded)
                    craft_out = outcome_to_verify(craft_action, old_inv_b, new_inv_b)
                    verify_outcome(near_str, craft_action.split("_")[0], craft_out, store)
                    _build_snapshot(engine, craft_action, near_str, "curiosity: craft babble",
                                   [], 0, 0,
                                   get_drive_strengths(new_inv_b), last_perception_sim, grounding_log[-3:])
                    continue
                else:
                    # Spatial exploration — move to unvisited area
                    unvisited = spatial_map.unvisited_neighbors(player_pos, radius=5)
                    if unvisited:
                        target = unvisited[rng.randint(len(unvisited))]
                        action_str = _step_toward(player_pos, target, rng)
                    # else: action_str is already a random direction from explore_action

                    pixels, _, done, info = env.step(action_str)
                    if done:
                        episode_done = True
                    _build_snapshot(engine, action_str, near_str, f"explore ({current_goal})",
                                   [], 0, 0,
                                   get_drive_strengths(inv), last_perception_sim, grounding_log[-3:])
                    continue

            # We have a plan — work on current step
            if plan_step_idx >= len(current_plan):
                # Plan complete → force replan
                current_plan = []
                steps_since_replan = REPLAN_INTERVAL
                continue

            step = current_plan[plan_step_idx]

            # Check prerequisites
            if step.requires:
                has_reqs = all(
                    inv_items.get(k, 0) >= v for k, v in step.requires.items()
                )
                if not has_reqs:
                    # Missing prerequisites — replan
                    engine.log_event(f"missing prereqs for {step.expected_gain}, replanning")
                    current_plan = []
                    steps_since_replan = REPLAN_INTERVAL
                    continue

            # Navigation to target
            target_concept = step.target
            reason = f"navigate → {target_concept}"

            if step.action == "do":
                # For "do" actions: navigate to target, then probe
                if near_str == target_concept:
                    # Arrived! Do the action via directional probe
                    reason = f"do {target_concept}"
                    old_inv = _get_inv(info)

                    # Predict before action
                    prediction = store.predict_before_action(near_str, "do", inv)

                    success, pixels, info = _do_probe(env, rng, labeler, target_concept)
                    done = False  # probe handles done internally

                    new_inv = _get_inv(info)
                    actual = labeler.label("do", old_inv, new_inv)

                    # Verify prediction
                    store.verify_after_action(prediction, "do", actual, near=near_str)

                    # Experiential grounding
                    if z_real is not None:
                        grounded = on_action_outcome("do", old_inv, new_inv, z_real, store, labeler)
                        if grounded:
                            grounding_log.append(f"grounded: {grounded}")
                            engine.log_event(f"grounded: {grounded}")

                    if success:
                        engine.log_event(f"collected {step.expected_gain}")
                        plan_step_idx += 1
                        nav_steps = 0
                        # Resource consumed — invalidate map position
                        spatial_map.update(player_pos, "empty")
                    else:
                        nav_steps += 8  # probe cost
                        if nav_steps > MAX_NAV_STEPS:
                            engine.log_event(f"gave up on {target_concept}")
                            plan_step_idx += 1
                            nav_steps = 0

                    _build_snapshot(engine, "do", near_str, reason,
                                   _plan_to_ui(current_plan, plan_step_idx),
                                   plan_step_idx, len(current_plan),
                                   get_drive_strengths(new_inv), last_perception_sim, grounding_log[-3:])
                    continue

                else:
                    # Navigate toward target — with motor babbling for discovery
                    known_pos = spatial_map.find_nearest(target_concept, player_pos)
                    if known_pos is not None:
                        action_str = _step_toward(player_pos, known_pos, rng)
                    else:
                        # Target not in map — curiosity-driven exploration
                        action_str = explore_action(rng, store, inventory=inv)

                    if action_str == "babble_do":
                        # Motor babbling during navigation
                        direction = _DIRECTIONS[rng.randint(0, 4)]
                        pixels, _, done, info = env.step(direction)
                        if not done:
                            if encoder is not None:
                                pix_t = torch.from_numpy(pixels).float()
                                _, z_b = perceive(pix_t, encoder, store)
                            else:
                                z_b = None
                            old_inv_b = _get_inv(info)
                            pixels, _, done, info = env.step("do")
                            if not done and z_b is not None:
                                new_inv_b = _get_inv(info)
                                grounded = on_action_outcome(
                                    "do", old_inv_b, new_inv_b, z_b, store, labeler)
                                if grounded:
                                    grounding_log.append(f"nav-babble→{grounded}")
                                    engine.log_event(f"DISCOVERY: {grounded}")
                                    spatial_map.update(
                                        info.get("player_pos", player_pos), grounded)
                        nav_steps += 2
                        if done:
                            episode_done = True
                    else:
                        # Normal navigation step
                        if action_str in _MOVE_ACTIONS:
                            pass  # already have direction
                        else:
                            unvisited = spatial_map.unvisited_neighbors(player_pos, radius=8)
                            if unvisited:
                                explore_target = unvisited[rng.randint(len(unvisited))]
                                action_str = _step_toward(player_pos, explore_target, rng)
                            else:
                                action_str = str(rng.choice(MOVE_ACTIONS))

                        pixels, _, done, info = env.step(action_str)
                        nav_steps += 1
                        if done:
                            episode_done = True

                    if nav_steps > MAX_NAV_STEPS:
                        engine.log_event(f"nav timeout for {target_concept}")
                        plan_step_idx += 1
                        nav_steps = 0

                    _build_snapshot(engine, action_str, near_str, reason,
                                   _plan_to_ui(current_plan, plan_step_idx),
                                   plan_step_idx, len(current_plan),
                                   get_drive_strengths(inv), last_perception_sim, grounding_log[-3:])
                    continue

            elif step.action == "sleep":
                # Sleep action — execute directly
                reason = "sleep"
                pixels, _, done, info = env.step("sleep")
                if done:
                    episode_done = True
                plan_step_idx += 1
                _build_snapshot(engine, "sleep", near_str, reason,
                               _plan_to_ui(current_plan, plan_step_idx),
                               plan_step_idx, len(current_plan),
                               get_drive_strengths(_get_inv(info)), last_perception_sim, grounding_log[-3:])
                continue

            elif step.action in ("make", "place"):
                # Craft/place: navigate to target (table/empty), then do action
                crafter_action = step.action
                if step.action == "make":
                    crafter_action = f"make_{step.expected_gain}"
                elif step.action == "place":
                    crafter_action = f"place_{step.expected_gain}"

                if near_str == target_concept:
                    # At target — execute craft/place
                    reason = f"{crafter_action}"
                    old_inv = _get_inv(info)

                    prediction = store.predict_before_action(near_str, step.action, inv)

                    pixels, _, done, info = env.step(crafter_action)
                    if done:
                        episode_done = True

                    new_inv = _get_inv(info)
                    actual = labeler.label(crafter_action, old_inv, new_inv)

                    store.verify_after_action(prediction, step.action, actual, near=near_str)

                    if actual is not None:
                        engine.log_event(f"crafted {step.expected_gain}")
                        plan_step_idx += 1
                        nav_steps = 0
                    else:
                        nav_steps += 1
                        if nav_steps > 15:
                            engine.log_event(f"craft failed: {crafter_action}")
                            plan_step_idx += 1
                            nav_steps = 0

                    _build_snapshot(engine, crafter_action, near_str, reason,
                                   _plan_to_ui(current_plan, plan_step_idx),
                                   plan_step_idx, len(current_plan),
                                   get_drive_strengths(new_inv), last_perception_sim, grounding_log[-3:])
                    continue

                else:
                    # Navigate to target
                    known_pos = spatial_map.find_nearest(target_concept, player_pos)
                    if known_pos is not None:
                        action_str = _step_toward(player_pos, known_pos, rng)
                    else:
                        unvisited = spatial_map.unvisited_neighbors(player_pos, radius=8)
                        if unvisited:
                            explore_target = unvisited[rng.randint(len(unvisited))]
                            action_str = _step_toward(player_pos, explore_target, rng)
                        else:
                            action_str = str(rng.choice(MOVE_ACTIONS))

                    pixels, _, done, info = env.step(action_str)
                    nav_steps += 1
                    if done:
                        episode_done = True

                    if nav_steps > MAX_NAV_STEPS:
                        engine.log_event(f"nav timeout for {target_concept}")
                        plan_step_idx += 1
                        nav_steps = 0

                    _build_snapshot(engine, action_str, near_str, f"navigate → {target_concept}",
                                   _plan_to_ui(current_plan, plan_step_idx),
                                   plan_step_idx, len(current_plan),
                                   get_drive_strengths(inv), last_perception_sim, grounding_log[-3:])
                    continue

            else:
                # Unknown action type — skip
                plan_step_idx += 1
                continue

        # Episode over
        engine.metrics.finish_episode(engine.step_count)
        engine.log_event(f"episode {engine.episode_count} done (step {engine.step_count})")
