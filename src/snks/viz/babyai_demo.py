"""BabyAI demo data generator for the dashboard (Stage 24c).

Generates episode data with base64-encoded frames for the /babyai page.
"""

from __future__ import annotations

import base64
import io

import gymnasium as gym
import minigrid
import numpy as np
from PIL import Image

from snks.language.babyai_executor import BabyAIExecutor
from snks.language.chunker import RuleBasedChunker
from snks.language.grid_navigator import GridNavigator
from snks.language.grid_perception import GridPerception
from snks.language.grounding_map import GroundingMap


ACTION_NAMES = {
    0: "turn left", 1: "turn right", 2: "forward",
    3: "pickup", 4: "drop", 5: "toggle", 6: "done",
}

# Default scenarios for the demo.
DEFAULT_SCENARIOS = [
    ("BabyAI-GoToObj-v0", 42),
    ("BabyAI-GoToObj-v0", 7),
    ("BabyAI-GoToObj-v0", 123),
    ("BabyAI-PickupLoc-v0", 10),
    ("BabyAI-PickupLoc-v0", 33),
    ("BabyAI-PickupLoc-v0", 55),
    ("BabyAI-GoToLocalS6N4-v0", 0),
    ("BabyAI-GoToLocalS6N4-v0", 8),
]


def _frame_to_b64(frame: np.ndarray, scale: int = 5) -> str:
    img = Image.fromarray(frame)
    w, h = img.size
    img = img.resize((w * scale, h * scale), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def run_episode(env_name: str, seed: int) -> dict | None:
    """Run one episode, return JSON-serializable dict with frames."""
    env = gym.make(env_name, render_mode="rgb_array")
    obs, _ = env.reset(seed=seed)
    mission = obs["mission"]

    gmap = GroundingMap()
    perc = GridPerception(gmap)
    chunker = RuleBasedChunker()
    nav = GridNavigator()

    chunks = chunker.chunk(mission)
    action_text, object_text, attr_text = BabyAIExecutor._extract_roles(chunks)
    if not action_text:
        env.close()
        return None

    uw = env.unwrapped
    perc.perceive(uw.grid, tuple(uw.agent_pos), int(uw.agent_dir))
    target_word = f"{attr_text} {object_text}" if attr_text else object_text
    target_obj = perc.find_object(target_word)
    if target_obj is None and attr_text:
        target_obj = perc.find_object(object_text)
    if target_obj is None:
        env.close()
        return None

    # Plan.
    nav_actions = nav.plan_path(
        uw.grid, tuple(uw.agent_pos), int(uw.agent_dir),
        target_obj.pos, stop_adjacent=True,
    )
    terminal = None
    if action_text == "pick up":
        terminal = 3
    elif action_text in ("open", "toggle"):
        terminal = 5
    all_actions = list(nav_actions)
    if terminal is not None:
        all_actions.append(terminal)

    # Capture frames.
    frames = []
    frame0 = env.render()
    frames.append({
        "img": _frame_to_b64(frame0),
        "action": "START",
        "reward": 0,
    })

    total_reward = 0.0
    for action in all_actions:
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        frame = env.render()
        frames.append({
            "img": _frame_to_b64(frame),
            "action": ACTION_NAMES.get(action, str(action)),
            "reward": round(float(total_reward), 3),
        })
        if terminated:
            break

    env.close()

    # Perceived objects list.
    objects_in_grid = []
    for obj in perc.objects:
        objects_in_grid.append(f"{obj.color} {obj.obj_type}")

    return {
        "mission": mission,
        "env": env_name.replace("BabyAI-", "").replace("-v0", ""),
        "seed": seed,
        "parsed": {"action": action_text, "attr": attr_text, "object": object_text},
        "target": {
            "word": target_word,
            "type": target_obj.obj_type,
            "color": target_obj.color,
            "pos": list(target_obj.pos),
        },
        "objects": objects_in_grid,
        "plan": [ACTION_NAMES.get(a, str(a)) for a in all_actions],
        "frames": frames,
        "success": total_reward > 0,
        "total_steps": len(all_actions),
    }


def generate_demo(
    scenarios: list[tuple[str, int]] | None = None,
) -> list[dict]:
    """Generate demo episodes for all scenarios."""
    if scenarios is None:
        scenarios = DEFAULT_SCENARIOS
    episodes = []
    for env_name, seed in scenarios:
        ep = run_episode(env_name, seed)
        if ep:
            episodes.append(ep)
    return episodes
