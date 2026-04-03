#!/usr/bin/env python3
"""Generate BossLevel demonstrations using BabyAI Bot oracle.

Records Bot trajectories on BabyAI-BossLevel-v0, extracts subgoals
from state transitions, saves to JSON for MissionModel training.
"""

import json
import sys
from pathlib import Path

import gymnasium as gym
import minigrid  # noqa: F401 — registers envs
from minigrid.utils.baby_ai_bot import BabyAIBot

ACTION_NAMES = ["left", "right", "forward", "pickup", "drop", "toggle", "done"]


def parse_mission_goals(mission: str) -> list[dict]:
    """Parse mission text into goal descriptors.

    Returns list of {"action": ..., "obj": ..., "color": ...} dicts.
    Mission-level goals only — navigation door-opens are NOT goals.
    """
    tokens = mission.lower().split()
    goals = []
    OBJ_TYPES = {"key", "door", "ball", "box"}
    COLORS = {"red", "green", "blue", "purple", "yellow", "grey"}

    def find_obj_after(start):
        color = ""
        for i in range(start, min(start + 5, len(tokens))):
            if tokens[i] in COLORS:
                color = tokens[i]
            if tokens[i] in OBJ_TYPES or tokens[i].rstrip("s") in OBJ_TYPES:
                return tokens[i].rstrip("s"), color
        return "", ""

    i = 0
    while i < len(tokens):
        if i + 1 < len(tokens) and tokens[i] == "pick" and tokens[i + 1] == "up":
            obj, color = find_obj_after(i + 2)
            if obj:
                goals.append({"action": "pick_up", "obj": obj, "color": color})
            i += 2
        elif tokens[i] == "open":
            obj, color = find_obj_after(i + 1)
            if obj:
                goals.append({"action": "open", "obj": obj, "color": color})
            i += 1
        elif i + 1 < len(tokens) and tokens[i] == "go" and tokens[i + 1] == "to":
            obj, color = find_obj_after(i + 2)
            if obj:
                goals.append({"action": "go_to", "obj": obj, "color": color})
            i += 2
        elif tokens[i] == "put":
            obj1, color1 = find_obj_after(i + 1)
            for j in range(i + 1, len(tokens) - 1):
                if tokens[j] == "next" and tokens[j + 1] == "to":
                    obj2, color2 = find_obj_after(j + 2)
                    if obj1 and obj2:
                        goals.append({
                            "action": "put_next_to",
                            "obj": obj1, "color": color1,
                            "obj2": obj2, "color2": color2,
                        })
                    break
            i += 1
        else:
            i += 1
    return goals


def goals_to_subgoals(goals: list[dict]) -> list[dict]:
    """Convert mission goals to expected subgoal sequence.

    Each mission goal expands to: GO_TO target + action.
    """
    subgoals = []
    for g in goals:
        if g["action"] == "pick_up":
            subgoals.append({"type": "GO_TO", "obj": g["obj"], "color": g["color"]})
            subgoals.append({"type": "PICK_UP", "obj": g["obj"], "color": g["color"]})
        elif g["action"] == "open":
            subgoals.append({"type": "GO_TO", "obj": g["obj"], "color": g["color"]})
            subgoals.append({"type": "OPEN", "obj": g["obj"], "color": g["color"]})
        elif g["action"] == "go_to":
            subgoals.append({"type": "GO_TO", "obj": g["obj"], "color": g["color"]})
        elif g["action"] == "put_next_to":
            subgoals.append({"type": "GO_TO", "obj": g["obj"], "color": g["color"]})
            subgoals.append({"type": "PICK_UP", "obj": g["obj"], "color": g["color"]})
            subgoals.append({"type": "GO_TO", "obj": g["obj2"], "color": g["color2"]})
            subgoals.append({
                "type": "PUT_NEXT_TO",
                "obj": g["obj"], "color": g["color"],
                "obj2": g["obj2"], "color2": g["color2"],
            })
    return subgoals


def record_episode(env_name: str, seed: int) -> dict | None:
    """Record a single Bot episode, return demo dict or None on failure."""
    env = gym.make(env_name)
    obs, _ = env.reset(seed=seed)
    uw = env.unwrapped
    mission = obs["mission"]

    bot = BabyAIBot(uw)
    frames = []
    done = False
    step = 0
    max_steps = uw.max_steps

    while not done and step < max_steps:
        # Snapshot state before action
        carrying = uw.carrying
        carrying_type = carrying.type if carrying else None
        carrying_color = carrying.color if carrying else None
        agent_col, agent_row = int(uw.agent_pos[0]), int(uw.agent_pos[1])
        agent_dir = int(uw.agent_dir)

        # Check door states before action (for toggle detection)
        doors_before = {}
        for x in range(uw.grid.width):
            for y in range(uw.grid.height):
                cell = uw.grid.get(x, y)
                if cell is not None and cell.type == "door":
                    doors_before[(x, y)] = cell.is_open

        try:
            action = bot.replan()
        except Exception:
            env.close()
            return None

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Check for toggled doors
        toggled_door = False
        toggled_door_color = None
        if ACTION_NAMES[action] == "toggle":
            for (x, y), was_open in doors_before.items():
                cell = uw.grid.get(x, y)
                if cell is not None and cell.type == "door" and cell.is_open != was_open:
                    toggled_door = True
                    toggled_door_color = cell.color
                    break

        # Post-action carrying state
        new_carrying = uw.carrying
        new_carrying_type = new_carrying.type if new_carrying else None
        new_carrying_color = new_carrying.color if new_carrying else None

        frame = {
            "step": step,
            "agent_col": agent_col,
            "agent_row": agent_row,
            "agent_dir": agent_dir,
            "action": ACTION_NAMES[action],
            "inventory_type": new_carrying_type,
            "inventory_color": new_carrying_color,
        }

        if toggled_door:
            frame["toggled_door"] = True
            frame["toggled_door_color"] = toggled_door_color

        if done:
            frame["reward"] = float(reward)

        frames.append(frame)
        step += 1

    success = reward > 0 if done else False

    # Extract subgoals from mission text (not from raw state transitions)
    goals = parse_mission_goals(mission)
    subgoals = goals_to_subgoals(goals) if success else []

    # Record all object positions from full grid (for navigation training)
    object_positions = []
    for x in range(uw.grid.width):
        for y in range(uw.grid.height):
            cell = uw.grid.get(x, y)
            if cell is not None and cell.type in ("key", "ball", "box", "door"):
                object_positions.append({
                    "type": cell.type,
                    "color": cell.color,
                    "col": x,
                    "row": y,
                })

    demo = {
        "env": env_name,
        "seed": seed,
        "mission": mission,
        "grid_width": uw.grid.width,
        "grid_height": uw.grid.height,
        "frames": frames,
        "subgoals_extracted": subgoals,
        "object_positions": object_positions,
        "success": success,
        "total_steps": len(frames),
    }

    env.close()
    return demo


def main():
    env_name = "BabyAI-BossLevel-v0"
    num_seeds = 200
    output_path = Path("_docs/demo_episodes_bosslevel.json")

    print(f"Generating {num_seeds} BossLevel demos...")
    demos = []
    successes = 0
    failures = 0

    for seed in range(num_seeds):
        demo = record_episode(env_name, seed)
        if demo is None or not demo["success"]:
            failures += 1
            if demo:
                print(f"  Seed {seed}: FAIL ({demo['total_steps']} steps) — {demo['mission']}")
            else:
                print(f"  Seed {seed}: ERROR")
            continue

        successes += 1
        demos.append(demo)

        if seed % 20 == 0:
            print(f"  Seed {seed}: OK ({demo['total_steps']} steps, "
                  f"{len(demo['subgoals_extracted'])} subgoals) — {demo['mission']}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(demos, f, indent=None, separators=(",", ":"))

    print(f"\nDone: {successes} successes, {failures} failures")
    print(f"Saved to {output_path} ({output_path.stat().st_size / 1024:.0f} KB)")

    # Stats
    if demos:
        mission_types = {}
        for d in demos:
            words = d["mission"].lower()
            for mt in ["put", "pick up", "open", "go to"]:
                if mt in words:
                    mission_types[mt] = mission_types.get(mt, 0) + 1
        print(f"\nMission type coverage: {mission_types}")
        avg_subgoals = sum(len(d["subgoals_extracted"]) for d in demos) / len(demos)
        avg_steps = sum(d["total_steps"] for d in demos) / len(demos)
        print(f"Avg subgoals: {avg_subgoals:.1f}, Avg steps: {avg_steps:.0f}")


if __name__ == "__main__":
    main()
