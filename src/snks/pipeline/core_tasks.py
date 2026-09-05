"""Explicit task construction and evaluator-side success checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from snks.env.core_adapter import CrafterCoreAdapter
from snks.env.core_grid import CoreGridWorld, GridCoreAdapter, GridRules
from snks.env.core_types import GoalSpec, Observation


_CRAFTER_GOALS = {
    "resource_acquisition": 4,  # wood
    "resource_recovery": 2,  # drink
}
_CRAFTER_SENSOR_MAX = 9.0
_GRID_RULESETS = {
    ("door_key", "key_reusable"): GridRules(consume_key=False),
    ("door_key", "key_consumed"): GridRules(consume_key=True),
    ("push_box", "push_1"): GridRules(push_distance=1),
    ("push_box", "push_2"): GridRules(push_distance=2),
}


@dataclass(frozen=True, slots=True)
class TaskCase:
    """Evaluator-owned task metadata; only its resolved goal reaches the agent."""

    uid: str
    family: str
    ruleset: str
    seed: int
    split: str
    goal: GoalSpec
    max_steps: int


def build_case(config: dict[str, Any]) -> TaskCase:
    """Build one case from a deliberately small task dictionary."""
    family = str(config["family"])
    ruleset = str(config.get("ruleset", "default"))
    seed = int(config.get("seed", 0))
    split = str(config.get("split", "train"))
    max_steps = int(config.get("max_steps", 64))
    uid = str(config.get("uid", f"{family}:{ruleset}:{split}:{seed}"))
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")

    if family in _CRAFTER_GOALS:
        if ruleset != "default":
            raise ValueError(f"unsupported Crafter ruleset: {ruleset}")
        sensor_index = _CRAFTER_GOALS[family]
        goal = GoalSpec(image=None, ranges={sensor_index: (1.0, _CRAFTER_SENSOR_MAX)})
    else:
        rules = _grid_rules(family, ruleset)
        world = CoreGridWorld(family, rules, seed=seed, max_steps=max_steps)
        goal = GoalSpec(image=world.goal_observation(), ranges={})
        world.close()

    return TaskCase(uid, family, ruleset, seed, split, goal, max_steps)


def resolve_goal(case: TaskCase, initial: Observation) -> GoalSpec:
    """Compile relative Crafter deltas against the reset observation."""
    sensor_index = _CRAFTER_GOALS.get(case.family)
    if sensor_index is None:
        return case.goal
    if initial.schema != "crafter-v1":
        raise ValueError(f"{case.family} requires a crafter-v1 observation")
    if sensor_index >= len(initial.sensors) or not initial.sensor_mask[sensor_index]:
        raise ValueError("relative goal sensor is unavailable")

    relative_minimum = case.goal.ranges[sensor_index][0]
    minimum = float(initial.sensors[sensor_index]) + relative_minimum
    maximum = case.goal.ranges[sensor_index][1]
    if minimum > maximum:
        raise ValueError(
            f"unreachable relative goal: minimum {minimum:g} exceeds sensor ceiling "
            f"{maximum:g}"
        )
    return GoalSpec(image=None, ranges={sensor_index: (minimum, maximum)})


def score_episode(case: TaskCase, audit: list[dict[str, Any]]) -> bool:
    """Evaluate actual sensor gain or grid ground truth from a runner audit."""
    if not audit:
        return False
    sensor_index = _CRAFTER_GOALS.get(case.family)
    if sensor_index is None:
        return any(bool(entry.get("diagnostic", {}).get("success")) for entry in audit)

    initial = audit[0]
    initial_sensors = initial.get("sensors", [])
    initial_mask = initial.get("sensor_mask", [])
    if not _sensor_available(initial_sensors, initial_mask, sensor_index):
        return False
    target = float(initial_sensors[sensor_index]) + case.goal.ranges[sensor_index][0]
    if target > case.goal.ranges[sensor_index][1]:
        return False
    for entry in audit[1:]:
        sensors = entry.get("sensors", [])
        mask = entry.get("sensor_mask", [])
        if _sensor_available(sensors, mask, sensor_index):
            if float(sensors[sensor_index]) >= target:
                return True
    return False


def make_task(
    family: str,
    ruleset: str,
    seed: int,
    split: str = "train",
    max_steps: int = 64,
) -> tuple[CrafterCoreAdapter | GridCoreAdapter, TaskCase]:
    """Return an executable adapter and its evaluator-owned task case."""
    config = {
        "family": family,
        "ruleset": ruleset,
        "seed": seed,
        "split": split,
        "max_steps": max_steps,
    }
    if family == "resource_acquisition":
        adapter: CrafterCoreAdapter | GridCoreAdapter = CrafterCoreAdapter(
            max_steps=max_steps,
            local_source="tree",
        )
    elif family == "resource_recovery":
        adapter = CrafterCoreAdapter(
            max_steps=max_steps,
            initial_inventory={"drink": 4},
            local_source="water",
        )
    else:
        adapter = GridCoreAdapter(
            CoreGridWorld(
                family=family,
                rules=_grid_rules(family, ruleset),
                seed=seed,
                max_steps=max_steps,
            )
        )
    return adapter, build_case(config)


def _grid_rules(family: str, ruleset: str) -> GridRules:
    try:
        return _GRID_RULESETS[(family, ruleset)]
    except KeyError as exc:
        raise ValueError(f"unsupported core task: {family}/{ruleset}") from exc


def _sensor_available(sensors: Any, mask: Any, index: int) -> bool:
    try:
        return len(sensors) > index and len(mask) > index and bool(mask[index])
    except TypeError:
        return False
