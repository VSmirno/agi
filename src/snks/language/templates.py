"""Verbalization templates for Stage 21.

Simple format-string templates for three verbalization types:
DESCRIBE, CAUSAL, PLAN. Designed for exact transmission of
world model content, not natural language generation.
"""

from __future__ import annotations


def describe_template(objects: list[str]) -> str:
    """Format active objects into a description.

    Args:
        objects: list of grounded object names (non-empty).

    Returns:
        "I see {obj1}, {obj2} and {objN}" or "" if empty.
    """
    if not objects:
        return ""
    if len(objects) == 1:
        return f"I see {objects[0]}"
    return "I see " + ", ".join(objects[:-1]) + " and " + objects[-1]


def causal_template(action: str, obj: str, effect: str) -> str:
    """Format a single causal link.

    Args:
        action: action name (e.g. "pick up").
        obj: object involved (e.g. "key").
        effect: resulting state (e.g. "key held").

    Returns:
        "{action} {obj} causes {effect}"
    """
    return f"{action} {obj} causes {effect}"


def plan_template(steps: list[tuple[str, str]]) -> str:
    """Format a plan as a sequence of action-object pairs.

    Args:
        steps: [(action_name, object_name), ...] in order.

    Returns:
        "I need to {action1} {obj1}, then {action2} {obj2}, ..."
        or "" if empty.
    """
    if not steps:
        return ""
    parts = [f"{action} {obj}" for action, obj in steps]
    if len(parts) == 1:
        return f"I need to {parts[0]}"
    return "I need to " + ", then ".join(parts)
