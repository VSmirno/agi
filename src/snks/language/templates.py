"""Verbalization templates for Stages 21-22.

Simple format-string templates for verbalization (Stage 21)
and QA answers (Stage 22). Designed for exact transmission of
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


# --- Stage 22: QA answer templates ---


def factual_answer_template(objects: list[str]) -> str:
    """Format factual QA answer.

    Args:
        objects: list of grounded object names.

    Returns:
        "the key" / "key and ball" / "I don't know" if empty.
    """
    if not objects:
        return "I don't know"
    if len(objects) == 1:
        return f"the {objects[0]}"
    return ", ".join(objects[:-1]) + " and " + objects[-1]


def simulation_answer_template(action: str, effects: list[str]) -> str:
    """Format simulation QA answer.

    Args:
        action: action name (e.g. "pick up").
        effects: list of effect object names.

    Returns:
        "you will have key" / "nothing happens" if empty.
    """
    if not effects:
        return "nothing happens"
    effect_str = " and ".join(effects)
    return f"you will have {effect_str}"


def reflective_answer_template(reason: str) -> str:
    """Format reflective QA answer from metacog reason.

    Args:
        reason: human-readable reason string.

    Returns:
        The reason string as-is (exact transmission).
    """
    return reason
