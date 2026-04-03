"""Stage 62: World Model Trainer — transition extraction and synthetic generation.

Two data sources for UnifiedWorldModel training:
1. State transitions from Bot demo trajectories
2. Exhaustive synthetic transitions from MiniGrid physics rules

Each transition: (situation, action, outcome, reward)
- situation: dict of facts about the state before action
- action: string action name
- outcome: dict of facts about state after action
- reward: +1 if state changed meaningfully, -1 if action failed, 0 if no-op
"""

from __future__ import annotations

from dataclasses import dataclass, field

# MiniGrid object types
OBJ_TYPES = ["key", "ball", "box", "door", "wall", "goal", "empty"]
INTERACTABLE = ["key", "ball", "box", "door"]
CARRYABLE = ["key", "ball", "box"]
COLORS = ["red", "green", "blue", "purple", "yellow", "grey"]
ACTIONS = ["forward", "pickup", "drop", "toggle"]
DOOR_STATES = ["open", "closed", "locked"]


@dataclass
class Transition:
    """A single state transition."""
    situation: dict[str, str]
    action: str
    outcome: dict[str, str]
    reward: float  # +1 = success, -1 = failed, 0 = no-op


def generate_synthetic_transitions() -> list[Transition]:
    """Generate exhaustive transitions from MiniGrid physics.

    Covers all (facing_object, object_state, carrying, action) combinations.
    """
    transitions: list[Transition] = []

    for obj in OBJ_TYPES:
        for color in COLORS:
            for action in ACTIONS:
                for carrying in [None] + [(c_type, c_color)
                                          for c_type in CARRYABLE
                                          for c_color in COLORS]:
                    carrying_type = carrying[0] if carrying else "nothing"
                    carrying_color = carrying[1] if carrying else ""

                    if obj == "door":
                        for door_state in DOOR_STATES:
                            t = _compute_transition(
                                obj, color, door_state,
                                carrying_type, carrying_color,
                                action,
                            )
                            if t is not None:
                                transitions.append(t)
                    else:
                        t = _compute_transition(
                            obj, color, "none",
                            carrying_type, carrying_color,
                            action,
                        )
                        if t is not None:
                            transitions.append(t)

    return transitions


def _compute_transition(facing_obj: str, obj_color: str, obj_state: str,
                        carrying_type: str, carrying_color: str,
                        action: str) -> Transition | None:
    """Compute outcome of action given situation, based on MiniGrid physics."""
    situation = {
        "facing_obj": facing_obj,
        "obj_color": obj_color,
        "obj_state": obj_state,
        "carrying": carrying_type,
        "carrying_color": carrying_color,
    }

    # === FORWARD ===
    if action == "forward":
        if facing_obj == "empty" or facing_obj == "goal":
            return Transition(
                situation=situation, action=action,
                outcome={"result": "moved", "facing_obj": "empty"},
                reward=1.0 if facing_obj == "goal" else 0.0,
            )
        if facing_obj == "door" and obj_state == "open":
            return Transition(
                situation=situation, action=action,
                outcome={"result": "moved", "facing_obj": "empty"},
                reward=0.0,
            )
        # Can't walk through wall, closed/locked door, or objects
        return Transition(
            situation=situation, action=action,
            outcome={"result": "blocked"},
            reward=-1.0,
        )

    # === PICKUP ===
    if action == "pickup":
        if facing_obj in CARRYABLE and carrying_type == "nothing":
            return Transition(
                situation=situation, action=action,
                outcome={
                    "result": "picked_up",
                    "carrying": facing_obj,
                    "carrying_color": obj_color,
                    "facing_obj": "empty",
                },
                reward=1.0,
            )
        if facing_obj in CARRYABLE and carrying_type != "nothing":
            return Transition(
                situation=situation, action=action,
                outcome={"result": "failed_carrying"},
                reward=-1.0,
            )
        return Transition(
            situation=situation, action=action,
            outcome={"result": "nothing_to_pickup"},
            reward=-1.0,
        )

    # === DROP ===
    if action == "drop":
        if carrying_type != "nothing" and facing_obj == "empty":
            return Transition(
                situation=situation, action=action,
                outcome={
                    "result": "dropped",
                    "carrying": "nothing",
                    "carrying_color": "",
                    "facing_obj": carrying_type,
                    "obj_color": carrying_color,
                },
                reward=1.0,
            )
        if carrying_type == "nothing":
            return Transition(
                situation=situation, action=action,
                outcome={"result": "nothing_to_drop"},
                reward=-1.0,
            )
        return Transition(
            situation=situation, action=action,
            outcome={"result": "drop_blocked"},
            reward=-1.0,
        )

    # === TOGGLE ===
    if action == "toggle":
        if facing_obj == "door":
            if obj_state == "closed":
                return Transition(
                    situation=situation, action=action,
                    outcome={"result": "door_opened", "obj_state": "open"},
                    reward=1.0,
                )
            if obj_state == "locked":
                # Need matching key
                if (carrying_type == "key"
                        and carrying_color == obj_color):
                    return Transition(
                        situation=situation, action=action,
                        outcome={
                            "result": "door_unlocked",
                            "obj_state": "open",
                            "carrying": "nothing",
                            "carrying_color": "",
                        },
                        reward=1.0,
                    )
                return Transition(
                    situation=situation, action=action,
                    outcome={"result": "door_still_locked"},
                    reward=-1.0,
                )
            if obj_state == "open":
                return Transition(
                    situation=situation, action=action,
                    outcome={"result": "door_closed", "obj_state": "closed"},
                    reward=0.0,
                )
        # Toggle non-door
        return Transition(
            situation=situation, action=action,
            outcome={"result": "nothing_happened"},
            reward=-1.0,
        )

    return None


def extract_demo_transitions(demos: list[dict]) -> list[Transition]:
    """Extract informative state transitions from Bot demo frames.

    Only keeps transitions where something meaningful happened
    (state changed, object interaction, door toggle).
    """
    transitions: list[Transition] = []
    n_noops = 0
    max_noops = 500  # cap no-op transitions for balance

    for demo in demos:
        if not demo.get("success"):
            continue

        frames = demo.get("frames", [])
        prev_carrying_type = None
        prev_carrying_color = None

        for i, frame in enumerate(frames):
            action = frame.get("action", "forward")
            if action in ("left", "right", "done"):
                continue  # turns don't change world state

            curr_carrying_type = frame.get("inventory_type")
            curr_carrying_color = frame.get("inventory_color", "")

            carrying_str = prev_carrying_type or "nothing"
            carrying_color_str = prev_carrying_color or ""

            # Detect what happened
            toggled = frame.get("toggled_door", False)
            picked_up = (curr_carrying_type is not None
                         and prev_carrying_type is None)
            dropped = (curr_carrying_type is None
                       and prev_carrying_type is not None)

            if toggled:
                door_color = frame.get("toggled_door_color", "grey")
                transitions.append(Transition(
                    situation={
                        "facing_obj": "door",
                        "obj_color": door_color,
                        "obj_state": "closed",  # was closed, now open
                        "carrying": carrying_str,
                        "carrying_color": carrying_color_str,
                    },
                    action="toggle",
                    outcome={"result": "door_opened", "obj_state": "open"},
                    reward=1.0,
                ))
            elif picked_up:
                transitions.append(Transition(
                    situation={
                        "facing_obj": curr_carrying_type,
                        "obj_color": curr_carrying_color or "",
                        "obj_state": "none",
                        "carrying": "nothing",
                        "carrying_color": "",
                    },
                    action="pickup",
                    outcome={
                        "result": "picked_up",
                        "carrying": curr_carrying_type,
                        "carrying_color": curr_carrying_color or "",
                    },
                    reward=1.0,
                ))
            elif dropped:
                transitions.append(Transition(
                    situation={
                        "facing_obj": "empty",
                        "obj_color": "",
                        "obj_state": "none",
                        "carrying": prev_carrying_type,
                        "carrying_color": prev_carrying_color or "",
                    },
                    action="drop",
                    outcome={
                        "result": "dropped",
                        "carrying": "nothing",
                        "carrying_color": "",
                    },
                    reward=1.0,
                ))
            elif action == "forward" and n_noops < max_noops:
                # Forward with no state change — sample some as negative
                if i % 10 == 0:
                    transitions.append(Transition(
                        situation={
                            "facing_obj": "empty",
                            "obj_color": "",
                            "obj_state": "none",
                            "carrying": carrying_str,
                            "carrying_color": carrying_color_str,
                        },
                        action="forward",
                        outcome={"result": "moved"},
                        reward=0.0,
                    ))
                    n_noops += 1

            prev_carrying_type = curr_carrying_type
            prev_carrying_color = curr_carrying_color

    return transitions
