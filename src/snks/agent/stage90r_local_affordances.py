from __future__ import annotations

from typing import Any


HOSTILE_CONCEPTS = frozenset({"zombie", "skeleton", "arrow"})
LOCAL_MOVE_ACTIONS = ("move_left", "move_right", "move_up", "move_down")
EMPTY_CONCEPTS = frozenset({"None", "empty", "unknown"})


def build_local_affordance_snapshot(
    *,
    player_pos: tuple[int, int],
    spatial_map: Any | None,
    dynamic_entities: list[Any],
    last_move: str | None,
) -> dict[str, Any]:
    facing_tile = _facing_tile(player_pos, last_move)
    facing_concept = _spatial_label_at(spatial_map, facing_tile)
    nearest_dist, nearest_dir = _nearest_hostile_distance_and_direction(
        player_pos=player_pos,
        dynamic_entities=dynamic_entities,
    )

    actions: dict[str, dict[str, Any]] = {}
    for action in LOCAL_MOVE_ACTIONS:
        actions[action] = _build_move_affordance(
            action=action,
            player_pos=player_pos,
            spatial_map=spatial_map,
            dynamic_entities=dynamic_entities,
        )
    actions["do"] = {
        "do_target_concept": facing_concept,
        "do_affordance_present": str(facing_concept or "empty") not in EMPTY_CONCEPTS,
        "do_under_contact_pressure": nearest_dist is not None and nearest_dist <= 1,
    }
    actions["sleep"] = {}

    return {
        "scene": {
            "facing_tile": list(facing_tile) if facing_tile is not None else None,
            "facing_concept": facing_concept,
            "facing_blocked": _tile_blocked(spatial_map, dynamic_entities, facing_tile),
            "nearest_hostile_distance": nearest_dist,
            "nearest_hostile_direction": nearest_dir,
        },
        "actions": actions,
    }


def _build_move_affordance(
    *,
    action: str,
    player_pos: tuple[int, int],
    spatial_map: Any | None,
    dynamic_entities: list[Any],
) -> dict[str, Any]:
    target = _move_target(player_pos, action)
    blocked_static = _tile_blocked_static(spatial_map, target)
    blocked_occupied = _tile_blocked_occupied(dynamic_entities, target)
    would_move = not blocked_static and not blocked_occupied
    post_pos = target if would_move else player_pos
    nearest_after, _nearest_dir = _nearest_hostile_distance_and_direction(
        player_pos=post_pos,
        dynamic_entities=dynamic_entities,
    )
    effective_displacement = (
        abs(int(post_pos[0]) - int(player_pos[0]))
        + abs(int(post_pos[1]) - int(player_pos[1]))
    )
    return {
        "would_move": would_move,
        "blocked_static": blocked_static,
        "blocked_occupied": blocked_occupied,
        "adjacent_hostile_after": nearest_after is not None and nearest_after <= 1,
        "contact_after": nearest_after == 0,
        "effective_displacement": int(effective_displacement),
    }


def _move_target(player_pos: tuple[int, int], action: str) -> tuple[int, int]:
    px, py = int(player_pos[0]), int(player_pos[1])
    if action == "move_right":
        return (px + 1, py)
    if action == "move_left":
        return (px - 1, py)
    if action == "move_down":
        return (px, py + 1)
    if action == "move_up":
        return (px, py - 1)
    return (px, py)


def _facing_delta(last_move: str | None) -> tuple[int, int]:
    if last_move == "move_right":
        return (1, 0)
    if last_move == "move_left":
        return (-1, 0)
    if last_move == "move_down":
        return (0, 1)
    if last_move == "move_up":
        return (0, -1)
    return (0, 0)


def _facing_tile(
    player_pos: tuple[int, int],
    last_move: str | None,
) -> tuple[int, int] | None:
    dx, dy = _facing_delta(last_move)
    if dx == 0 and dy == 0:
        return None
    return (int(player_pos[0]) + dx, int(player_pos[1]) + dy)


def _spatial_label_at(
    spatial_map: Any | None,
    pos: tuple[int, int] | None,
) -> str | None:
    if spatial_map is None or pos is None or not hasattr(spatial_map, "_map"):
        return None
    entry = spatial_map._map.get((int(pos[0]), int(pos[1])))
    if entry is None:
        return None
    if isinstance(entry, tuple):
        return str(entry[0])
    return str(entry)


def _tile_blocked_static(
    spatial_map: Any | None,
    pos: tuple[int, int] | None,
) -> bool:
    if spatial_map is None or pos is None or not hasattr(spatial_map, "is_blocked"):
        return False
    return bool(spatial_map.is_blocked(pos))


def _tile_blocked_occupied(
    dynamic_entities: list[Any],
    pos: tuple[int, int] | None,
) -> bool:
    if pos is None:
        return False
    return any(tuple(entity.position) == tuple(pos) for entity in dynamic_entities)


def _tile_blocked(
    spatial_map: Any | None,
    dynamic_entities: list[Any],
    pos: tuple[int, int] | None,
) -> bool:
    return _tile_blocked_static(spatial_map, pos) or _tile_blocked_occupied(
        dynamic_entities, pos
    )


def _nearest_hostile_distance_and_direction(
    *,
    player_pos: tuple[int, int],
    dynamic_entities: list[Any],
) -> tuple[int | None, str | None]:
    best_dist: int | None = None
    best_dir: str | None = None
    for entity in dynamic_entities:
        if str(getattr(entity, "concept_id", "")) not in HOSTILE_CONCEPTS:
            continue
        ex, ey = int(entity.position[0]), int(entity.position[1])
        dx = ex - int(player_pos[0])
        dy = ey - int(player_pos[1])
        dist = abs(dx) + abs(dy)
        if best_dist is not None and dist >= best_dist:
            continue
        best_dist = dist
        if abs(dx) >= abs(dy) and dx != 0:
            best_dir = "right" if dx > 0 else "left"
        elif dy != 0:
            best_dir = "down" if dy > 0 else "up"
        else:
            best_dir = "contact"
    return best_dist, best_dir
