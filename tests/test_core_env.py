"""Behavioral safeguards for the transferable core environments."""

import numpy as np
import pytest
from minigrid.core.world_object import Key

from snks.env.core_adapter import CrafterCoreAdapter, project_crafter_observation
from snks.env.core_grid import CoreGridWorld, GridCoreAdapter, GridRules, PushLayout
from snks.env.core_types import Observation
from snks.pipeline.core_tasks import build_case, resolve_goal


def test_crafter_projection_exposes_only_allowlisted_inventory_sensors() -> None:
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    rgb[7, 11] = (3, 5, 8)
    info = {
        "inventory": {"health": 8, "wood": 3},
        "diamond": 99,
        "player_pos": (17, 23),
        "semantic": np.ones((64, 64), dtype=np.uint8),
    }

    obs = project_crafter_observation(
        rgb,
        info,
        names=("health", "wood", "diamond"),
        step=4,
    )

    assert obs.schema == "crafter-v1"
    assert obs.step == 4
    assert obs.rgb[:, 7, 11].tolist() == [3, 5, 8]
    assert obs.sensors.tolist() == [8.0, 3.0, 0.0]
    assert obs.sensor_mask.tolist() == [True, True, False]
    assert not hasattr(obs, "info")


@pytest.mark.parametrize(
    "info",
    [
        {"inventory": None},
        {"inventory": []},
        {"inventory": {"health": np.nan}},
        {"inventory": {"health": np.inf}},
        {"inventory": {"health": "8"}},
    ],
)
def test_crafter_projection_rejects_present_corrupt_inventory_data(
    info: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="inventory"):
        project_crafter_observation(
            np.zeros((64, 64, 3), dtype=np.uint8),
            info,
            names=("health",),
            step=0,
        )


def _face_cell_from_left(world: CoreGridWorld, position: tuple[int, int]) -> None:
    world.agent_pos = np.array([position[0] - 1, position[1]])
    world.agent_dir = 0


@pytest.mark.parametrize(
    ("consume_key", "expected_carrying"),
    [(False, Key), (True, type(None))],
)
def test_door_rule_changes_whether_unlocking_consumes_the_key(
    consume_key: bool,
    expected_carrying: type,
) -> None:
    world = CoreGridWorld(
        family="door_key",
        rules=GridRules(consume_key=consume_key),
        seed=5,
    )
    adapter = GridCoreAdapter(world)
    adapter.reset(seed=5)
    _face_cell_from_left(world, world.door_pos)
    world.carrying = Key("yellow")

    transition = adapter.step(3)

    door = world.grid.get(*world.door_pos)
    assert door is not None and door.is_open
    assert isinstance(world.carrying, expected_carrying)
    assert not transition.terminated


@pytest.mark.parametrize("distance", [1, 2])
def test_push_rule_moves_the_box_by_the_declared_distance(distance: int) -> None:
    world = CoreGridWorld(
        family="push_box",
        rules=GridRules(push_distance=distance),
        seed=7,
    )
    adapter = GridCoreAdapter(world)
    adapter.reset(seed=7)
    start = world.box_pos
    _face_cell_from_left(world, start)

    adapter.step(3)

    assert world.diagnostic_snapshot()["box_pos"] == (
        start[0] + distance,
        start[1],
    )


def test_grid_goal_image_is_a_separate_local_desired_observation() -> None:
    world = CoreGridWorld(
        family="push_box",
        rules=GridRules(push_distance=2),
        seed=11,
    )
    adapter = GridCoreAdapter(world)
    live = adapter.reset(seed=11)

    desired = adapter.goal_observation()

    assert desired.schema == "grid-v1"
    assert desired.rgb.shape == (3, 64, 64)
    assert desired.sensors.tolist() == [0.0]
    assert not np.shares_memory(desired.rgb, live.rgb)
    assert not np.array_equal(desired.rgb, live.rgb)


def test_push_layout_changes_geometry_and_keeps_goal_reachable() -> None:
    layout = PushLayout(
        agent_pos=(3, 5), agent_dir=3, box_pos=(3, 4), goal_pos=(3, 2)
    )
    world = CoreGridWorld(
        family="push_box",
        rules=GridRules(push_distance=1),
        seed=13,
        layout=layout,
    )
    adapter = GridCoreAdapter(world)
    try:
        initial = adapter.reset(seed=13)
        desired = adapter.goal_observation()
        for action in (3, 2, 3):
            transition = adapter.step(action)

        assert world.diagnostic_snapshot()["goal_pos"] == layout.goal_pos
        assert transition.terminated
        assert np.array_equal(transition.after.rgb, desired.rgb)
        assert initial.schema == desired.schema == "grid-v1"
        assert len(initial.sensors) == 1
    finally:
        adapter.close()


@pytest.mark.parametrize(
    ("family", "rules", "actions"),
    [
        (
            "door_key",
            GridRules(consume_key=False),
            (1, 2, 3, 0, 2, 0, 2, 1, 3, 2, 2, 2),
        ),
        (
            "door_key",
            GridRules(consume_key=True),
            (1, 2, 3, 0, 2, 0, 2, 1, 3, 2, 2, 2),
        ),
        ("push_box", GridRules(push_distance=1), (3, 2, 3)),
        ("push_box", GridRules(push_distance=2), (3,)),
    ],
)
def test_grid_goal_matches_a_reachable_hand_executed_terminal(
    family: str,
    rules: GridRules,
    actions: tuple[int, ...],
) -> None:
    world = CoreGridWorld(family=family, rules=rules, seed=23)
    adapter = GridCoreAdapter(world)
    try:
        adapter.reset(seed=23)
        desired = adapter.goal_observation()
        for action in actions:
            transition = adapter.step(action)

        assert transition.terminated
        assert np.array_equal(transition.after.rgb, desired.rgb)
        assert np.array_equal(transition.after.sensors, desired.sensors)
    finally:
        adapter.close()


@pytest.mark.parametrize(
    ("family", "sensor_index", "initial_value"),
    [("resource_acquisition", 4, 2.0), ("resource_recovery", 2, 4.0)],
)
def test_relative_crafter_goal_is_compiled_from_the_reset_observation(
    family: str,
    sensor_index: int,
    initial_value: float,
) -> None:
    sensors = np.array([8.0, 7.0, 4.0, 6.0, 2.0], dtype=np.float32)
    sensors[sensor_index] = initial_value
    initial = Observation(
        np.zeros((3, 64, 64), dtype=np.uint8),
        sensors,
        np.ones(5, dtype=bool),
        "crafter-v1",
        0,
    )
    case = build_case(
        {
            "uid": f"{family}-0",
            "family": family,
            "ruleset": "default",
            "seed": 13,
            "split": "train",
            "max_steps": 32,
        }
    )

    goal = resolve_goal(case, initial)

    assert goal.image is None
    assert goal.ranges == {sensor_index: (initial_value + 1.0, 9.0)}


def test_saturated_recovery_goal_is_rejected_as_unreachable() -> None:
    initial = Observation(
        np.zeros((3, 64, 64), dtype=np.uint8),
        np.array([9.0, 9.0, 9.0, 9.0, 0.0], dtype=np.float32),
        np.ones(5, dtype=bool),
        "crafter-v1",
        0,
    )
    case = build_case(
        {
            "uid": "recovery-saturated",
            "family": "resource_recovery",
            "ruleset": "default",
            "seed": 17,
            "split": "train",
            "max_steps": 32,
        }
    )

    with pytest.raises(ValueError, match="unreachable"):
        resolve_goal(case, initial)


def test_crafter_adapter_declares_native_actions_and_reset_accounting() -> None:
    adapter = CrafterCoreAdapter(max_steps=8)
    try:
        initial = adapter.reset(seed=19)
        assert adapter.actions.schema == "crafter-v1"
        assert adapter.actions.names[0] == "noop"
        assert initial.schema == "crafter-v1"
        assert adapter.reset_transitions == 1
    finally:
        adapter.close()
