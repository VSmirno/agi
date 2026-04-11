"""Stage 77a: Data types for forward simulation through ConceptStore.

Pure dataclasses — no behavior beyond trivial helpers. All forward-sim
logic lives in ConceptStore.simulate_forward and mpc_agent.

Design: docs/superpowers/specs/2026-04-10-stage77a-conceptstore-forward-sim-design.md
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from snks.agent.concept_store import CausalLink
    from snks.agent.crafter_spatial_map import CrafterSpatialMap


# ---------------------------------------------------------------------------
# Stateful condition (used inside RuleEffect for passive stateful rules)
# ---------------------------------------------------------------------------


@dataclass
class StatefulCondition:
    """Predicate on a body variable, evaluated per sim tick.

    Example: `food > 0` → StatefulCondition(var="food", op=">", threshold=0).
    """

    var: str
    op: str  # ">" | "<" | "==" | ">=" | "<="
    threshold: float

    def satisfied(self, sim: "SimState") -> bool:
        val = sim.body.get(self.var, 0.0)
        if self.op == ">":
            return val > self.threshold
        if self.op == "<":
            return val < self.threshold
        if self.op == "==":
            return val == self.threshold
        if self.op == ">=":
            return val >= self.threshold
        if self.op == "<=":
            return val <= self.threshold
        raise ValueError(f"unknown op: {self.op}")


# ---------------------------------------------------------------------------
# Rule effect — structured replacement for CausalLink.result: str
# ---------------------------------------------------------------------------


# Valid RuleEffect.kind values
EFFECT_KINDS = frozenset({
    "gather",       # inventory_delta (do X → get Y)
    "craft",        # inventory_delta (make X near Y → get X, lose requires)
    "place",        # world_place + inventory_delta (place X on empty)
    "remove",       # scene_remove (combat: do X → remove X from scene)
    "movement",     # movement_behavior (passive: entity movement pattern)
    "spatial",      # body_delta within spatial_range (passive: adjacent damage)
    "stateful",     # body_delta when stateful_condition holds (passive)
    "body_rate",    # body_delta per tick unconditional (background decay)
    "consume",      # body_delta (do X → body change, no inventory)
    "self",         # body_delta (sleep → energy)
})


@dataclass
class RuleEffect:
    """Structured effect of applying a causal rule on SimState.

    Replaces the old CausalLink.result: str. Effect is dispatchable directly
    by the simulator — no string parsing, no pseudo-nouns like "kill_zombie".

    Different `kind` values use different subsets of fields:
      - "gather" / "craft" / "consume" / "self": inventory_delta, body_delta
      - "place": world_place + inventory_delta
      - "remove": scene_remove
      - "movement": movement_behavior
      - "spatial": body_delta + spatial_range
      - "stateful": body_delta + stateful_condition
      - "body_rate": body_rate + body_rate_variable (per-tick unconditional)
    """

    kind: str  # one of EFFECT_KINDS

    # Inventory and body deltas (used by most kinds)
    inventory_delta: dict[str, int] = field(default_factory=dict)
    body_delta: dict[str, float] = field(default_factory=dict)

    # "remove" — concept_id of entity to remove from sim.dynamic_entities
    scene_remove: str | None = None

    # "place" — (item, where) e.g. ("table", "adjacent_empty")
    world_place: tuple[str, str] | None = None

    # "movement" — behavior string, e.g. "chase_player" | "flee_player" | "random_walk"
    movement_behavior: str | None = None

    # "spatial" — manhattan distance threshold, default 1 (adjacent)
    spatial_range: int = 1

    # "stateful" — condition that must hold for effect to fire
    stateful_condition: StatefulCondition | None = None

    # "body_rate" — per-tick unconditional delta (background decay)
    body_rate: float = 0.0
    body_rate_variable: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in EFFECT_KINDS:
            raise ValueError(
                f"RuleEffect.kind must be one of {sorted(EFFECT_KINDS)}, got {self.kind!r}"
            )


# ---------------------------------------------------------------------------
# Dynamic entities — things with positions that can move during simulation
# ---------------------------------------------------------------------------


@dataclass
class DynamicEntity:
    """Entity with a position that may move each sim tick.

    Neutral name ("entity", not "enemy") — hostility is a function of what
    rules apply to it, not of its type. A cow with `movement: random_walk` is
    a DynamicEntity just as much as a zombie with `movement: chase_player`.
    """

    concept_id: str
    pos: tuple[int, int]


# ---------------------------------------------------------------------------
# SimState — snapshot of the imagined world for one rollout tick
# ---------------------------------------------------------------------------


@dataclass
class SimState:
    """Snapshot of imagined world state at one sim tick.

    Deep-copied at the start of simulate_forward so the real world stays
    untouched. `spatial_map` is kept as a reference (not copied) — rollout
    reads but never mutates it. Everything else is owned by this SimState.
    """

    inventory: dict[str, int]
    body: dict[str, float]  # float for fractional rates (passive regen, decay)
    player_pos: tuple[int, int]
    dynamic_entities: list[DynamicEntity]
    spatial_map: Any  # CrafterSpatialMap, untyped to avoid circular import
    last_action: str | None
    step: int

    def copy(self) -> "SimState":
        """Deep-copy for starting a fresh rollout.

        Stage 81 (Bug 7): spatial_map is now COPIED (was shared by
        reference). Sim rollouts need to mutate the spatial_map
        locally — when sim fires a "do" rule that gathers a
        resource, that tile must be marked as empty in sim so
        subsequent rollout ticks don't oscillate trying to gather
        the same already-collected resource. Real planner state
        is unaffected (the copy is independent).

        spatial_map.copy() is supplied for objects with that method;
        otherwise the field is passed through (lightweight stubs in
        unit tests use None or simple dicts).
        """
        if self.spatial_map is not None and hasattr(self.spatial_map, "copy"):
            spatial_copy = self.spatial_map.copy()
        else:
            spatial_copy = self.spatial_map
        return SimState(
            inventory=dict(self.inventory),
            body=dict(self.body),
            player_pos=tuple(self.player_pos),
            dynamic_entities=[
                DynamicEntity(concept_id=e.concept_id, pos=tuple(e.pos))
                for e in self.dynamic_entities
            ],
            spatial_map=spatial_copy,
            last_action=self.last_action,
            step=self.step,
        )

    def is_dead(self, vital_mins: dict[str, float]) -> bool:
        """A vital body variable reached its catastrophic minimum → dead.

        `vital_mins` maps {vital_var_name: min_threshold}. Only variables in
        this dict cause death. Non-vital vars (food, drink, energy in Crafter)
        can be at 0 without terminating the episode — they trigger stateful
        damage to vital vars instead.

        Caller (simulate_forward) passes tracker.vital_mins.
        """
        for var, ref_min in vital_mins.items():
            if self.body.get(var, 0.0) <= ref_min:
                return True
        return False


# ---------------------------------------------------------------------------
# SimEvent — one logged event in a trajectory (for extract_failures)
# ---------------------------------------------------------------------------


# Valid SimEvent.kind values
EVENT_KINDS = frozenset({
    "body_delta",    # body variable changed by some amount
    "inv_gain",      # inventory item added/removed
    "rule_applied",  # a causal rule fired (action-triggered)
    "death",         # body var reached reference_min
    "entity_moved",  # dynamic entity position changed
    "entity_removed", # scene_remove fired
})


@dataclass
class SimEvent:
    """One significant event during a rollout tick.

    Used by extract_failures to attribute failures to sources (e.g., damage
    from zombie, not background decay) and by debug logging.
    """

    step: int
    kind: str  # one of EVENT_KINDS
    var: str | None  # body var name for body_delta
    amount: float  # signed delta
    source: str  # concept_id | "_background" | "stateful:<var>" | "rule:<kind>:<concept>"


# ---------------------------------------------------------------------------
# Plan and PlannedStep — what the agent intends to do
# ---------------------------------------------------------------------------


@dataclass
class PlannedStep:
    """One symbolic step in a plan.

    Expanded into primitive env actions by expand_to_primitive (mpc_agent).
    The `rule` reference is used for completion checking and for executing
    the rule's effect when the step fires.
    """

    action: str  # "do" | "make" | "place" | "sleep" | "inertia" | "move"
    target: str | None  # concept_id to interact with / navigate toward
    near: str | None  # for make/place: what must be adjacent
    rule: Any = None  # CausalLink | None — reference to the rule this step applies
    requires: dict = field(default_factory=dict)  # carried from rule.requires for convenience


@dataclass
class Plan:
    """Ordered sequence of PlannedSteps. Produced by plan_toward_rule or
    as a synthetic baseline (inertia) / explore (least-visited neighbor)."""

    steps: list[PlannedStep]
    origin: str = "unknown"  # "baseline" | "remedy" | "explore" — for debug/logging


# ---------------------------------------------------------------------------
# Trajectory — output of simulate_forward
# ---------------------------------------------------------------------------


@dataclass
class Trajectory:
    """Record of a rollout: per-tick body values, events, termination."""

    plan: Plan
    body_series: dict[str, list[float]]  # var → per-tick values
    events: list[SimEvent]
    final_state: SimState
    terminated: bool
    terminated_reason: str  # "body_dead" | "horizon" | "plan_complete"
    plan_progress: int  # number of PlannedSteps whose rule fired

    def failure_step(self, var: str) -> int | None:
        """First tick at which `var` became ≤ 0 (catastrophic).

        Returns None if the variable never reached zero in this trajectory.
        Used by extract_failures to attribute depletion events.
        """
        series = self.body_series.get(var, [])
        for i, value in enumerate(series):
            if value <= 0:
                return i
        return None

    def tick_count(self) -> int:
        """How many ticks were actually simulated.

        Empty body_series → 0. Assumes all vars have the same length
        (invariant maintained by simulate_forward).
        """
        for series in self.body_series.values():
            return len(series)
        return 0


# ---------------------------------------------------------------------------
# Failure — observation from baseline trajectory
# ---------------------------------------------------------------------------


# Valid Failure.kind values
FAILURE_KINDS = frozenset({
    "var_depleted",    # body var reached reference_min during rollout
    "attributed_to",   # negative body_delta attributed to a concept source
})


@dataclass
class Failure:
    """Observation from baseline rollout — what went wrong and when.

    Consumed by find_remedies to query the world model for counteracting
    rules. Drives emerge from failures, not from hardcoded categories.
    """

    kind: str  # one of FAILURE_KINDS
    var: str | None  # for var_depleted: which body var
    cause: str | None  # for attributed_to: concept_id of the damaging entity
    step: int  # when in the trajectory this failure was first observed
    severity: float = 1.0  # optional weight for prioritization

    def __post_init__(self) -> None:
        if self.kind not in FAILURE_KINDS:
            raise ValueError(
                f"Failure.kind must be one of {sorted(FAILURE_KINDS)}, got {self.kind!r}"
            )
