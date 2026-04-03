"""Stage 62: NavigationPolicy — learned exploration via VSA+SDM.

Learns high-level navigation decisions from Bot demonstrations.
At decision points, predicts which direction to explore to find
the target object. Low-level BFS navigation handles execution.

Training: from Bot demo trajectories, extract decision points
(direction changes, door toggles, object interactions) and record
(abstract_state → direction_to_target) pairs in SDM.

Runtime: encode current abstract state → SDM read → predicted
direction → find nearest door/frontier in that direction → BFS.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from snks.agent.vsa_world_model import SDMMemory, VSACodebook


# 8 compass directions
DIRECTIONS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

# Direction vectors (row_delta, col_delta) — row increases downward
DIR_VECTORS = {
    "N": (-1, 0), "NE": (-1, 1), "E": (0, 1), "SE": (1, 1),
    "S": (1, 0), "SW": (1, -1), "W": (0, -1), "NW": (-1, -1),
}


def pos_to_direction(from_row: int, from_col: int,
                     to_row: int, to_col: int) -> str:
    """Compute compass direction from one position to another."""
    dr = to_row - from_row
    dc = to_col - from_col
    if dr == 0 and dc == 0:
        return "N"  # same position, arbitrary
    angle = math.atan2(dc, -dr)  # -dr because row increases downward
    # Quantize to 8 directions (each 45°)
    idx = round(angle / (math.pi / 4)) % 8
    return DIRECTIONS[idx]


def quadrant(row: int, col: int, height: int, width: int) -> int:
    """Map position to quadrant (0=NW, 1=NE, 2=SW, 3=SE)."""
    r = 0 if row < height // 2 else 2
    c = 0 if col < width // 2 else 1
    return r + c


class NavStateEncoder:
    """Encodes abstract navigation state as VSA vector."""

    def __init__(self, codebook: VSACodebook):
        self.cb = codebook

    def encode(self, agent_row: int, agent_col: int,
               grid_height: int, grid_width: int,
               target_type: str, target_color: str,
               explored_ratio: float,
               n_rooms_visited: int,
               nearest_door_dir: str | None) -> torch.Tensor:
        """Encode abstract navigation state into VSA vector."""
        facts = []

        # Agent quadrant
        q = quadrant(agent_row, agent_col, grid_height, grid_width)
        facts.append(VSACodebook.bind(
            self.cb.role("agent_quad"),
            self.cb.filler(f"quad_{q}")
        ))

        # Target type and color
        facts.append(VSACodebook.bind(
            self.cb.role("target_type"),
            self.cb.filler(f"ttype_{target_type}")
        ))
        facts.append(VSACodebook.bind(
            self.cb.role("target_color"),
            self.cb.filler(f"tcolor_{target_color}")
        ))

        # Exploration ratio (discretized)
        if explored_ratio < 0.25:
            exp_level = "low"
        elif explored_ratio < 0.60:
            exp_level = "mid"
        else:
            exp_level = "high"
        facts.append(VSACodebook.bind(
            self.cb.role("explored"),
            self.cb.filler(f"exp_{exp_level}")
        ))

        # Rooms visited (discretized)
        if n_rooms_visited <= 2:
            rooms_level = "few"
        elif n_rooms_visited <= 5:
            rooms_level = "some"
        else:
            rooms_level = "many"
        facts.append(VSACodebook.bind(
            self.cb.role("rooms"),
            self.cb.filler(f"rooms_{rooms_level}")
        ))

        # Nearest door direction
        if nearest_door_dir is not None:
            facts.append(VSACodebook.bind(
                self.cb.role("door_dir"),
                self.cb.filler(f"doordir_{nearest_door_dir}")
            ))

        return VSACodebook.bundle(facts)


class NavigationPolicy:
    """Learned high-level navigation policy via VSA+SDM.

    Predicts which direction to explore to find a target object,
    based on patterns learned from Bot demonstration trajectories.
    """

    def __init__(self, dim: int = 1024, n_locations: int = 5000,
                 seed: int = 200, n_amplify: int = 5,
                 device: torch.device | str | None = None):
        self.dim = dim
        self.n_amplify = n_amplify
        self.device = torch.device(device) if device else torch.device("cpu")

        self.codebook = VSACodebook(dim=dim, seed=seed, device=self.device)
        self.encoder = NavStateEncoder(self.codebook)

        # SDM: state_vec as address → direction_vec as content
        # (action slot = zeros, so address = bind(state, zeros) = state)
        self.sdm = SDMMemory(
            n_locations=n_locations, dim=dim,
            seed=seed + 1, device=self.device,
        )
        self._zeros = torch.zeros(dim, device=self.device)
        self.n_trained = 0
        self.confidence_threshold = 0.05

    def train_from_demos(self, demos: list[dict]) -> int:
        """Train navigation policy from Bot demo episodes.

        Extracts decision points from each demo and learns
        (abstract_state → direction_to_target) pairs.
        """
        n_points = 0
        for demo in demos:
            if not demo.get("success"):
                continue
            points = self._extract_decision_points(demo)
            for point in points:
                self._learn_point(point)
                n_points += 1

        self.n_trained = n_points
        return n_points

    def predict_direction(self, agent_row: int, agent_col: int,
                          grid_height: int, grid_width: int,
                          target_type: str, target_color: str,
                          explored_ratio: float,
                          n_rooms_visited: int,
                          nearest_door_dir: str | None
                          ) -> tuple[str | None, float]:
        """Predict best direction to explore.

        Returns (direction, confidence) or (None, 0) if SDM has no signal.
        """
        state_vec = self.encoder.encode(
            agent_row, agent_col, grid_height, grid_width,
            target_type, target_color,
            explored_ratio, n_rooms_visited, nearest_door_dir,
        )

        result_vec, confidence = self.sdm.read_next(state_vec, self._zeros)

        if confidence < self.confidence_threshold:
            return None, 0.0

        # Decode: find most similar direction filler
        best_dir = None
        best_sim = -1.0
        for d in DIRECTIONS:
            d_vec = self.codebook.filler(f"dir_{d}")
            sim = VSACodebook.similarity(result_vec, d_vec)
            if sim > best_sim:
                best_sim = sim
                best_dir = d

        return best_dir, confidence

    def _extract_decision_points(self, demo: dict) -> list[dict]:
        """Extract strategic decision points from a Bot demo.

        Decision points = frames where Bot changes direction, toggles door,
        or interacts with object. NOT every forward step.
        """
        frames = demo.get("frames", [])
        if not frames:
            return []

        # Find target position from subgoals
        # Use the first subgoal's object as target
        subgoals = demo.get("subgoals_extracted", [])
        target_type = subgoals[0].get("obj", "ball") if subgoals else "ball"
        target_color = subgoals[0].get("color", "") if subgoals else ""

        # Find actual target position in the demo grid
        # We need to scan the grid data — but demos don't store full grid
        # Instead, use the LAST frame position as approximate target
        # (Bot reaches target at the end)
        last_frame = frames[-1]
        target_col = last_frame.get("agent_col", 0)
        target_row = last_frame.get("agent_row", 0)

        grid_w = demo.get("grid_width", 22)
        grid_h = demo.get("grid_height", 22)

        points = []
        prev_action = None
        prev_dir = None
        rooms_seen = set()

        for i, frame in enumerate(frames):
            action = frame.get("action", "forward")
            agent_col = frame.get("agent_col", 0)
            agent_row = frame.get("agent_row", 0)
            agent_dir = frame.get("agent_dir", 0)

            # Track rooms (approximate: quadrant)
            rooms_seen.add(quadrant(agent_row, agent_col, grid_h, grid_w))

            is_decision = False
            # Direction change
            if prev_dir is not None and agent_dir != prev_dir:
                is_decision = True
            # Door toggle
            if action == "toggle":
                is_decision = True
            # Object interaction
            if action in ("pickup", "drop"):
                is_decision = True
            # First frame
            if i == 0:
                is_decision = True

            if is_decision:
                direction = pos_to_direction(
                    agent_row, agent_col, target_row, target_col
                )
                explored_ratio = min(i / 100, 1.0)  # approximate

                points.append({
                    "agent_row": agent_row,
                    "agent_col": agent_col,
                    "grid_height": grid_h,
                    "grid_width": grid_w,
                    "target_type": target_type,
                    "target_color": target_color,
                    "explored_ratio": explored_ratio,
                    "n_rooms_visited": len(rooms_seen),
                    "nearest_door_dir": None,  # not available from demo frames
                    "direction_to_target": direction,
                })

            prev_action = action
            prev_dir = agent_dir

        return points

    def _learn_point(self, point: dict) -> None:
        """Write one decision point to SDM."""
        state_vec = self.encoder.encode(
            point["agent_row"], point["agent_col"],
            point["grid_height"], point["grid_width"],
            point["target_type"], point["target_color"],
            point["explored_ratio"],
            point["n_rooms_visited"],
            point.get("nearest_door_dir"),
        )
        dir_vec = self.codebook.filler(f"dir_{point['direction_to_target']}")

        for _ in range(self.n_amplify):
            self.sdm.write(state_vec, self._zeros, dir_vec, 1.0)

    def get_stats(self) -> dict:
        return {
            "n_trained": self.n_trained,
            "sdm_writes": self.sdm.n_writes,
            "dim": self.dim,
            "n_locations": self.sdm.n_locations,
        }
