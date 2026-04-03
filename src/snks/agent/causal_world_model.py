"""Stage 60: Causal World Model via Demonstrations.

Learns causal rules from synthetic demonstrations using VSA+SDM.
Each rule type gets its own SDM instance (per-rule SDM architecture).
Generalization via VSA identity property: bind(X,X) = zero_vector.

Components:
- RuleEncoder: encodes demonstrations into VSA rule vectors
- CausalWorldModel: learns rules, answers QA queries at 3 levels
  - QA-A: True/False facts (color matching)
  - QA-B: Precondition lookup (what is needed for action X?)
  - QA-C: Causal chains (backward chaining plans)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from snks.agent.vsa_world_model import SDMMemory, VSACodebook


@dataclass
class DemoStep:
    """One step of a synthetic demonstration."""
    state: dict[str, Any]
    action: str
    next_state: dict[str, Any]
    reward: float


# Rule type names
RULE_SAME_COLOR_UNLOCK = "same_color_unlock"
RULE_PICKUP_ADJACENT = "pickup_requires_adjacent"
RULE_DOOR_BLOCKS = "door_blocks_passage"
RULE_CARRYING_LIMITS = "carrying_limits"
RULE_OPEN_REQUIRES_KEY = "open_requires_key"
RULE_PUT_NEXT_TO = "put_next_to"
RULE_GO_TO_COMPLETE = "go_to_complete"

ALL_RULE_TYPES = [
    RULE_SAME_COLOR_UNLOCK,
    RULE_PICKUP_ADJACENT,
    RULE_DOOR_BLOCKS,
    RULE_CARRYING_LIMITS,
    RULE_OPEN_REQUIRES_KEY,
    RULE_PUT_NEXT_TO,
    RULE_GO_TO_COMPLETE,
]


class RuleEncoder:
    """Encodes causal rules as VSA vectors for SDM storage."""

    def __init__(self, codebook: VSACodebook):
        self.cb = codebook
        self._zeros = torch.zeros(codebook.dim, device=codebook.device)

    def encode_color_pair(self, key_color: str, door_color: str) -> torch.Tensor:
        """Encode a (key_color, door_color) pair via bind.

        bind(X, X) = zero_vector → all same-color pairs share the same address.
        bind(X, Y) = random vector → different-color pairs get unique addresses.
        """
        kv = self.cb.filler(f"color_{key_color}")
        dv = self.cb.filler(f"color_{door_color}")
        return VSACodebook.bind(kv, dv)

    def encode_color_rule(self, key_color: str, door_color: str,
                          reward: float) -> torch.Tensor:
        """Encode a same-color-unlock rule. Returns the SDM address."""
        return self.encode_color_pair(key_color, door_color)

    def encode_adjacency(self, is_adjacent: bool) -> torch.Tensor:
        """Encode adjacency condition."""
        return self.cb.filler("adjacent_true" if is_adjacent else "adjacent_false")

    def encode_has_key(self, has_key: bool) -> torch.Tensor:
        """Encode key possession."""
        return self.cb.filler("has_key_true" if has_key else "has_key_false")

    def encode_door_state(self, state: str) -> torch.Tensor:
        """Encode door state (locked/unlocked)."""
        return self.cb.filler(f"door_{state}")

    def encode_put_next_to(self, carrying: bool, adjacent: bool) -> torch.Tensor:
        """Encode PUT_NEXT_TO preconditions: carrying + adjacent to target."""
        carry_vec = self.cb.filler(f"carrying_{carrying}")
        adj_vec = self.cb.filler(f"put_adjacent_{adjacent}")
        return VSACodebook.bind(carry_vec, adj_vec)

    def encode_go_to_complete(self, adjacent: bool) -> torch.Tensor:
        """Encode GO_TO completion: adjacent to target = success."""
        return self.cb.filler(f"goto_adjacent_{adjacent}")


class CausalWorldModel:
    """Causal world model that learns rules from demonstrations.

    Architecture: per-rule SDM — one SDM instance per rule type.
    This avoids noisy unbinding from bundled vectors.
    """

    def __init__(self, dim: int = 512, n_locations: int = 1000,
                 seed: int = 42, n_amplify: int = 10,
                 device: torch.device | str | None = None):
        self.dim = dim
        self.n_locations = n_locations
        self.n_amplify = n_amplify
        self.device = torch.device(device) if device else torch.device("cpu")

        self.codebook = VSACodebook(dim=dim, seed=seed, device=self.device)
        self.encoder = RuleEncoder(self.codebook)

        # Per-rule SDMs
        self._rule_sdms: dict[str, SDMMemory] = {}
        for i, rule_type in enumerate(ALL_RULE_TYPES):
            self._rule_sdms[rule_type] = SDMMemory(
                n_locations=n_locations, dim=dim,
                seed=seed + i + 1, device=self.device,
            )

        self._zeros = torch.zeros(dim, device=self.device)

    def _sdm(self, rule_type: str) -> SDMMemory:
        return self._rule_sdms[rule_type]

    # ── Learning from demonstrations ──

    def learn_color_rules(self, colors: list[str]) -> None:
        """Learn same-color-unlock rules from color demos.

        For each pair (key_color, door_color):
          - same color → reward +1
          - different color → reward -1
        """
        sdm = self._sdm(RULE_SAME_COLOR_UNLOCK)
        for kc in colors:
            for dc in colors:
                address = self.encoder.encode_color_pair(kc, dc)
                reward = 1.0 if kc == dc else -1.0
                for _ in range(self.n_amplify):
                    sdm.write(address, self._zeros, address, reward)

    def learn_pickup_rules(self) -> None:
        """Learn pickup-requires-adjacent rule."""
        sdm = self._sdm(RULE_PICKUP_ADJACENT)
        addr_adj = self.encoder.encode_adjacency(True)
        addr_not_adj = self.encoder.encode_adjacency(False)
        for _ in range(self.n_amplify):
            sdm.write(addr_adj, self._zeros, addr_adj, 1.0)
            sdm.write(addr_not_adj, self._zeros, addr_not_adj, -1.0)

    def learn_door_blocks_rules(self) -> None:
        """Learn door-blocks-passage rule."""
        sdm = self._sdm(RULE_DOOR_BLOCKS)
        addr_locked = self.encoder.encode_door_state("locked")
        addr_unlocked = self.encoder.encode_door_state("unlocked")
        for _ in range(self.n_amplify):
            sdm.write(addr_locked, self._zeros, addr_locked, -1.0)
            sdm.write(addr_unlocked, self._zeros, addr_unlocked, 1.0)

    def learn_carrying_limits_rules(self) -> None:
        """Learn carrying-limits rule (can carry only one object)."""
        sdm = self._sdm(RULE_CARRYING_LIMITS)
        addr_has = self.encoder.encode_has_key(True)
        addr_empty = self.encoder.encode_has_key(False)
        for _ in range(self.n_amplify):
            # Trying to pickup while carrying → fail
            sdm.write(addr_has, self._zeros, addr_has, -1.0)
            # Pickup while empty → success
            sdm.write(addr_empty, self._zeros, addr_empty, 1.0)

    def learn_open_requires_key_rules(self) -> None:
        """Learn open-requires-key rule."""
        sdm = self._sdm(RULE_OPEN_REQUIRES_KEY)
        addr_has = self.encoder.encode_has_key(True)
        addr_no = self.encoder.encode_has_key(False)
        for _ in range(self.n_amplify):
            sdm.write(addr_has, self._zeros, addr_has, 1.0)
            sdm.write(addr_no, self._zeros, addr_no, -1.0)

    def learn_put_next_to_rules(self) -> None:
        """Learn put-next-to rules: need carrying + adjacent to target."""
        sdm = self._sdm(RULE_PUT_NEXT_TO)
        # Success: carrying AND adjacent
        addr_ok = self.encoder.encode_put_next_to(carrying=True, adjacent=True)
        # Fail cases
        addr_no_carry = self.encoder.encode_put_next_to(carrying=False, adjacent=True)
        addr_no_adj = self.encoder.encode_put_next_to(carrying=True, adjacent=False)
        for _ in range(self.n_amplify):
            sdm.write(addr_ok, self._zeros, addr_ok, 1.0)
            sdm.write(addr_no_carry, self._zeros, addr_no_carry, -1.0)
            sdm.write(addr_no_adj, self._zeros, addr_no_adj, -1.0)

    def learn_go_to_complete_rules(self) -> None:
        """Learn go-to completion: adjacent = success."""
        sdm = self._sdm(RULE_GO_TO_COMPLETE)
        addr_adj = self.encoder.encode_go_to_complete(adjacent=True)
        addr_not_adj = self.encoder.encode_go_to_complete(adjacent=False)
        for _ in range(self.n_amplify):
            sdm.write(addr_adj, self._zeros, addr_adj, 1.0)
            sdm.write(addr_not_adj, self._zeros, addr_not_adj, -1.0)

    def learn_all_rules(self, colors: list[str]) -> None:
        """Learn all 7 rule types."""
        self.learn_color_rules(colors)
        self.learn_pickup_rules()
        self.learn_door_blocks_rules()
        self.learn_carrying_limits_rules()
        self.learn_open_requires_key_rules()
        self.learn_put_next_to_rules()
        self.learn_go_to_complete_rules()

    # ── QA-A: True/False queries ──

    def query_color_match(self, key_color: str, door_color: str) -> bool:
        """Does key_color key open door_color door?"""
        sdm = self._sdm(RULE_SAME_COLOR_UNLOCK)
        address = self.encoder.encode_color_pair(key_color, door_color)
        reward = sdm.read_reward(address, self._zeros)
        return reward > 0

    # ── QA-B: Precondition lookup ──

    def query_precondition(self, action: str, param: str | None) -> str:
        """What is needed to perform action?

        Returns the precondition as a string.
        """
        if action == "open" and param and param != "no_key":
            # What color key is needed to open a door of color=param?
            # Iterate candidate colors, find which gives positive reward
            candidates = self._get_known_colors()
            # Always include the query color itself — identity generalization
            # means unseen colors work via bind(X,X)=zero even if X wasn't in training
            if param not in candidates:
                candidates.append(param)
            best_color = None
            best_reward = -float("inf")
            for color in candidates:
                address = self.encoder.encode_color_pair(color, param)
                reward = self._sdm(RULE_SAME_COLOR_UNLOCK).read_reward(
                    address, self._zeros
                )
                if reward > best_reward:
                    best_reward = reward
                    best_color = color
            return best_color if best_color else "unknown"

        if action == "open" and param == "no_key":
            # Can we open without a key?
            sdm = self._sdm(RULE_OPEN_REQUIRES_KEY)
            addr_no = self.encoder.encode_has_key(False)
            reward = sdm.read_reward(addr_no, self._zeros)
            return "need_key" if reward < 0 else "no_key_needed"

        if action == "pickup":
            # What is needed to pickup?
            sdm = self._sdm(RULE_PICKUP_ADJACENT)
            addr_adj = self.encoder.encode_adjacency(True)
            reward = sdm.read_reward(addr_adj, self._zeros)
            return "adjacent" if reward > 0 else "unknown"

        if action == "forward":
            # Can we move through locked door?
            sdm = self._sdm(RULE_DOOR_BLOCKS)
            addr_locked = self.encoder.encode_door_state("locked")
            reward = sdm.read_reward(addr_locked, self._zeros)
            return "blocked" if reward < 0 else "passable"

        if action == "put_next_to":
            # Need carrying + adjacent
            sdm = self._sdm(RULE_PUT_NEXT_TO)
            addr = self.encoder.encode_put_next_to(carrying=True, adjacent=True)
            reward = sdm.read_reward(addr, self._zeros)
            return "carrying_and_adjacent" if reward > 0 else "unknown"

        if action == "go_to":
            # Complete when adjacent
            sdm = self._sdm(RULE_GO_TO_COMPLETE)
            addr = self.encoder.encode_go_to_complete(adjacent=True)
            reward = sdm.read_reward(addr, self._zeros)
            return "adjacent" if reward > 0 else "unknown"

        return "unknown"

    def query_can_act(self, action: str, has_key: bool = True,
                      carrying: bool = False, adjacent: bool = False) -> bool:
        """Can the agent perform this action given current state?"""
        if action == "open":
            sdm = self._sdm(RULE_OPEN_REQUIRES_KEY)
            addr = self.encoder.encode_has_key(has_key)
            reward = sdm.read_reward(addr, self._zeros)
            return reward > 0
        if action == "put_next_to":
            sdm = self._sdm(RULE_PUT_NEXT_TO)
            addr = self.encoder.encode_put_next_to(carrying=carrying, adjacent=adjacent)
            reward = sdm.read_reward(addr, self._zeros)
            return reward > 0
        if action == "go_to":
            sdm = self._sdm(RULE_GO_TO_COMPLETE)
            addr = self.encoder.encode_go_to_complete(adjacent=adjacent)
            reward = sdm.read_reward(addr, self._zeros)
            return reward > 0
        return True

    # ── QA-C: Causal chains ──

    def query_chain(self, goal: str, color: str = "red",
                    max_depth: int = 5) -> list[str]:
        """Build a causal chain (plan) to achieve a goal.

        Backward chaining: goal → find rule with matching effect →
        precondition becomes new subgoal → repeat.
        """
        if goal == "pass_locked_door":
            return self._chain_pass_locked_door(color, max_depth)
        return []

    def _chain_pass_locked_door(self, color: str,
                                max_depth: int) -> list[str]:
        """Backward chain for 'pass through a locked door of given color'."""
        chain: list[str] = []

        # Step 1: To pass through, door must be unlocked
        sdm_blocks = self._sdm(RULE_DOOR_BLOCKS)
        addr_unlocked = self.encoder.encode_door_state("unlocked")
        reward = sdm_blocks.read_reward(addr_unlocked, self._zeros)
        if reward <= 0:
            return ["cannot_plan"]

        # Step 2: To unlock door, need same-color key
        sdm_unlock = self._sdm(RULE_SAME_COLOR_UNLOCK)
        addr_same = self.encoder.encode_color_pair(color, color)
        reward = sdm_unlock.read_reward(addr_same, self._zeros)
        if reward <= 0:
            return ["cannot_plan"]

        # Step 3: To use key, must have it (open_requires_key)
        sdm_req = self._sdm(RULE_OPEN_REQUIRES_KEY)
        addr_has = self.encoder.encode_has_key(True)
        reward = sdm_req.read_reward(addr_has, self._zeros)
        if reward <= 0:
            return ["cannot_plan"]

        # Step 4: To have key, must pickup (pickup_requires_adjacent)
        sdm_pickup = self._sdm(RULE_PICKUP_ADJACENT)
        addr_adj = self.encoder.encode_adjacency(True)
        reward = sdm_pickup.read_reward(addr_adj, self._zeros)
        if reward <= 0:
            return ["cannot_plan"]

        # All preconditions verified via SDM — build chain
        chain = ["find_key", "pickup_key", "open_door", "pass_through"]
        return chain

    # ── Helpers ──

    def _get_known_colors(self) -> list[str]:
        """Return colors that have been registered in the codebook."""
        colors = []
        for name in self.codebook._fillers:
            if name.startswith("color_"):
                colors.append(name[len("color_"):])
        return colors

    def get_stats(self) -> dict[str, int]:
        """Return write counts per rule SDM."""
        return {
            rule_type: sdm.n_writes
            for rule_type, sdm in self._rule_sdms.items()
        }
