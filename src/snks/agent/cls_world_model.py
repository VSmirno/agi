"""Stage 62: CLS World Model — Complementary Learning System.

Bio-inspired two-tier world model:
- Hippocampus (SDM): fast one-shot learning, limited capacity, generalization
- Neocortex (dict): consolidated verified rules, unlimited, exact match

Consolidation loop: after training, replay patterns. If SDM predicts
correctly → promote to neocortex dict, free SDM capacity.

Query: neocortex first (exact), then hippocampus (fuzzy/generalization).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from snks.agent.vsa_world_model import SDMMemory, VSACodebook
from snks.agent.world_model_trainer import Transition


@dataclass
class Rule:
    """A consolidated world model rule."""
    situation_key: str
    outcome: dict[str, str]
    reward: float
    confidence: int = 0
    source: str = "synthetic"


# Canonical outcomes for SDM decoding
CANONICAL_OUTCOMES = [
    {"result": "moved"},
    {"result": "blocked"},
    {"result": "picked_up"},
    {"result": "failed_carrying"},
    {"result": "nothing_to_pickup"},
    {"result": "dropped"},
    {"result": "nothing_to_drop"},
    {"result": "drop_blocked"},
    {"result": "door_opened", "obj_state": "open"},
    {"result": "door_unlocked", "obj_state": "open"},
    {"result": "door_still_locked"},
    {"result": "door_closed", "obj_state": "closed"},
    {"result": "nothing_happened"},
]


def make_situation_key(situation: dict[str, str], action: str) -> str:
    """Build compound key from situation + action."""
    facing = situation.get("facing_obj", "empty")
    color = situation.get("obj_color", "none")
    state = situation.get("obj_state", "none")
    carrying = situation.get("carrying", "nothing")
    carry_color = situation.get("carrying_color", "none")
    return f"{facing}_{color}_{state}_{carrying}_{carry_color}_{action}"


class CLSWorldModel:
    """Complementary Learning System world model.

    Neocortex (dict) for verified rules. Hippocampus (SDM) for novel patterns.
    """

    def __init__(self, dim: int = 2048, n_locations: int = 5000,
                 seed: int = 400, n_amplify: int = 5,
                 consolidation_threshold: int = 3,
                 device: torch.device | str | None = None):
        self.dim = dim
        self.n_amplify = n_amplify
        self.consolidation_threshold = consolidation_threshold
        self.device = torch.device(device) if device else torch.device("cpu")

        # Hippocampus: fast, limited
        self.codebook = VSACodebook(dim=dim, seed=seed, device=self.device)
        self.hippocampus = SDMMemory(
            n_locations=n_locations, dim=dim,
            seed=seed + 1, device=self.device,
        )
        self._zeros = torch.zeros(dim, device=self.device)

        # Neocortex: exact, unlimited
        self.neocortex: dict[str, Rule] = {}

        # Stats
        self.n_sdm_writes = 0
        self.n_sdm_skipped = 0
        self.n_consolidated = 0

    def train(self, transitions: list[Transition]) -> dict:
        """Train from transitions: write-on-surprise to SDM, then consolidate."""
        # Phase 1: Write novel transitions to hippocampus
        for t in transitions:
            self._write_on_surprise(t)

        # Phase 2: Consolidate — promote verified rules to neocortex
        self._consolidate(transitions)

        return {
            "sdm_writes": self.n_sdm_writes,
            "sdm_skipped": self.n_sdm_skipped,
            "consolidated": self.n_consolidated,
            "neocortex_size": len(self.neocortex),
        }

    def _write_on_surprise(self, t: Transition) -> None:
        """Write to SDM only if the transition is surprising (not already known)."""
        key = make_situation_key(t.situation, t.action)

        # Already in neocortex? Skip.
        if key in self.neocortex:
            self.n_sdm_skipped += 1
            return

        # SDM already predicts correctly? Skip.
        sit_vec = self._encode_situation(t.situation, t.action)
        predicted, confidence = self.hippocampus.read_next(sit_vec, self._zeros)
        if confidence > 0.01:
            predicted_outcome = self._decode_outcome(predicted)
            if predicted_outcome.get("result") == t.outcome.get("result"):
                self.n_sdm_skipped += 1
                return

        # Novel! Write to hippocampus.
        out_vec = self._encode_outcome(t.outcome)
        for _ in range(self.n_amplify):
            self.hippocampus.write(sit_vec, self._zeros, out_vec, t.reward)
        self.n_sdm_writes += 1

    def _consolidate(self, transitions: list[Transition]) -> None:
        """Replay transitions and promote consistent SDM predictions to neocortex."""
        # Group transitions by situation key
        by_key: dict[str, Transition] = {}
        for t in transitions:
            key = make_situation_key(t.situation, t.action)
            by_key[key] = t  # last write wins (deterministic physics)

        # Test each unique situation
        for key, t in by_key.items():
            if key in self.neocortex:
                continue  # already consolidated

            sit_vec = self._encode_situation(t.situation, t.action)
            predicted, confidence = self.hippocampus.read_next(sit_vec, self._zeros)

            if confidence < 0.01:
                # SDM has no signal — store directly
                self.neocortex[key] = Rule(
                    situation_key=key,
                    outcome=t.outcome,
                    reward=t.reward,
                    confidence=1,
                    source="direct",
                )
                self.n_consolidated += 1
                continue

            predicted_outcome = self._decode_outcome(predicted)
            if predicted_outcome.get("result") == t.outcome.get("result"):
                # SDM predicted correctly — consolidate
                self.neocortex[key] = Rule(
                    situation_key=key,
                    outcome=t.outcome,
                    reward=t.reward,
                    confidence=self.consolidation_threshold,
                    source="consolidated",
                )
                self.n_consolidated += 1
            else:
                # SDM wrong — store ground truth directly
                self.neocortex[key] = Rule(
                    situation_key=key,
                    outcome=t.outcome,
                    reward=t.reward,
                    confidence=1,
                    source="direct_override",
                )
                self.n_consolidated += 1

    # ── Query ──

    def query(self, situation: dict[str, str],
              action: str) -> tuple[dict[str, str], float, str]:
        """Predict outcome. Returns (outcome, confidence, source).

        source: "neocortex" or "hippocampus"
        """
        key = make_situation_key(situation, action)

        # Neocortex first (exact match)
        if key in self.neocortex:
            rule = self.neocortex[key]
            return rule.outcome, 1.0, "neocortex"

        # Hippocampus (fuzzy/generalization)
        sit_vec = self._encode_situation(situation, action)
        predicted, confidence = self.hippocampus.read_next(sit_vec, self._zeros)
        if confidence > 0.01:
            outcome = self._decode_outcome(predicted)
            return outcome, confidence, "hippocampus"

        return {"result": "unknown"}, 0.0, "none"

    def query_reward(self, situation: dict[str, str], action: str) -> float:
        """Get reward for situation+action."""
        key = make_situation_key(situation, action)
        if key in self.neocortex:
            return self.neocortex[key].reward

        sit_vec = self._encode_situation(situation, action)
        return self.hippocampus.read_reward(sit_vec, self._zeros)

    # ── QA Methods ──

    def qa_can_interact(self, obj_type: str, action: str) -> bool:
        """Level 1: Can you do <action> with <obj_type>?"""
        situation = {
            "facing_obj": obj_type,
            "obj_color": "red",
            "obj_state": "none",
            "carrying": "nothing",
            "carrying_color": "",
        }
        if action == "toggle" and obj_type == "door":
            situation["obj_state"] = "closed"
        if action == "drop":
            situation["carrying"] = "ball"
            situation["carrying_color"] = "red"
            situation["facing_obj"] = "empty"

        reward = self.query_reward(situation, action)
        return reward > 0

    def qa_can_pass(self, obj_type: str, obj_state: str = "none") -> bool:
        """Level 1: Can you walk through <obj_type>?"""
        situation = {
            "facing_obj": obj_type,
            "obj_color": "grey",
            "obj_state": obj_state,
            "carrying": "nothing",
            "carrying_color": "",
        }
        reward = self.query_reward(situation, "forward")
        return reward >= 0

    def qa_precondition(self, action: str, obj_type: str,
                        obj_color: str, obj_state: str = "none") -> str:
        """Level 2: What do you need to <action> the <obj>?"""
        if action == "toggle" and obj_state == "locked":
            for key_color in ["red", "green", "blue", "purple", "yellow", "grey"]:
                situation = {
                    "facing_obj": "door",
                    "obj_color": obj_color,
                    "obj_state": "locked",
                    "carrying": "key",
                    "carrying_color": key_color,
                }
                reward = self.query_reward(situation, "toggle")
                if reward > 0:
                    return f"key_{key_color}"
            return "matching_key"

        if action == "pickup":
            situation = {
                "facing_obj": obj_type,
                "obj_color": obj_color,
                "obj_state": "none",
                "carrying": "nothing",
                "carrying_color": "",
            }
            reward = self.query_reward(situation, "pickup")
            if reward > 0:
                return "adjacent_and_empty_hands"
            return "cannot_pickup"

        if action == "drop":
            return "must_be_carrying"

        return "unknown"

    def qa_consequence(self, situation: dict[str, str],
                       action: str) -> dict[str, str]:
        """Level 3: What happens if you <action> in <situation>?"""
        outcome, _, _ = self.query(situation, action)
        return outcome

    def qa_plan(self, goal: str, current_state: dict[str, str],
                max_steps: int = 6) -> list[dict]:
        """Level 4: Plan to achieve goal via forward chaining through neocortex."""
        plan = []
        state = dict(current_state)

        for _ in range(max_steps):
            if self._goal_achieved(goal, state):
                break

            best_action = None
            best_outcome = None
            best_reward = -float("inf")

            for action in ["pickup", "toggle", "drop", "forward"]:
                reward = self.query_reward(state, action)
                if reward > best_reward:
                    best_reward = reward
                    best_action = action
                    out, _, _ = self.query(state, action)
                    best_outcome = out

            if best_action is None or best_reward <= -1.0:
                break

            plan.append({"action": best_action, "outcome": best_outcome})
            state = self._apply_outcome(state, best_outcome)

        return plan

    # ── Internals ──

    def _encode_situation(self, situation: dict[str, str],
                          action: str) -> torch.Tensor:
        """Encode situation for SDM. Compound filler + 2 color binds."""
        facing = situation.get("facing_obj", "empty")
        obj_state = situation.get("obj_state", "none")
        carrying = situation.get("carrying", "nothing")
        obj_color = situation.get("obj_color", "none")
        carry_color = situation.get("carrying_color", "none")

        compound = f"sit_{facing}_{obj_state}_{carrying}_{action}"
        vec = self.codebook.filler(compound)
        vec = VSACodebook.bind(vec, self.codebook.filler(f"ocol_{obj_color}"))
        vec = VSACodebook.bind(vec, self.codebook.filler(f"ccol_{carry_color}"))
        return vec

    def _encode_outcome(self, outcome: dict[str, str]) -> torch.Tensor:
        """Encode outcome for SDM storage."""
        result = outcome.get("result", "unknown")
        return self.codebook.filler(f"res_{result}")

    def _decode_outcome(self, vec: torch.Tensor) -> dict[str, str]:
        """Decode SDM output to nearest canonical outcome."""
        best = {"result": "unknown"}
        best_sim = -1.0
        for outcome in CANONICAL_OUTCOMES:
            candidate = self._encode_outcome(outcome)
            sim = VSACodebook.similarity(vec, candidate)
            if sim > best_sim:
                best_sim = sim
                best = outcome
        return best

    def _goal_achieved(self, goal: str, state: dict[str, str]) -> bool:
        if goal == "open_door":
            return state.get("obj_state") == "open"
        if goal.startswith("have_"):
            return state.get("carrying") == goal[5:]
        if goal == "drop":
            return state.get("carrying") == "nothing"
        return False

    def _apply_outcome(self, state: dict[str, str],
                       outcome: dict[str, str]) -> dict[str, str]:
        """Apply outcome to state, inferring implicit changes."""
        new = dict(state)
        result = outcome.get("result", "")

        # Explicit fields
        for k, v in outcome.items():
            if k != "result":
                new[k] = v

        # Implicit state changes based on result
        if result == "picked_up":
            # Now carrying the object we were facing
            new["carrying"] = state.get("facing_obj", "nothing")
            new["carrying_color"] = state.get("obj_color", "")
            new["facing_obj"] = "empty"
            new["obj_color"] = ""
            new["obj_state"] = "none"
        elif result == "dropped":
            new["carrying"] = "nothing"
            new["carrying_color"] = ""
        elif result in ("door_opened", "door_unlocked"):
            new["obj_state"] = "open"
            if result == "door_unlocked":
                # Key consumed when unlocking
                new["carrying"] = "nothing"
                new["carrying_color"] = ""
        elif result == "moved":
            # After moving, we face empty (simplification)
            new["facing_obj"] = "empty"
            new["obj_color"] = ""
            new["obj_state"] = "none"

        return new

    def get_stats(self) -> dict:
        return {
            "neocortex_size": len(self.neocortex),
            "sdm_writes": self.n_sdm_writes,
            "sdm_skipped": self.n_sdm_skipped,
            "consolidated": self.n_consolidated,
        }
