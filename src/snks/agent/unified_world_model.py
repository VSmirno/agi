"""Stage 62: Unified World Model — single SDM for all MiniGrid physics.

Replaces 7 hardcoded per-rule SDMs with one unified SDM that learns
ALL domain physics from state transitions. Can answer arbitrary questions
about the world and generate plans via forward chaining.

Architecture:
- Single VSACodebook (dim=1024) for encoding situations and outcomes
- Single SDMMemory (n_locations=10000) storing (situation → outcome) pairs
- Role-filler encoding: facts as bind(role, filler) pairs, bundled

Training: from synthetic transitions (exhaustive MiniGrid physics)
+ Bot demo transitions (real interaction patterns).

Inference:
- query(situation, action) → predicted outcome
- query_qa(question) → answer (preconditions, consequences, affordances)
- query_plan(goal, state) → action sequence via forward chaining
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from snks.agent.vsa_world_model import SDMMemory, VSACodebook
from snks.agent.world_model_trainer import Transition


class SituationEncoder:
    """Encodes situations and outcomes as VSA vectors."""

    def __init__(self, codebook: VSACodebook):
        self.cb = codebook

    def encode_situation(self, situation: dict[str, str],
                         action: str) -> torch.Tensor:
        """Encode a full situation (state + action) as VSA vector.

        Uses chained bind (not bundle) to preserve orthogonality.
        Bundle of 5+ items degrades to noise; chained bind stays sharp.
        """
        # Use COMPOUND fillers instead of chained bind.
        # Binary XOR loses structure after 3+ binds.
        # Instead: create unique filler per (obj+state, action, carry_state) combo.
        facing = situation.get('facing_obj', 'empty')
        obj_state = situation.get('obj_state', 'none')
        carrying = situation.get('carrying', 'nothing')
        obj_color = situation.get('obj_color', 'none')
        carry_color = situation.get('carrying_color', 'none')

        # Compound filler: unique per situation class
        compound_key = f"sit_{facing}_{obj_state}_{carrying}_{action}"
        vec = self.cb.filler(compound_key)

        # Bind with color info (max 2 binds: obj_color, carry_color)
        vec = VSACodebook.bind(vec, self.cb.filler(f"ocol_{obj_color}"))
        vec = VSACodebook.bind(vec, self.cb.filler(f"ccol_{carry_color}"))
        return vec

    def encode_outcome(self, outcome: dict[str, str]) -> torch.Tensor:
        """Encode an outcome as VSA vector.

        Uses the 'result' field as primary signal — it's the most
        discriminative part of the outcome.
        """
        result = outcome.get("result", "unknown")
        vec = self.cb.filler(f"res_{result}")
        # Chain additional outcome fields if present
        if "obj_state" in outcome:
            vec = VSACodebook.bind(vec, self.cb.filler(f"ost_{outcome['obj_state']}"))
        if "carrying" in outcome:
            vec = VSACodebook.bind(vec, self.cb.filler(f"carry_{outcome['carrying']}"))
        if "carrying_color" in outcome:
            vec = VSACodebook.bind(vec, self.cb.filler(f"ccol_{outcome['carrying_color']}"))
        return vec

    def decode_outcome(self, vec: torch.Tensor,
                       candidate_outcomes: list[dict[str, str]]
                       ) -> tuple[dict[str, str], float]:
        """Find most similar outcome from candidates."""
        best = {}
        best_sim = -1.0
        for outcome in candidate_outcomes:
            candidate_vec = self.encode_outcome(outcome)
            sim = VSACodebook.similarity(vec, candidate_vec)
            if sim > best_sim:
                best_sim = sim
                best = outcome
        return best, best_sim


# Canonical outcomes for decoding
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


class UnifiedWorldModel:
    """Single-SDM world model for all MiniGrid physics.

    All domain knowledge in one SDM. No hardcoded rule types.
    """

    def __init__(self, dim: int = 1024, n_locations: int = 10000,
                 seed: int = 300, n_amplify: int = 3,
                 device: torch.device | str | None = None):
        self.dim = dim
        self.n_amplify = n_amplify
        self.device = torch.device(device) if device else torch.device("cpu")

        self.codebook = VSACodebook(dim=dim, seed=seed, device=self.device)
        self.encoder = SituationEncoder(self.codebook)

        self.sdm = SDMMemory(
            n_locations=n_locations, dim=dim,
            seed=seed + 1, device=self.device,
        )
        self._zeros = torch.zeros(dim, device=self.device)
        self.n_trained = 0

    def train(self, transitions: list[Transition]) -> int:
        """Train on a list of transitions.

        Balances positive/negative transitions to avoid reward signal
        being dominated by failures (typical 10:1 neg/pos ratio in
        synthetic data).
        """
        pos = [t for t in transitions if t.reward > 0]
        neg = [t for t in transitions if t.reward < 0]
        zero = [t for t in transitions if t.reward == 0]

        # Amplify positives more, negatives less
        pos_amplify = self.n_amplify * 3
        neg_amplify = max(1, self.n_amplify // 2)
        zero_amplify = 1

        for t in pos:
            sit_vec = self.encoder.encode_situation(t.situation, t.action)
            out_vec = self.encoder.encode_outcome(t.outcome)
            for _ in range(pos_amplify):
                self.sdm.write(sit_vec, self._zeros, out_vec, t.reward)
            self.n_trained += 1

        for t in neg:
            sit_vec = self.encoder.encode_situation(t.situation, t.action)
            out_vec = self.encoder.encode_outcome(t.outcome)
            for _ in range(neg_amplify):
                self.sdm.write(sit_vec, self._zeros, out_vec, t.reward)
            self.n_trained += 1

        for t in zero:
            sit_vec = self.encoder.encode_situation(t.situation, t.action)
            out_vec = self.encoder.encode_outcome(t.outcome)
            for _ in range(zero_amplify):
                self.sdm.write(sit_vec, self._zeros, out_vec, t.reward)
            self.n_trained += 1

        return self.n_trained

    def query(self, situation: dict[str, str],
              action: str) -> tuple[dict[str, str], float]:
        """Predict outcome of action in situation.

        Returns (predicted_outcome, confidence).
        """
        sit_vec = self.encoder.encode_situation(situation, action)
        result_vec, confidence = self.sdm.read_next(sit_vec, self._zeros)

        if confidence < 0.01:
            return {"result": "unknown"}, 0.0

        outcome, sim = self.encoder.decode_outcome(result_vec, CANONICAL_OUTCOMES)
        return outcome, confidence

    def query_reward(self, situation: dict[str, str],
                     action: str) -> float:
        """Predict reward for action in situation.

        Positive = action succeeds, negative = action fails.
        """
        sit_vec = self.encoder.encode_situation(situation, action)
        return self.sdm.read_reward(sit_vec, self._zeros)

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
                        obj_color: str, obj_state: str = "none"
                        ) -> str:
        """Level 2: What do you need to <action> the <obj>?

        Returns the precondition as a string.
        """
        if action == "toggle" and obj_state == "locked":
            # Try each key color
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
            # Check: can pickup when not carrying?
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
        outcome, _ = self.query(situation, action)
        return outcome

    def qa_plan(self, goal: str, current_state: dict[str, str],
                max_steps: int = 6) -> list[dict]:
        """Level 4: Generate a plan to achieve goal.

        Forward chaining: at each step, try all actions,
        pick the one with highest reward, apply predicted outcome.
        """
        plan = []
        state = dict(current_state)

        for _ in range(max_steps):
            # Check if goal achieved
            if self._goal_achieved(goal, state):
                break

            # Try all relevant actions
            best_action = None
            best_reward = -float("inf")
            best_outcome = None

            for action in ["forward", "pickup", "drop", "toggle"]:
                reward = self.query_reward(state, action)
                if reward > best_reward:
                    best_reward = reward
                    best_action = action
                    out, _ = self.query(state, action)
                    best_outcome = out

            if best_action is None or best_reward <= -1.0:
                break

            plan.append({"action": best_action, "outcome": best_outcome,
                         "state": dict(state)})

            # Apply outcome to state
            if best_outcome:
                state = self._apply_outcome(state, best_outcome)

        return plan

    def _goal_achieved(self, goal: str, state: dict[str, str]) -> bool:
        """Check if goal is achieved in current state."""
        if goal == "open_door":
            return state.get("obj_state") == "open"
        if goal == "pickup_key":
            return state.get("carrying") == "key"
        if goal == "pickup_ball":
            return state.get("carrying") == "ball"
        if goal == "pickup_box":
            return state.get("carrying") == "box"
        if goal == "drop":
            return state.get("carrying") == "nothing"
        if goal.startswith("have_"):
            obj = goal[5:]  # "have_key" → "key"
            return state.get("carrying") == obj
        return False

    def _apply_outcome(self, state: dict[str, str],
                       outcome: dict[str, str]) -> dict[str, str]:
        """Apply predicted outcome to state."""
        new_state = dict(state)
        for key, value in outcome.items():
            if key == "result":
                continue  # meta field
            new_state[key] = value
        return new_state

    def get_stats(self) -> dict:
        return {
            "n_trained": self.n_trained,
            "sdm_writes": self.sdm.n_writes,
            "dim": self.dim,
            "n_locations": self.sdm.n_locations,
        }
