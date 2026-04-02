"""Stage 45: VSA World Model — Vector Symbolic Architecture + Sparse Distributed Memory.

Components:
- VSACodebook: Binary Spatter Code vectors (XOR bind, majority bundle)
- VSAEncoder: MiniGrid symbolic obs → structured VSA vector
- SDMMemory: Sparse Distributed Memory for transition storage
- CausalPlanner: forward beam search through SDM predictions (model-based planning)
- WorldModelAgent: Full agent — explore phase fills SDM, plan phase uses beam search
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class WorldModelConfig:
    dim: int = 512
    n_locations: int = 10000
    n_actions: int = 7
    min_confidence: float = 0.1
    epsilon: float = 0.1
    max_episode_steps: int = 200
    explore_episodes: int = 50     # random exploration before planning
    plan_depth: int = 8            # forward search depth
    beam_width: int = 5            # beam search width


class VSACodebook:
    """Binary Spatter Code codebook with lazy allocation."""

    def __init__(self, dim: int = 512, seed: int = 42):
        self.dim = dim
        self._rng = torch.Generator()
        self._rng.manual_seed(seed)
        self._roles: dict[str, torch.Tensor] = {}
        self._fillers: dict[str, torch.Tensor] = {}
        self._actions: dict[int, torch.Tensor] = {}
        # Pre-allocate reward vectors
        self._reward_positive = self._random_vec()
        self._reward_negative = self._random_vec()

    def _random_vec(self) -> torch.Tensor:
        return torch.randint(0, 2, (self.dim,), dtype=torch.float32, generator=self._rng)

    def role(self, name: str) -> torch.Tensor:
        if name not in self._roles:
            self._roles[name] = self._random_vec()
        return self._roles[name]

    def filler(self, name: str) -> torch.Tensor:
        if name not in self._fillers:
            self._fillers[name] = self._random_vec()
        return self._fillers[name]

    def action(self, idx: int) -> torch.Tensor:
        if idx not in self._actions:
            self._actions[idx] = self._random_vec()
        return self._actions[idx]

    @property
    def reward_positive(self) -> torch.Tensor:
        return self._reward_positive

    @property
    def reward_negative(self) -> torch.Tensor:
        return self._reward_negative

    @staticmethod
    def bind(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """XOR binding — self-inverse: bind(bind(a,b), b) = a."""
        return (a + b) % 2  # equivalent to XOR for binary {0,1}

    @staticmethod
    def bundle(vecs: list[torch.Tensor]) -> torch.Tensor:
        """Majority vote bundling."""
        stacked = torch.stack(vecs)
        summed = stacked.sum(dim=0)
        # Tie-breaking: random (use > for deterministic threshold)
        return (summed > len(vecs) / 2).float()

    @staticmethod
    def similarity(a: torch.Tensor, b: torch.Tensor) -> float:
        """Normalized similarity: 1 - normalized_hamming_distance."""
        matches = (a == b).float().mean()
        return matches.item()


class VSAEncoder:
    """Encodes MiniGrid symbolic observation (7x7x3) into VSA vector."""

    # MiniGrid object types
    OBJ_EMPTY = 1
    OBJ_WALL = 2
    OBJ_DOOR = 4
    OBJ_KEY = 5
    OBJ_GOAL = 8
    OBJ_AGENT = 10

    def __init__(self, codebook: VSACodebook):
        self.cb = codebook

    def encode(self, obs: np.ndarray) -> torch.Tensor:
        """Encode 7x7x3 symbolic observation into VSA vector.

        obs channels: [object_type, color, state]
        """
        facts: list[torch.Tensor] = []

        for r in range(obs.shape[0]):
            for c in range(obs.shape[1]):
                obj_type = int(obs[r, c, 0])
                color = int(obs[r, c, 1])
                state = int(obs[r, c, 2])

                if obj_type == self.OBJ_AGENT:
                    facts.append(self.cb.bind(
                        self.cb.role("agent_pos"),
                        self.cb.filler(f"pos_{r}_{c}"),
                    ))
                    facts.append(self.cb.bind(
                        self.cb.role("agent_dir"),
                        self.cb.filler(f"dir_{state}"),
                    ))
                elif obj_type == self.OBJ_KEY:
                    facts.append(self.cb.bind(
                        self.cb.role("key_pos"),
                        self.cb.filler(f"pos_{r}_{c}"),
                    ))
                    facts.append(self.cb.bind(
                        self.cb.role("key_color"),
                        self.cb.filler(f"color_{color}"),
                    ))
                elif obj_type == self.OBJ_DOOR:
                    facts.append(self.cb.bind(
                        self.cb.role("door_pos"),
                        self.cb.filler(f"pos_{r}_{c}"),
                    ))
                    door_state = "locked" if state == 2 else ("open" if state == 0 else "closed")
                    facts.append(self.cb.bind(
                        self.cb.role("door_state"),
                        self.cb.filler(door_state),
                    ))
                elif obj_type == self.OBJ_GOAL:
                    facts.append(self.cb.bind(
                        self.cb.role("goal_pos"),
                        self.cb.filler(f"pos_{r}_{c}"),
                    ))

        if not facts:
            return torch.zeros(self.cb.dim, dtype=torch.float32)

        return self.cb.bundle(facts)


class SDMMemory:
    """Sparse Distributed Memory for transition storage."""

    def __init__(self, n_locations: int = 10000, dim: int = 512, seed: int = 42):
        self.n_locations = n_locations
        self.dim = dim
        self.n_writes = 0

        rng = torch.Generator()
        rng.manual_seed(seed)

        # Hard location addresses (random, fixed)
        self.addresses = torch.randint(
            0, 2, (n_locations, dim), dtype=torch.float32, generator=rng,
        )
        # Content counters
        self.content_next = torch.zeros(n_locations, dim, dtype=torch.float32)
        self.content_reward = torch.zeros(n_locations, dim, dtype=torch.float32)

        # Calibrate activation radius
        self.activation_radius = self._calibrate_radius()

    def _calibrate_radius(self) -> int:
        """Find radius so 1-5% of locations activate on random query."""
        # Sample pairwise distances
        sample_idx = torch.randperm(min(self.n_locations, 1000))[:200]
        sample = self.addresses[sample_idx]
        # Compute distances between first 100 pairs
        dists = []
        for i in range(0, min(100, len(sample) - 1)):
            d = (sample[i] != sample[i + 1]).sum().item()
            dists.append(d)
        median_dist = sorted(dists)[len(dists) // 2]

        # Start at 0.45 * median, adjust if needed
        radius = int(median_dist * 0.45)
        for attempt in range(20):
            query = torch.randint(0, 2, (self.dim,), dtype=torch.float32)
            n_act = self._count_activated(query, radius)
            pct = n_act / self.n_locations
            if 0.005 <= pct <= 0.15:
                return radius
            if pct < 0.005:
                radius = int(radius * 1.1)
            else:
                radius = int(radius * 0.9)
        return radius

    def _count_activated(self, query: torch.Tensor, radius: int | None = None) -> int:
        if radius is None:
            radius = self.activation_radius
        dists = (self.addresses != query.unsqueeze(0)).sum(dim=1)
        return int((dists <= radius).sum().item())

    def _get_activated_mask(self, address: torch.Tensor) -> torch.Tensor:
        dists = (self.addresses != address.unsqueeze(0)).sum(dim=1)
        return dists <= self.activation_radius

    def write(self, state: torch.Tensor, action: torch.Tensor,
              next_state: torch.Tensor, reward: float) -> None:
        address = VSACodebook.bind(state, action)
        mask = self._get_activated_mask(address)

        # ±1 update for next state
        update_next = 2 * next_state - 1  # map {0,1} → {-1,+1}
        self.content_next[mask] += update_next.unsqueeze(0)

        # Reward encoding: store reward_positive or reward_negative pattern
        if reward > 0:
            update_r = torch.ones(self.dim, dtype=torch.float32)
        elif reward < 0:
            update_r = -torch.ones(self.dim, dtype=torch.float32)
        else:
            update_r = torch.zeros(self.dim, dtype=torch.float32)
        self.content_reward[mask] += update_r.unsqueeze(0)

        self.n_writes += 1

    def read_next(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, float]:
        address = VSACodebook.bind(state, action)
        mask = self._get_activated_mask(address)
        n_activated = mask.sum().item()

        if n_activated == 0:
            return torch.zeros(self.dim, dtype=torch.float32), 0.0

        summed = self.content_next[mask].sum(dim=0)
        predicted = (summed > 0).float()

        # Confidence: ratio of activated locations that have been written to
        # Use absolute sum magnitude as proxy for write count
        magnitude = summed.abs().mean().item()
        confidence = min(magnitude / 5.0, 1.0)  # normalize

        return predicted, confidence

    def read_reward(self, state: torch.Tensor, action: torch.Tensor) -> float:
        address = VSACodebook.bind(state, action)
        mask = self._get_activated_mask(address)
        n_activated = mask.sum().item()

        if n_activated == 0:
            return 0.0

        summed = self.content_reward[mask].sum(dim=0)
        # Net reward signal: positive sum = good, negative = bad
        return summed.mean().item()


class SDMPlanner:
    """1-step lookahead action selection via SDM reward prediction."""

    def __init__(self, sdm: SDMMemory, codebook: VSACodebook,
                 n_actions: int = 7, min_confidence: float = 0.1,
                 epsilon: float = 0.1):
        self.sdm = sdm
        self.cb = codebook
        self.n_actions = n_actions
        self.min_confidence = min_confidence
        self.epsilon = epsilon

    def select(self, state: torch.Tensor) -> int:
        scores = []
        confidences = []

        for a_idx in range(self.n_actions):
            action_vsa = self.cb.action(a_idx)
            reward_score = self.sdm.read_reward(state, action_vsa)
            _, confidence = self.sdm.read_next(state, action_vsa)
            scores.append(reward_score * confidence)
            confidences.append(confidence)

        if max(confidences) < self.min_confidence:
            return int(torch.randint(0, self.n_actions, (1,)).item())

        # Epsilon-greedy
        if self.epsilon > 0 and torch.rand(1).item() < self.epsilon:
            return int(torch.randint(0, self.n_actions, (1,)).item())

        return int(np.argmax(scores))


class CausalPlanner:
    """Forward beam search through SDM world model.

    Model-based planning: "imagine" action consequences N steps ahead via SDM,
    pick the action sequence whose terminal state is closest to goal.
    Falls back to random exploration when SDM confidence is too low.
    """

    def __init__(self, sdm: SDMMemory, codebook: VSACodebook,
                 n_actions: int = 7, plan_depth: int = 8,
                 beam_width: int = 5, min_confidence: float = 0.1,
                 epsilon: float = 0.1):
        self.sdm = sdm
        self.cb = codebook
        self.n_actions = n_actions
        self.plan_depth = plan_depth
        self.beam_width = beam_width
        self.min_confidence = min_confidence
        self.epsilon = epsilon
        self._goal_features: torch.Tensor | None = None

    def set_goal(self, goal_vsa: torch.Tensor) -> None:
        """Set goal features for similarity scoring."""
        self._goal_features = goal_vsa

    def select(self, state: torch.Tensor) -> int:
        """Select action via forward beam search through SDM."""
        if self._goal_features is None:
            return int(torch.randint(0, self.n_actions, (1,)).item())

        # Epsilon-greedy exploration
        if self.epsilon > 0 and torch.rand(1).item() < self.epsilon:
            return int(torch.randint(0, self.n_actions, (1,)).item())

        # Quick 1-step check: is any action directly rewarding?
        best_1step = -1
        best_1step_score = -float('inf')
        any_confident = False

        for a_idx in range(self.n_actions):
            action_vsa = self.cb.action(a_idx)
            reward = self.sdm.read_reward(state, action_vsa)
            _, conf = self.sdm.read_next(state, action_vsa)
            if conf >= 0.01:
                any_confident = True
                score = reward * conf
                if score > best_1step_score:
                    best_1step_score = score
                    best_1step = a_idx

        if not any_confident:
            return int(torch.randint(0, self.n_actions, (1,)).item())

        # If 1-step has strong signal, use it (fast path)
        if best_1step_score > 0.3:
            return best_1step

        # Beam search for deeper planning
        # beam = (predicted_state, first_action, score)
        beams: list[tuple[torch.Tensor, int, float]] = []
        goal = self._goal_features

        for a_idx in range(self.n_actions):
            action_vsa = self.cb.action(a_idx)
            pred_next, conf = self.sdm.read_next(state, action_vsa)
            if conf < 0.01:
                continue
            goal_sim = self.cb.similarity(pred_next, goal)
            reward = self.sdm.read_reward(state, action_vsa)
            score = (goal_sim + reward * 0.5) * conf
            beams.append((pred_next, a_idx, score))

        if not beams:
            return best_1step if best_1step >= 0 else int(torch.randint(0, self.n_actions, (1,)).item())

        for depth in range(1, self.plan_depth):
            new_beams: list[tuple[torch.Tensor, int, float]] = []
            for beam_state, first_action, cum_score in beams:
                for a_idx in range(self.n_actions):
                    action_vsa = self.cb.action(a_idx)
                    pred_next, conf = self.sdm.read_next(beam_state, action_vsa)
                    if conf < 0.01:
                        continue
                    goal_sim = self.cb.similarity(pred_next, goal)
                    score = (goal_sim + self.sdm.read_reward(beam_state, action_vsa) * 0.5) * conf
                    new_beams.append((pred_next, first_action, max(score, cum_score)))

            if not new_beams:
                break
            new_beams.sort(key=lambda b: b[2], reverse=True)
            beams = new_beams[:self.beam_width]

        best = max(beams, key=lambda b: b[2])
        return best[1]


class WorldModelAgent:
    """VSA World Model agent: explore → build model → plan → act.

    Two-phase operation:
    1. Explore phase (first N episodes): random actions, fill SDM with transitions
    2. Plan phase: forward beam search through SDM predictions toward goal

    The agent learns a causal model of the world (state+action→next_state)
    and uses it for model-based planning — no reward shaping needed.
    """

    def __init__(self, config: WorldModelConfig):
        self.config = config
        self.codebook = VSACodebook(dim=config.dim)
        self.encoder = VSAEncoder(self.codebook)
        self.sdm = SDMMemory(
            n_locations=config.n_locations,
            dim=config.dim,
        )
        self.causal_planner = CausalPlanner(
            sdm=self.sdm,
            codebook=self.codebook,
            n_actions=config.n_actions,
            plan_depth=config.plan_depth,
            beam_width=config.beam_width,
            min_confidence=config.min_confidence,
            epsilon=config.epsilon,
        )
        # Keep SDMPlanner for backward compat with tests
        self.planner = self.causal_planner
        self._prev_state: torch.Tensor | None = None
        self._prev_action: int | None = None
        self._episode_count: int = 0
        self._exploring: bool = True

    def set_goal_from_obs(self, obs: np.ndarray) -> None:
        """Extract goal features from observation and set planner goal."""
        # Encode a "goal state" — what would the world look like if we were at the goal?
        # We extract goal position from obs and create a synthetic goal state
        goal_facts: list[torch.Tensor] = []
        for r in range(obs.shape[0]):
            for c in range(obs.shape[1]):
                if int(obs[r, c, 0]) == VSAEncoder.OBJ_GOAL:
                    # Goal state = agent at goal position
                    goal_facts.append(self.codebook.bind(
                        self.codebook.role("agent_pos"),
                        self.codebook.filler(f"pos_{r}_{c}"),
                    ))
                    # Include goal presence
                    goal_facts.append(self.codebook.bind(
                        self.codebook.role("goal_pos"),
                        self.codebook.filler(f"pos_{r}_{c}"),
                    ))
                    # Door should be open in goal state
                    goal_facts.append(self.codebook.bind(
                        self.codebook.role("door_state"),
                        self.codebook.filler("open"),
                    ))
        if goal_facts:
            goal_vsa = self.codebook.bundle(goal_facts)
            self.causal_planner.set_goal(goal_vsa)

    def step(self, obs: np.ndarray) -> int:
        state = self.encoder.encode(obs)

        if self._exploring:
            action = int(torch.randint(0, self.config.n_actions, (1,)).item())
        else:
            action = self.causal_planner.select(state)

        self._prev_state = state
        self._prev_action = action
        return action

    def observe(self, obs: np.ndarray, reward: float) -> None:
        if self._prev_state is None:
            return
        next_state = self.encoder.encode(obs)
        action_vsa = self.codebook.action(self._prev_action)
        self.sdm.write(self._prev_state, action_vsa, next_state, reward)

    def run_episode(self, env, max_steps: int = 200) -> tuple[bool, int, float]:
        obs = env.reset()
        total_reward = 0.0
        self._prev_state = None
        self._prev_action = None

        # Set goal from initial observation
        if self._episode_count == 0 or self.causal_planner._goal_features is None:
            self.set_goal_from_obs(obs)

        # Check if we should switch from explore to plan
        self._exploring = self._episode_count < self.config.explore_episodes

        for step_i in range(max_steps):
            action = self.step(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            self.observe(obs, reward)
            total_reward += reward
            if terminated or truncated:
                self._episode_count += 1
                return total_reward > 0, step_i + 1, total_reward

        self._episode_count += 1
        return total_reward > 0, max_steps, total_reward
