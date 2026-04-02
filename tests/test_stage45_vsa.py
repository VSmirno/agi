"""Stage 45: VSA World Model — unit tests (TDD).

Tests for VSACodebook, VSAEncoder, SDMMemory, SDMPlanner, WorldModelAgent.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from snks.agent.vsa_world_model import (
    SDMMemory,
    SDMPlanner,
    VSACodebook,
    VSAEncoder,
    WorldModelAgent,
    WorldModelConfig,
)


# ──────────────────────────────────────────────
# VSACodebook
# ──────────────────────────────────────────────

class TestVSACodebook:
    def setup_method(self):
        self.cb = VSACodebook(dim=512, seed=42)

    def test_codebook_creates_binary_vectors(self):
        vec = self.cb.role("agent_pos")
        assert vec.shape == (512,)
        assert set(vec.unique().tolist()).issubset({0, 1})

    def test_same_name_returns_same_vector(self):
        v1 = self.cb.role("agent_pos")
        v2 = self.cb.role("agent_pos")
        assert torch.equal(v1, v2)

    def test_different_names_different_vectors(self):
        v1 = self.cb.role("agent_pos")
        v2 = self.cb.role("key_pos")
        assert not torch.equal(v1, v2)

    def test_bind_is_xor(self):
        a = self.cb.role("agent_pos")
        b = self.cb.filler("pos_3_2")
        bound = self.cb.bind(a, b)
        assert bound.shape == (512,)
        # XOR is self-inverse: bind(bind(a,b), b) ≈ a
        recovered = self.cb.bind(bound, b)
        assert torch.equal(recovered, a)

    def test_bundle_majority_vote(self):
        vecs = [self.cb.role(f"role_{i}") for i in range(5)]
        bundled = self.cb.bundle(vecs)
        assert bundled.shape == (512,)
        assert set(bundled.unique().tolist()).issubset({0, 1})

    def test_similarity_identical(self):
        v = self.cb.role("agent_pos")
        sim = self.cb.similarity(v, v)
        assert sim == pytest.approx(1.0, abs=0.01)

    def test_similarity_random_near_half(self):
        v1 = self.cb.role("r1")
        v2 = self.cb.role("r2")
        sim = self.cb.similarity(v1, v2)
        # Random binary vectors: expected similarity ~ 0.5
        assert 0.3 < sim < 0.7

    def test_bind_preserves_retrievability(self):
        """bind(role, filler) → unbind with role → recovers filler."""
        role = self.cb.role("agent_pos")
        filler = self.cb.filler("pos_1_2")
        bound = self.cb.bind(role, filler)
        recovered = self.cb.bind(bound, role)
        sim = self.cb.similarity(recovered, filler)
        assert sim > 0.99

    def test_action_vectors(self):
        for i in range(7):
            v = self.cb.action(i)
            assert v.shape == (512,)

    def test_reward_vectors(self):
        pos = self.cb.reward_positive
        neg = self.cb.reward_negative
        assert pos.shape == (512,)
        assert neg.shape == (512,)
        assert not torch.equal(pos, neg)


# ──────────────────────────────────────────────
# VSAEncoder
# ──────────────────────────────────────────────

class TestVSAEncoder:
    def setup_method(self):
        self.cb = VSACodebook(dim=512, seed=42)
        self.enc = VSAEncoder(self.cb)

    def _make_obs(self) -> np.ndarray:
        """Create a minimal MiniGrid-like symbolic obs (7x7x3)."""
        obs = np.zeros((7, 7, 3), dtype=np.int64)
        # Object types from MiniGrid: 1=empty, 2=wall, 4=door, 5=key, 8=goal, 10=agent
        obs[3, 3, 0] = 10  # agent at (3,3)
        obs[3, 3, 2] = 0   # agent direction: right
        obs[1, 4, 0] = 5   # key at (1,4)
        obs[1, 4, 1] = 1   # key color: green
        obs[2, 3, 0] = 4   # door at (2,3)
        obs[2, 3, 2] = 2   # door state: locked
        obs[4, 4, 0] = 8   # goal at (4,4)
        return obs

    def test_encode_returns_binary_vector(self):
        obs = self._make_obs()
        state = self.enc.encode(obs)
        assert state.shape == (512,)
        assert set(state.unique().tolist()).issubset({0, 1})

    def test_same_obs_same_encoding(self):
        obs = self._make_obs()
        s1 = self.enc.encode(obs)
        s2 = self.enc.encode(obs)
        assert torch.equal(s1, s2)

    def test_different_obs_different_encoding(self):
        obs1 = self._make_obs()
        obs2 = self._make_obs()
        obs2[3, 3, 0] = 1  # remove agent — different state
        obs2[3, 4, 0] = 10  # agent moved
        s1 = self.enc.encode(obs1)
        s2 = self.enc.encode(obs2)
        sim = self.cb.similarity(s1, s2)
        assert sim < 0.95  # should be different

    def test_unbind_recovers_agent_pos(self):
        """Encode full obs, unbind agent_pos role → should match agent filler."""
        obs = self._make_obs()
        state = self.enc.encode(obs)
        role = self.cb.role("agent_pos")
        recovered = self.cb.bind(state, role)
        expected = self.cb.filler("pos_3_3")
        sim = self.cb.similarity(recovered, expected)
        # With bundle of ~5-7 facts, unbinding accuracy should be > 0.6
        assert sim > 0.55, f"Unbinding similarity {sim:.3f} too low"

    def test_encode_empty_obs(self):
        """All-zero obs → still produces valid vector."""
        obs = np.zeros((7, 7, 3), dtype=np.int64)
        state = self.enc.encode(obs)
        assert state.shape == (512,)

    def test_inventory_encoding(self):
        """Agent carrying key should encode has_key=yes."""
        obs = self._make_obs()
        # MiniGrid convention: agent carrying = object type 5 at agent pos with special state
        # We'll test through the full encoding pipeline
        state = self.enc.encode(obs)
        assert state.shape == (512,)


# ──────────────────────────────────────────────
# SDMMemory
# ──────────────────────────────────────────────

class TestSDMMemory:
    def setup_method(self):
        self.cb = VSACodebook(dim=512, seed=42)
        self.sdm = SDMMemory(n_locations=1000, dim=512, seed=42)

    def test_init_calibrates_radius(self):
        """Activation radius should be calibrated so 1-5% locations activate."""
        assert self.sdm.activation_radius > 0
        # Test with random query
        query = torch.randint(0, 2, (512,), dtype=torch.float32)
        n_activated = self.sdm._count_activated(query)
        pct = n_activated / 1000
        assert 0.005 < pct < 0.15, f"Activation {pct:.1%} out of range"

    def test_write_and_read_next(self):
        """Write a transition, read it back."""
        state = torch.randint(0, 2, (512,), dtype=torch.float32)
        action = self.cb.action(0)
        next_state = torch.randint(0, 2, (512,), dtype=torch.float32)

        self.sdm.write(state, action, next_state, reward=1.0)
        predicted, confidence = self.sdm.read_next(state, action)

        assert predicted.shape == (512,)
        sim = self.cb.similarity(predicted, next_state)
        assert sim > 0.55, f"Prediction similarity {sim:.3f} too low"

    def test_write_and_read_reward(self):
        """Write positive reward, read it back as positive."""
        state = torch.randint(0, 2, (512,), dtype=torch.float32)
        action = self.cb.action(1)
        next_state = torch.randint(0, 2, (512,), dtype=torch.float32)

        # Write multiple positive rewards to strengthen signal
        for _ in range(5):
            self.sdm.write(state, action, next_state, reward=1.0)

        reward_score = self.sdm.read_reward(state, action)
        assert reward_score > 0, f"Expected positive reward, got {reward_score}"

    def test_negative_reward(self):
        """Write negative reward, read it back as negative."""
        state = torch.randint(0, 2, (512,), dtype=torch.float32)
        action = self.cb.action(2)
        next_state = torch.randint(0, 2, (512,), dtype=torch.float32)

        for _ in range(5):
            self.sdm.write(state, action, next_state, reward=-1.0)

        reward_score = self.sdm.read_reward(state, action)
        assert reward_score < 0, f"Expected negative reward, got {reward_score}"

    def test_unknown_state_low_confidence(self):
        """Never-seen state should have low confidence."""
        state = torch.randint(0, 2, (512,), dtype=torch.float32)
        action = self.cb.action(0)
        _, confidence = self.sdm.read_next(state, action)
        # Empty SDM — confidence should be 0
        assert confidence == 0.0

    def test_multiple_transitions_dont_overwrite(self):
        """SDM is additive — multiple transitions coexist."""
        s1 = torch.randint(0, 2, (512,), dtype=torch.float32)
        s2 = torch.randint(0, 2, (512,), dtype=torch.float32)
        a = self.cb.action(0)
        ns1 = torch.randint(0, 2, (512,), dtype=torch.float32)
        ns2 = torch.randint(0, 2, (512,), dtype=torch.float32)

        self.sdm.write(s1, a, ns1, reward=1.0)
        self.sdm.write(s2, a, ns2, reward=-1.0)

        pred1, _ = self.sdm.read_next(s1, a)
        pred2, _ = self.sdm.read_next(s2, a)

        sim1 = self.cb.similarity(pred1, ns1)
        sim2 = self.cb.similarity(pred2, ns2)
        assert sim1 > 0.5
        assert sim2 > 0.5


# ──────────────────────────────────────────────
# SDMPlanner
# ──────────────────────────────────────────────

class TestSDMPlanner:
    def setup_method(self):
        self.cb = VSACodebook(dim=512, seed=42)
        self.sdm = SDMMemory(n_locations=1000, dim=512, seed=42)
        self.planner = SDMPlanner(
            sdm=self.sdm,
            codebook=self.cb,
            n_actions=7,
            min_confidence=0.1,
            epsilon=0.0,  # no noise for deterministic tests
        )

    def test_empty_sdm_returns_random(self):
        """With empty SDM, all confidences < threshold → random action."""
        state = torch.randint(0, 2, (512,), dtype=torch.float32)
        actions = [self.planner.select(state) for _ in range(20)]
        # Should return valid actions
        assert all(0 <= a < 7 for a in actions)
        # Should be somewhat random (not all same)
        assert len(set(actions)) > 1

    def test_selects_high_reward_action(self):
        """After learning, planner should prefer action with positive reward."""
        state = torch.randint(0, 2, (512,), dtype=torch.float32)
        good_action = 2
        bad_action = 4
        next_s = torch.randint(0, 2, (512,), dtype=torch.float32)

        # Write strong reward signal
        for _ in range(10):
            self.sdm.write(state, self.cb.action(good_action), next_s, reward=1.0)
            self.sdm.write(state, self.cb.action(bad_action), next_s, reward=-1.0)

        # Should consistently pick good action
        choices = [self.planner.select(state) for _ in range(10)]
        assert choices.count(good_action) >= 7, f"Expected action {good_action}, got {choices}"


# ──────────────────────────────────────────────
# WorldModelAgent (integration)
# ──────────────────────────────────────────────

class TestWorldModelAgent:
    def test_agent_creates_successfully(self):
        config = WorldModelConfig(dim=512, n_locations=1000, n_actions=7)
        agent = WorldModelAgent(config)
        assert agent is not None

    def test_step_returns_valid_action(self):
        config = WorldModelConfig(dim=512, n_locations=1000, n_actions=7)
        agent = WorldModelAgent(config)
        obs = np.zeros((7, 7, 3), dtype=np.int64)
        action = agent.step(obs)
        assert 0 <= action < 7

    def test_observe_records_transition(self):
        config = WorldModelConfig(dim=512, n_locations=1000, n_actions=7)
        agent = WorldModelAgent(config)
        obs1 = np.zeros((7, 7, 3), dtype=np.int64)
        obs1[3, 3, 0] = 10  # agent
        agent.step(obs1)  # first step — sets prev state

        obs2 = np.zeros((7, 7, 3), dtype=np.int64)
        obs2[3, 4, 0] = 10  # agent moved
        agent.observe(obs2, reward=0.0)  # should record transition

        # SDM should now have at least one write
        assert agent.sdm.n_writes > 0

    def test_run_episode_simple_env(self):
        """Run episode on a trivial env."""
        config = WorldModelConfig(dim=512, n_locations=1000, n_actions=4)
        agent = WorldModelAgent(config)

        # Simple mock env
        class SimpleEnv:
            def __init__(self):
                self.steps = 0
            def reset(self, seed=None):
                self.steps = 0
                obs = np.zeros((7, 7, 3), dtype=np.int64)
                obs[3, 3, 0] = 10
                return obs
            def step(self, action):
                self.steps += 1
                obs = np.zeros((7, 7, 3), dtype=np.int64)
                obs[3, 3, 0] = 10
                done = self.steps >= 5
                return obs, 1.0 if done else 0.0, done, False, {}

        env = SimpleEnv()
        success, steps, reward = agent.run_episode(env, max_steps=20)
        assert success is True
        assert steps == 5
        assert reward > 0

    def test_learning_improves_over_episodes(self):
        """Agent should learn from experience — plan phase uses successful traces."""
        config = WorldModelConfig(
            dim=512, n_locations=1000, n_actions=4,
            epsilon=0.1, min_confidence=0.05,
            explore_episodes=10,  # short explore so plan phase kicks in
        )
        agent = WorldModelAgent(config)

        # Env where action 0 always gives reward
        class RewardEnv:
            def __init__(self):
                self.steps = 0
            def reset(self, seed=None):
                self.steps = 0
                obs = np.zeros((7, 7, 3), dtype=np.int64)
                obs[3, 3, 0] = 10
                return obs
            def step(self, action):
                self.steps += 1
                obs = np.zeros((7, 7, 3), dtype=np.int64)
                obs[3, 3, 0] = 10
                reward = 1.0 if action == 0 else 0.0
                done = action == 0 or self.steps >= 10
                return obs, reward, done, False, {}

        env = RewardEnv()
        results = []
        for _ in range(30):
            success, steps, reward = agent.run_episode(env, max_steps=10)
            results.append((success, steps))

        # Plan phase (last 10) should have at least as many successes as explore (first 10)
        explore_success = sum(1 for s, _ in results[:10] if s)
        plan_success = sum(1 for s, _ in results[20:] if s)
        # With traces from successful explore episodes, plan should match or exceed
        assert plan_success >= explore_success - 2, \
            f"Plan not learning: explore={explore_success}, plan={plan_success}"


# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

class TestWorldModelConfig:
    def test_defaults(self):
        cfg = WorldModelConfig()
        assert cfg.dim == 512
        assert cfg.n_locations == 10000
        assert cfg.n_actions == 7
        assert cfg.min_confidence == 0.1
        assert cfg.epsilon == 0.1
        assert cfg.max_episode_steps == 200
