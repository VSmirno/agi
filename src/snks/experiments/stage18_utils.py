"""Shared utilities for Stage 18 experiments."""
from __future__ import annotations

import os

import numpy as np
import torch

from snks.agent.embodied_agent import EmbodiedAgent, EmbodiedAgentConfig
from snks.daf.types import (
    CausalAgentConfig,
    ConfiguratorConfig,
    ConsolidationConfig,
    CostModuleConfig,
    DafConfig,
    DcamConfig,
    EncoderConfig,
    HACPredictionConfig,
    HierarchicalConfig,
    PipelineConfig,
    ReplayConfig,
    SKSConfig,
)

# 15 envs: (env_id, difficulty, n_train_steps)
ENVS: list[tuple[str, str, int]] = [
    ("MiniGrid-Empty-5x5-v0",             "easy",   30_000),
    ("MiniGrid-Empty-8x8-v0",             "easy",   30_000),
    ("MiniGrid-FourRooms-v0",             "medium", 50_000),
    ("MiniGrid-DoorKey-5x5-v0",           "medium", 50_000),
    ("MiniGrid-DoorKey-8x8-v0",           "hard",   80_000),
    ("MiniGrid-MultiRoom-N2-S4-v0",       "medium", 50_000),
    ("MiniGrid-MultiRoom-N4-S5-v0",       "hard",   80_000),
    ("MiniGrid-LavaCrossingS9N1-v0",      "medium", 50_000),
    ("MiniGrid-LavaCrossingS9N2-v0",      "hard",   80_000),
    ("MiniGrid-SimpleCrossingS9N1-v0",    "medium", 50_000),
    ("MiniGrid-KeyCorridorS3R1-v0",       "medium", 50_000),
    ("MiniGrid-Unlock-v0",                "medium", 50_000),
    ("MiniGrid-UnlockPickup-v0",          "hard",   80_000),
    ("MiniGrid-MemoryS7-v0",              "hard",   80_000),
    ("MiniGrid-ObstructedMaze-1Dlhb-v0",  "hard",   80_000),
]

ENV_STEPS: dict[str, int] = {env_id: n for env_id, _, n in ENVS}
ENV_DIFFICULTY: dict[str, str] = {env_id: d for env_id, d, _ in ENVS}


def build_agent_config(device: str, N: int = 50_000) -> EmbodiedAgentConfig:
    """Build EmbodiedAgentConfig for Stage 18 experiments.

    Configuration: N=50K DAF nodes, disable_csr=True (required for AMD ROCm),
    steps_per_cycle=20 (performance-optimal for large N on GPU),
    replay_mode=uniform, replay_buffer=N=5000 (from exp39 validated config).

    Args:
        device: PyTorch device string, e.g. "cuda" or "cpu".
        N: Number of DAF nodes. Default 50_000.

    Returns:
        EmbodiedAgentConfig ready for construction.
    """
    daf = DafConfig(
        num_nodes=N,
        avg_degree=10,
        oscillator_model="fhn",
        dt=0.01,
        noise_sigma=0.01,
        fhn_I_base=0.0,
        device=device,
        disable_csr=True,  # required for large N on AMD ROCm
    )
    encoder = EncoderConfig(sdr_size=512, sdr_sparsity=0.04)
    sks = SKSConfig(coherence_mode="rate", min_cluster_size=5, dbscan_min_samples=5)
    pipeline = PipelineConfig(
        daf=daf,
        encoder=encoder,
        sks=sks,
        # steps_per_cycle=20: optimal for N=50K on AMD ROCm GPU (exp38 confirmed).
        # At 100 steps: ~119ms/cycle → 3.75 steps/sec (FAIL).
        # At 20 steps: ~24ms/cycle → 13+ steps/sec (PASS).
        steps_per_cycle=20,
        device=device,
        hierarchical=HierarchicalConfig(enabled=True),
        hac_prediction=HACPredictionConfig(enabled=True),
        cost_module=CostModuleConfig(enabled=True),
        configurator=ConfiguratorConfig(
            enabled=True,
            goal_cost_threshold=0.10,
            hysteresis_cycles=4,
        ),
    )
    dcam = DcamConfig(hac_dim=256, lsh_n_tables=8, lsh_n_bits=8, episodic_capacity=1000)
    causal = CausalAgentConfig(pipeline=pipeline, motor_sdr_size=80, dcam=dcam)
    consolidation = ConsolidationConfig(
        enabled=True,
        every_n=10,
        top_k=20,
        cold_threshold=0.3,
        node_threshold=0.7,
    )
    # exp39 validated: N=5000+uniform is the only config that passes all 3 env types.
    # top_k=5, n_steps=5 (n_steps=30 caused replay-attractor bias → coverage decrease).
    replay = ReplayConfig(enabled=True, top_k=5, n_steps=5, mode="uniform")
    return EmbodiedAgentConfig(
        causal=causal,
        consolidation=consolidation,
        replay=replay,
        use_stochastic_planner=True,
        n_plan_samples=8,
        max_plan_depth=5,
        goal_cost_value=1.0,
        plan_min_confidence=0.05,
    )


def make_env(env_id: str):
    """Create MiniGrid env with RGB observations.

    Args:
        env_id: MiniGrid environment ID registered in gymnasium.

    Returns:
        gymnasium.Env instance with max_episode_steps=500.
    """
    import minigrid  # registers MiniGrid envs into gymnasium  # noqa: F401
    import gymnasium
    return gymnasium.make(env_id, max_episode_steps=500)


def img(obs) -> np.ndarray:
    """Extract image array from observation dict.

    MiniGrid observations are dicts with an "image" key. This mirrors
    the _img() helper used in exp39/exp40 exactly.

    Args:
        obs: Raw observation from env.step() or env.reset().

    Returns:
        numpy array (H, W, C) uint8.
    """
    return obs["image"] if isinstance(obs, dict) else obs


def coverage_ratio(visited: set[tuple], env) -> float:
    """Fraction of walkable grid cells visited by the agent.

    Uses total grid area (width * height) as the denominator, which includes
    walls. This is consistent across all MiniGrid env types.

    Args:
        visited: Set of (x, y) position tuples visited during an episode.
        env: Gymnasium environment with unwrapped.width / unwrapped.height.

    Returns:
        Float in [0.0, 1.0].
    """
    total = env.unwrapped.width * env.unwrapped.height
    return len(visited) / total if total > 0 else 0.0


def checkpoint_path(exp: str, env_id: str, step: int | str) -> str:
    """Build a base path prefix for agent checkpoint files.

    The path is relative to the working directory. The caller (or
    EmbodiedAgent.save_checkpoint) is responsible for creating parent dirs.

    Args:
        exp: Experiment name, e.g. "exp41".
        env_id: MiniGrid environment ID.
        step: Step count (int) or "final" (str).

    Returns:
        Path string like "checkpoints/exp41/MiniGrid-Empty-5x5-v0/step_10000"
        or "checkpoints/exp41/MiniGrid-Empty-5x5-v0/final".

    Example:
        >>> checkpoint_path("exp41", "MiniGrid-Empty-5x5-v0", 10000)
        'checkpoints/exp41/MiniGrid-Empty-5x5-v0/step_10000'
        >>> checkpoint_path("exp41", "MiniGrid-Empty-5x5-v0", "final")
        'checkpoints/exp41/MiniGrid-Empty-5x5-v0/final'
    """
    safe_env_id = env_id.replace("/", "_")
    step_str = str(step) if isinstance(step, str) else f"step_{step}"
    return os.path.join("checkpoints", exp, safe_env_id, step_str)


def get_device() -> str:
    """Return 'cuda' if a CUDA/ROCm device is available, else 'cpu'.

    Returns:
        "cuda" or "cpu".
    """
    return "cuda" if torch.cuda.is_available() else "cpu"
