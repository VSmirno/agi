from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

import stage90r_eval_local_policy as eval_local_policy


def test_eval_episode_rng_is_deterministic_per_seed_and_episode():
    first = eval_local_policy._eval_episode_rng(base_seed=7, episode_index=0)
    second = eval_local_policy._eval_episode_rng(base_seed=7, episode_index=0)
    other_episode = eval_local_policy._eval_episode_rng(base_seed=7, episode_index=1)

    assert first.randint(0, 1_000_000, size=8).tolist() == second.randint(0, 1_000_000, size=8).tolist()
    assert first.randint(0, 1_000_000, size=8).tolist() != other_episode.randint(0, 1_000_000, size=8).tolist()
