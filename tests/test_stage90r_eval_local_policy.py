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


def test_trace_tail_keeps_terminal_steps_without_mutating_head_excerpt():
    trace = [{"step": step} for step in range(6)]

    assert eval_local_policy._trace_tail(trace, 3) == [{"step": 3}, {"step": 4}, {"step": 5}]
    assert trace[:2] == [{"step": 0}, {"step": 1}]


def test_trace_tail_allows_zero_to_disable_terminal_excerpt():
    assert eval_local_policy._trace_tail([{"step": 1}], 0) == []


def test_terminal_rescue_event_count_uses_last_n_steps_window():
    rescue_trace = [{"step": 5}, {"step": 15}, {"step": 18}, {"step": 19}]

    assert (
        eval_local_policy._terminal_rescue_event_count(
            rescue_trace,
            episode_steps=20,
            terminal_window=4,
        )
        == 2
    )


def test_terminal_rescue_event_count_disables_with_nonpositive_window():
    assert (
        eval_local_policy._terminal_rescue_event_count(
            [{"step": 9}],
            episode_steps=10,
            terminal_window=0,
        )
        == 0
    )


def test_is_hostile_death_only_counts_hostile_causes():
    assert eval_local_policy._is_hostile_death("zombie") is True
    assert eval_local_policy._is_hostile_death("arrow") is True
    assert eval_local_policy._is_hostile_death("dehydration") is False
