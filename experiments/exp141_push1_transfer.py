"""Diagnostic wrapper for the bounded push_box/push_1 transfer probe."""

import sys

from snks.pipeline import core_experiment


_original_seed_lists = core_experiment._seed_lists
core_experiment.TRANSFER_TARGETS = (("push_box", "push_1"),)


def _seed_lists(seed: int, episodes: int, eval_episodes: int):
    previous_targets = core_experiment.TRANSFER_TARGETS
    try:
        core_experiment.TRANSFER_TARGETS = (("push_box", "push_2"),)
        seeds = _original_seed_lists(seed, episodes, eval_episodes)
    finally:
        core_experiment.TRANSFER_TARGETS = previous_targets
    return {
        **seeds,
        "transfer": {"push_box/push_1": seeds["transfer"]["push_box/push_2"]},
    }


core_experiment._seed_lists = _seed_lists


if __name__ == "__main__":
    raise SystemExit(core_experiment.main(sys.argv[1:]))
