"""Audit replay transition coverage for the fixed exp146 source protocol."""

from __future__ import annotations

import argparse
from collections import defaultdict
import sys
from pathlib import Path
from typing import Any

import numpy as np

from experiments.exp145_physics_transfer import SOURCE_LAYOUTS, _collect
from experiments.exp146_temporal_mpc_physics import ProgressJournal
from snks.env.core_types import Observation
from snks.pipeline import core_experiment as core


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _progress_interval(value: str) -> int:
    parsed = _positive(value)
    if parsed > 30:
        raise argparse.ArgumentTypeError("progress interval must not exceed 30 seconds")
    return parsed


def _observation_changes(before: Observation, after: Observation) -> dict[str, bool]:
    rgb = not np.array_equal(before.rgb, after.rgb)
    sensors = not np.array_equal(before.sensors, after.sensors)
    sensor_mask = not np.array_equal(before.sensor_mask, after.sensor_mask)
    return {"rgb": rgb, "sensors": sensors, "sensor_mask": sensor_mask,
            "exact": rgb or sensors or sensor_mask}


def _empty_action() -> dict[str, int]:
    return {"total": 0, "rgb_changed": 0, "rgb_no_change": 0,
            "exact_changed": 0, "exact_no_change": 0}


def _fraction(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _validate_terminal_counts(
    total: dict[str, int], fit: dict[str, int], *, episodes_per_layout: int
) -> bool:
    if episodes_per_layout != 512:
        return True
    return total == {"east_row2": 2, "west_row3": 7, "south_col4": 2, "north_col5": 7} and fit == {
        "east_row2": 2, "west_row3": 4, "south_col4": 2, "north_col5": 5
    }


def _audit_counts(episodes_by_layout: dict[str, list[Any]]) -> dict[str, Any]:
    by_layout: dict[str, Any] = {}
    overall = defaultdict(_empty_action)
    episodes_with_rgb_interact = total_rgb_interact = 0
    for layout_name, episodes in episodes_by_layout.items():
        actions = defaultdict(_empty_action)
        layout_rgb_interact_episodes = layout_rgb_interact = 0
        for episode in episodes:
            has_rgb_interact = False
            for transition in episode.transitions:
                changes = _observation_changes(transition.before, transition.after)
                for row in (actions[str(transition.action)], overall[str(transition.action)]):
                    row["total"] += 1
                    row["rgb_changed"] += int(changes["rgb"])
                    row["rgb_no_change"] += int(not changes["rgb"])
                    row["exact_changed"] += int(changes["exact"])
                    row["exact_no_change"] += int(not changes["exact"])
                if transition.action == 3 and changes["rgb"]:
                    has_rgb_interact = True
                    layout_rgb_interact += 1
            layout_rgb_interact_episodes += int(has_rgb_interact)
        by_layout[layout_name] = {"transitions": sum(item["total"] for item in actions.values()),
                                  "actions": dict(actions),
                                  "rgb_changing_interact_episodes": layout_rgb_interact_episodes,
                                  "rgb_changing_interact_transitions": layout_rgb_interact}
        episodes_with_rgb_interact += layout_rgb_interact_episodes
        total_rgb_interact += layout_rgb_interact
    total = sum(item["total"] for item in overall.values())
    for rows in (overall, *(value["actions"] for value in by_layout.values())):
        for row in rows.values():
            row["rgb_change_fraction"] = _fraction(row["rgb_changed"], row["total"])
            row["exact_change_fraction"] = _fraction(row["exact_changed"], row["total"])
    interact = overall["3"]
    def fractions(rows: dict[str, dict[str, int]], denominator: int) -> dict[str, Any]:
        return {action: {"rgb_changed": _fraction(row["rgb_changed"], denominator),
                         "exact_changed": _fraction(row["exact_changed"], denominator)}
                for action, row in rows.items()}

    return {"actions": dict(overall),
            "action_names": {"2": "forward", "3": "interact", "4": "noop"},
            "fractions": {"all_transitions": fractions(overall, total),
                "interact_actions": {
                    "rgb_changed": _fraction(interact["rgb_changed"], interact["total"]),
                    "exact_changed": _fraction(interact["exact_changed"], interact["total"])},
                "per_layout": {name: fractions(item["actions"], item["transitions"])
                    for name, item in by_layout.items()}},
            "episodes_with_rgb_changing_interact": episodes_with_rgb_interact,
            "rgb_changing_interact_transitions": total_rgb_interact,
            "by_layout": by_layout}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episodes-per-layout", type=_positive, default=512)
    parser.add_argument("--collection-steps", type=_positive, default=64)
    parser.add_argument("--progress-interval", type=_progress_interval, default=30)
    parser.add_argument("--collection-log-every", type=_positive, default=32)
    return parser


def _exit_code(error: BaseException) -> int:
    return 130 if isinstance(error, KeyboardInterrupt) else 1


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = {"argv": list(sys.orig_argv), "cwd": str(Path.cwd()),
                "git_head": core._git_commit(), "args": core._jsonable(vars(args))}
    total = len(SOURCE_LAYOUTS) * args.episodes_per_layout
    with ProgressJournal(args.out / "progress.jsonl", args.progress_interval) as journal:
        try:
            journal.update("collect", 0, total)
            episodes = {name: [] for name in SOURCE_LAYOUTS}
            completed = 0
            for offset in range(args.episodes_per_layout):
                for layout_index, (name, (layout, _actions)) in enumerate(SOURCE_LAYOUTS.items()):
                    seed = 10000 + layout_index * 100000 + offset
                    episodes[name].append(_collect(name, layout, seed, args.collection_steps))
                    completed += 1
                    if completed % args.collection_log_every == 0 or completed == total:
                        journal.update("collect", completed, total, layout=name, offset=offset)
            transitions = sum(len(ep.transitions) for items in episodes.values() for ep in items)
            terminals = {name: sum(bool(ep.transitions and ep.transitions[-1].terminated) for ep in items)
                         for name, items in episodes.items()}
            fixed = args.episodes_per_layout == 512 and args.collection_steps == 64
            fit_cutoff = round(0.75 * args.episodes_per_layout)
            fit_terminals = {name: sum(
                bool(ep.transitions and ep.transitions[-1].terminated)
                for ep in items[:fit_cutoff]
            ) for name, items in episodes.items()}
            if fixed and (transitions != 130676 or not _validate_terminal_counts(
                    terminals, fit_terminals, episodes_per_layout=args.episodes_per_layout)):
                raise AssertionError(f"scientific protocol mismatch: {transitions=}, {terminals=}, {fit_terminals=}")
            journal.update("artifacts", 0, 1, operation="write_results")
            result = {"status": "completed", "scientific_protocol": fixed,
                      "protocol": {"source_layouts_insertion_order": list(SOURCE_LAYOUTS),
                                   "episodes_per_layout": args.episodes_per_layout,
                                   "collection_steps": args.collection_steps, "interleaved_by_offset": True,
                                   "seed_scheme": "10000 + layout_index * 100000 + offset"},
                      "corpus": {"episodes": completed, "transitions": transitions,
                                 "natural_terminals_by_layout": terminals,
                                 "fit_cutoff_episodes_per_layout": fit_cutoff,
                                 "natural_terminals_fit_cutoff_by_layout": fit_terminals},
                      "coverage": _audit_counts(episodes),
                      "limitations": ["action 3 RGB change is a Push-domain proxy for a box interaction, not a generic semantic label.",
                                      "coverage counts are not used for training."]}
            core._write_json(args.out / "results.json", result)
            core._write_json(args.out / "manifest.json", {**manifest, "exit_code": 0, "exit_status": 0, "status": "completed"})
            journal.close(status="completed")
            return 0
        except BaseException as error:
            code = _exit_code(error)
            core._write_json(args.out / "manifest.json", {**manifest, "exit_code": code, "exit_status": code,
                             "status": "failed", "error": f"{type(error).__name__}: {error}"})
            raise


if __name__ == "__main__":
    raise SystemExit(main())
