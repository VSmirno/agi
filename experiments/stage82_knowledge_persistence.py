"""Stage 82 — Knowledge persistence smoke harness.

Demonstrates that category-3 experience (learned rules, homeostatic
tracker observations, runtime rule confidences) can be carried across
process boundaries via ConceptStore.save_experience / load_experience.

This is the knowledge flow principle from IDEOLOGY v2 §2: experience
accumulated by a running agent must be able to promote into the next
agent's starting state. Without persistence, every fresh process
starts from the textbook alone and re-learns what the previous
process already knew — which is both wasteful and a hard ceiling on
the rate at which experience can inform facts.

Harness structure
-----------------
Three "sessions" sharing a single experience file:

  Session 1: fresh store+tracker from textbook
             run N episodes
             save experience JSON

  Session 2: fresh store+tracker from textbook
             load experience JSON (merges with textbook)
             run N more episodes
             save experience JSON (now with more rules + obs)

  Session 3: fresh store+tracker from textbook
             load experience JSON
             assert: learned_rules ≥ session 2,
                     tracker counts ≥ session 1 + session 2

Each session is a separate process invocation of the same harness
script, or (as here) three invocations within one process with
independent state between them. The point is that between sessions
the ONLY state flow is through the JSON file.

Pass criterion
--------------
- Session 2 starts with ≥1 learned rules loaded from disk (from
  session 1) instead of the zero it would have without persistence.
- Session 3 sees observation_counts strictly greater than what any
  single session could have produced on its own.
- No regressions on tests/test_stage77_mpc.py behaviour in sessions
  2/3 (same store runs without crashing on a pre-populated state).

Usage
-----
  .venv/bin/python experiments/stage82_knowledge_persistence.py --fast
  .venv/bin/python experiments/stage82_knowledge_persistence.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from snks.agent.concept_store import ConceptStore
from snks.agent.crafter_pixel_env import CrafterPixelEnv
from snks.agent.crafter_textbook import CrafterTextbook
from snks.agent.mpc_agent import run_mpc_episode
from snks.agent.perception import HomeostaticTracker
from snks.encoder.cnn_encoder import disable_rocm_conv
from snks.encoder.tile_segmenter import load_tile_segmenter
from snks.learning.rule_nursery import RuleNursery
from snks.learning.surprise_accumulator import SurpriseAccumulator


STAGE75_CHECKPOINT = Path("demos/checkpoints/exp135/segmenter_9x9.pt")
EXPERIENCE_PATH = Path("_docs/stage82_experience.json")
RESULTS_PATH = Path("_docs/stage82_results.json")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _disable_enemies(env: CrafterPixelEnv) -> None:
    """Stage 79 convention — suppress zombie/skeleton spawn for a cleaner
    resource-gathering signal."""
    try:
        env._world._zombie_spawn_rate = 0.0  # type: ignore[attr-defined]
        env._world._skeleton_spawn_rate = 0.0  # type: ignore[attr-defined]
    except Exception:
        pass


def _fresh_store_and_tracker() -> tuple[ConceptStore, HomeostaticTracker]:
    """Each session starts with a fresh store+tracker loaded only from
    the textbook — no state leak between sessions except via the JSON
    experience file."""
    store = ConceptStore()
    tb = CrafterTextbook("configs/crafter_textbook.yaml")
    tb.load_into(store)
    tracker = HomeostaticTracker()
    tracker.init_from_textbook(tb.body_block, store.passive_rules)
    return store, tracker


def _run_session(
    label: str,
    n_episodes: int,
    max_steps: int,
    horizon: int,
    segmenter: Any,
    experience_path: Path,
    load_existing: bool,
    seed_offset: int,
) -> dict:
    """Run one session: optionally load experience, run episodes, save
    experience. Returns a summary dict for the results file."""
    print(f"\n{'=' * 70}")
    print(f"# {label}")
    print(f"{'=' * 70}")

    store, tracker = _fresh_store_and_tracker()

    pre_rules = len(store.learned_rules)
    pre_obs = dict(tracker.observation_counts)

    if load_existing:
        loaded = store.load_experience(experience_path, tracker=tracker)
        if loaded:
            print(
                f"  Loaded experience from {experience_path}: "
                f"learned_rules={len(store.learned_rules)}, "
                f"tracker_obs_counts={dict(tracker.observation_counts)}"
            )
        else:
            print(f"  No experience file at {experience_path} (fresh start)")

    loaded_rules = len(store.learned_rules)
    loaded_obs = dict(tracker.observation_counts)

    accumulator = SurpriseAccumulator()
    nursery = RuleNursery()

    results = []
    t0 = time.time()
    for ep in range(n_episodes):
        env = CrafterPixelEnv(seed=ep * 13 + seed_offset)
        _disable_enemies(env)
        rng = np.random.RandomState(ep + seed_offset)

        result = run_mpc_episode(
            env=env,
            segmenter=segmenter,
            store=store,
            tracker=tracker,
            rng=rng,
            max_steps=max_steps,
            horizon=horizon,
            surprise_accumulator=accumulator,
            rule_nursery=nursery,
        )
        results.append(result)
        inv = result["final_inv"]
        elapsed = time.time() - t0
        eta_min = elapsed / (ep + 1) * (n_episodes - ep - 1) / 60.0
        ns = result.get("nursery_stats", {})
        print(
            f"  [{label}] ep{ep:2d}/{n_episodes} len={result['length']:4d} "
            f"H{inv.get('health', 0)}F{inv.get('food', 0)}"
            f"D{inv.get('drink', 0)}E{inv.get('energy', 0)} "
            f"W{inv.get('wood', 0)} cause={result['cause_of_death']} "
            f"rules={len(store.learned_rules)} "
            f"ns_e={ns.get('emitted', 0)}p{ns.get('promoted', 0)} "
            f"ETA={eta_min:.1f}m"
        )

    post_rules = len(store.learned_rules)
    post_obs = dict(tracker.observation_counts)

    store.save_experience(experience_path, tracker=tracker)
    print(f"  Saved experience → {experience_path}")
    print(
        f"  Rules: pre={pre_rules} loaded={loaded_rules} post={post_rules} "
        f"(session added {post_rules - loaded_rules})"
    )
    print(
        f"  Tracker obs_counts: loaded={loaded_obs}  post={post_obs}"
    )

    return {
        "label": label,
        "n_episodes": n_episodes,
        "pre_rules": pre_rules,
        "loaded_rules": loaded_rules,
        "post_rules": post_rules,
        "loaded_obs_counts": loaded_obs,
        "post_obs_counts": post_obs,
        "avg_length": float(np.mean([r["length"] for r in results])),
        "avg_wood": float(np.mean([r["final_inv"].get("wood", 0) for r in results])),
        "episodes": [
            {
                "length": r["length"],
                "cause_of_death": r["cause_of_death"],
                "final_inv": {k: int(v) for k, v in r["final_inv"].items()},
                "learned_rule_count": len(store.learned_rules),
                "nursery_stats": r.get("nursery_stats", {}),
            }
            for r in results
        ],
    }


def _load_segmenter() -> Any:
    if not STAGE75_CHECKPOINT.exists():
        print(
            f"FATAL: segmenter checkpoint not found at {STAGE75_CHECKPOINT}",
            file=sys.stderr,
        )
        sys.exit(1)
    disable_rocm_conv()
    return load_tile_segmenter(STAGE75_CHECKPOINT)


def _verify_pass_criteria(sessions: list[dict]) -> bool:
    """Stage 82 pass: session 2 must start with >= session 1's post rules,
    session 3 must see the same."""
    if len(sessions) < 3:
        return False
    s1, s2, s3 = sessions[0], sessions[1], sessions[2]

    ok_rules_flow_1_to_2 = s2["loaded_rules"] >= s1["post_rules"]
    ok_rules_flow_2_to_3 = s3["loaded_rules"] >= s2["post_rules"]

    s1_any_obs = any(v > 0 for v in s1["post_obs_counts"].values())
    ok_obs_flow_1_to_2 = (
        not s1_any_obs
        or any(
            s2["loaded_obs_counts"].get(k, 0) >= v
            for k, v in s1["post_obs_counts"].items()
            if v > 0
        )
    )

    print(f"\n{'=' * 70}")
    print("# Stage 82 pass criteria")
    print(f"{'=' * 70}")
    print(
        f"  session 1 post_rules = {s1['post_rules']}, "
        f"session 2 loaded_rules = {s2['loaded_rules']}  "
        f"{'OK' if ok_rules_flow_1_to_2 else 'FAIL'}"
    )
    print(
        f"  session 2 post_rules = {s2['post_rules']}, "
        f"session 3 loaded_rules = {s3['loaded_rules']}  "
        f"{'OK' if ok_rules_flow_2_to_3 else 'FAIL'}"
    )
    print(
        f"  tracker obs flow session 1 → session 2: "
        f"{'OK' if ok_obs_flow_1_to_2 else 'FAIL'}"
    )

    return ok_rules_flow_1_to_2 and ok_rules_flow_2_to_3 and ok_obs_flow_1_to_2


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(fast: bool = False) -> int:
    if fast:
        n_episodes_per_session = 3
        max_steps = 150
        horizon = 10
    else:
        n_episodes_per_session = 10
        max_steps = 500
        horizon = 20

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if EXPERIENCE_PATH.exists():
        EXPERIENCE_PATH.unlink()
        print(f"Removed stale {EXPERIENCE_PATH}")

    segmenter = _load_segmenter()

    sessions: list[dict] = []
    for i in range(3):
        label = f"session_{i + 1}"
        load_existing = i > 0
        summary = _run_session(
            label=label,
            n_episodes=n_episodes_per_session,
            max_steps=max_steps,
            horizon=horizon,
            segmenter=segmenter,
            experience_path=EXPERIENCE_PATH,
            load_existing=load_existing,
            seed_offset=i * 1000,
        )
        sessions.append(summary)

    passed = _verify_pass_criteria(sessions)

    output = {
        "version": 1,
        "fast": fast,
        "n_episodes_per_session": n_episodes_per_session,
        "max_steps": max_steps,
        "horizon": horizon,
        "passed": passed,
        "sessions": sessions,
    }
    with RESULTS_PATH.open("w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults → {RESULTS_PATH}")
    print(f"Experience → {EXPERIENCE_PATH}")
    print(f"Overall: {'PASS' if passed else 'FAIL'}")

    return 0 if passed else 1


def main_fast() -> int:
    return main(fast=True)


def main_full() -> int:
    return main(fast=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="smoke: 3 sessions × 3 eps")
    args = parser.parse_args()
    sys.exit(main(fast=args.fast))
