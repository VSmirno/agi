"""Stage 80 diagnostic — instrument 5 baseline eval episodes to see WHERE
the agent's time goes. We're stuck at avg_len ~180 across all approaches
(residual, nursery, baseline) — none of the learning fixes have moved
the wall. Before designing Stage 80 we need to understand the failure
mode at the action / trajectory level, not just the aggregate.

For each episode, this script captures:
  - Per-action counts (move_left/right/up/down, do, sleep, place_*, make_*)
  - Action entropy
  - Cause of death + final_inv
  - Wood/stone/coal counts (gathering progress proxy)
  - Episode length
  - Trace JSONL of the first episode (per-tick: visible, body, plan, primitive)

Goal: find one of the following:
  (a) Agent spends most time on navigation (move_*) and barely "do"s →
      action selection is wrong (planner picks navigate over interact)
  (b) Agent does a lot of "do" but no resources accrue → Bug 2 fix
      didn't fully take, OR there's another do-action bug
  (c) Agent rarely sees trees/water/cow in spatial_map → exploration
      isn't reaching resource zones
  (d) Agent sees them but planner doesn't pick gathering plans →
      score_trajectory weights are wrong
  (e) Plans require tools (wood_pickaxe etc) that the agent never makes →
      crafting bottleneck

Output: _docs/stage80_diag_results.txt (raw output) + diagnostic dump
saved alongside the trace.
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

from snks.agent.concept_store import ConceptStore
from snks.agent.crafter_pixel_env import CrafterPixelEnv
from snks.agent.crafter_textbook import CrafterTextbook
from snks.agent.mpc_agent import run_mpc_episode
from snks.agent.perception import HomeostaticTracker
from snks.encoder.cnn_encoder import disable_rocm_conv
from snks.encoder.tile_segmenter import load_tile_segmenter


STAGE75_CHECKPOINT = Path("demos/checkpoints/exp135/segmenter_9x9.pt")
TRACE_DIR = Path("_docs/stage80_diag_trace")
RESULTS_PATH = Path("_docs/stage80_diag_results.json")


def main_full() -> bool:
    return main()


def main() -> bool:
    disable_rocm_conv()
    TRACE_DIR.mkdir(parents=True, exist_ok=True)

    print("[Phase 0] Load segmenter")
    from snks.encoder.tile_segmenter import pick_device as _seg_dev
    seg_device = _seg_dev()
    segmenter = load_tile_segmenter(str(STAGE75_CHECKPOINT), device=seg_device)
    print(f"  segmenter device: {seg_device}")

    print("\n[Phase 1] Build store + tracker")
    store = ConceptStore()
    tb = CrafterTextbook("configs/crafter_textbook.yaml")
    tb.load_into(store)
    tracker = HomeostaticTracker()
    tracker.init_from_textbook(tb.body_block, store.passive_rules)
    print(f"  {len(store.concepts)} concepts, {len(store.passive_rules)} passive rules")

    print("\n[Phase 2] Run 5 eval episodes (enemies on, max_steps=1000) with trace")

    n_eps = 5
    seed_offset = 1000  # same as Stage 78c/79 eval_run0
    results: list[dict] = []
    t_start = time.time()

    for ep in range(n_eps):
        env = CrafterPixelEnv(seed=ep * 11 + seed_offset)
        rng = np.random.RandomState(ep + seed_offset)

        trace_path = TRACE_DIR / f"ep{ep}.jsonl"
        if trace_path.exists():
            trace_path.unlink()

        result = run_mpc_episode(
            env=env,
            segmenter=segmenter,
            store=store,
            tracker=tracker,
            rng=rng,
            max_steps=1000,
            horizon=20,
            trace_path=trace_path,
        )
        result["episode"] = ep
        results.append(result)

        inv = result["final_inv"]
        action_counts = result.get("action_counts", {})
        # Aggregate by action category
        by_category = Counter()
        for action, count in action_counts.items():
            if action.startswith("move_"):
                by_category["move"] += count
            elif action.startswith("place_"):
                by_category["place"] += count
            elif action.startswith("make_"):
                by_category["make"] += count
            elif action == "do":
                by_category["do"] += count
            elif action == "sleep":
                by_category["sleep"] += count
            else:
                by_category["other"] += count

        elapsed = time.time() - t_start
        print(
            f"\n  ep{ep:2d} len={result['length']:4d} "
            f"H{inv.get('health', 0)}F{inv.get('food', 0)}D{inv.get('drink', 0)}E{inv.get('energy', 0)}  "
            f"W{inv.get('wood', 0)}S{inv.get('stone', 0)}C{inv.get('coal', 0)}I{inv.get('iron', 0)}  "
            f"cause={result['cause_of_death']}"
        )
        print(
            f"      action_categories: {dict(by_category)}"
        )
        print(f"      action_counts (top 10): "
              f"{dict(Counter(action_counts).most_common(10))}")
        print(f"      action_entropy: {result['action_entropy']:.2f}")
        print(f"      elapsed: {elapsed:.0f}s")

    # Aggregate
    print("\n" + "=" * 70)
    print("AGGREGATE (5 eval eps, enemies on, max_steps=1000)")
    print("=" * 70)
    avg_len = float(np.mean([r["length"] for r in results]))
    print(f"  avg_len: {avg_len:.1f}")
    causes = Counter(r["cause_of_death"] for r in results)
    print(f"  causes: {dict(causes)}")

    total_actions: Counter = Counter()
    for r in results:
        for k, v in r.get("action_counts", {}).items():
            total_actions[k] += v
    print(f"  total actions: {sum(total_actions.values())}")

    by_cat: Counter = Counter()
    for action, count in total_actions.items():
        if action.startswith("move_"):
            by_cat["move"] += count
        elif action.startswith("place_"):
            by_cat["place"] += count
        elif action.startswith("make_"):
            by_cat["make"] += count
        elif action == "do":
            by_cat["do"] += count
        elif action == "sleep":
            by_cat["sleep"] += count
        else:
            by_cat["other"] += count

    total = sum(by_cat.values())
    print("  by category (across 5 eps):")
    for cat, count in by_cat.most_common():
        print(f"    {cat:8s}: {count:5d} ({100 * count / total:.1f}%)")

    print(f"  top 15 individual actions:")
    for action, count in total_actions.most_common(15):
        print(f"    {action:25s}: {count:5d}")

    # Resource gathering
    final_invs = [r["final_inv"] for r in results]
    woods = [inv.get("wood", 0) for inv in final_invs]
    stones = [inv.get("stone", 0) for inv in final_invs]
    coals = [inv.get("coal", 0) for inv in final_invs]
    irons = [inv.get("iron", 0) for inv in final_invs]
    print(f"\n  Resource collection (5 eps):")
    print(f"    wood:  mean={np.mean(woods):.2f} max={max(woods)} ≥3:{sum(1 for w in woods if w >= 3)}/5")
    print(f"    stone: mean={np.mean(stones):.2f} max={max(stones)}")
    print(f"    coal:  mean={np.mean(coals):.2f} max={max(coals)}")
    print(f"    iron:  mean={np.mean(irons):.2f} max={max(irons)}")

    # Save results
    summary = {
        "avg_len": avg_len,
        "causes": dict(causes),
        "total_actions": dict(total_actions),
        "by_category": dict(by_cat),
        "resources": {
            "wood": {"mean": float(np.mean(woods)), "max": int(max(woods))},
            "stone": {"mean": float(np.mean(stones)), "max": int(max(stones))},
            "coal": {"mean": float(np.mean(coals)), "max": int(max(coals))},
            "iron": {"mean": float(np.mean(irons)), "max": int(max(irons))},
        },
        "episodes": [
            {
                "episode": r["episode"],
                "length": r["length"],
                "cause_of_death": r["cause_of_death"],
                "final_inv": dict(r["final_inv"]),
                "action_counts": dict(r.get("action_counts", {})),
                "action_entropy": r["action_entropy"],
            }
            for r in results
        ],
    }
    with RESULTS_PATH.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved: {RESULTS_PATH}")
    print(f"  Per-episode traces: {TRACE_DIR}/ep{{0..4}}.jsonl")

    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
