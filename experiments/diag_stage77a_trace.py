"""Stage 77a diagnostic — per-tick trace + aggregated stats.

Runs a small number of enemy-on episodes with `run_mpc_episode(trace_path=...)`
and dumps one JSONL file per episode. Then aggregates across episodes to show
where the agent spends time, which rules fire, and how surprising reality is
compared to the MPC prediction each tick.

This is a Phase-1 systematic-debugging tool: gather evidence before touching
architecture. No fixes here — print and exit.

Usage (minipc):
  PYTHONPATH=src venv/bin/python experiments/diag_stage77a_trace.py \
    --n-episodes 3 --max-steps 300 --horizon 40

Output:
  _docs/diag_stage77a_trace/ep{0..N}.jsonl   — per-tick traces
  stdout                                     — aggregate stats
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from snks.agent.concept_store import ConceptStore
from snks.agent.crafter_pixel_env import CrafterPixelEnv
from snks.agent.crafter_textbook import CrafterTextbook
from snks.agent.mpc_agent import run_mpc_episode
from snks.agent.perception import HomeostaticTracker
from snks.encoder.cnn_encoder import disable_rocm_conv
from snks.encoder.tile_segmenter import load_tile_segmenter, pick_device


STAGE75_CHECKPOINT = Path("demos/checkpoints/exp135/segmenter_9x9.pt")
TRACE_DIR = Path("_docs/diag_stage77a_trace")


def phase_warmup(segmenter, store, tracker, horizon: int, n: int = 30) -> None:
    """Small warmup to let the tracker accumulate some observed rates before
    the diagnostic eval. Without this the innate priors dominate and the
    surprise signal becomes noisier."""
    print(f"\n[warmup] {n} episodes (safe), horizon={horizon}")
    t0 = time.time()
    for ep in range(n):
        env = CrafterPixelEnv(seed=ep * 7 + 100)
        try:
            env._env._balance_chunk = lambda *a, **kw: None
        except Exception:
            pass
        rng = np.random.RandomState(ep + 100)
        run_mpc_episode(
            env=env, segmenter=segmenter, store=store, tracker=tracker,
            rng=rng, max_steps=300, horizon=horizon,
        )
    print(f"[warmup] done in {time.time() - t0:.0f}s")


def run_diagnostic_episodes(
    segmenter, store, tracker, n_episodes: int, max_steps: int, horizon: int,
) -> list[Path]:
    """Run N enemy-on episodes with trace enabled."""
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    trace_files: list[Path] = []
    for ep in range(n_episodes):
        trace_path = TRACE_DIR / f"ep{ep}.jsonl"
        if trace_path.exists():
            trace_path.unlink()
        env = CrafterPixelEnv(seed=ep * 13 + 2000)  # enemies ON
        rng = np.random.RandomState(ep + 2000)
        t0 = time.time()
        result = run_mpc_episode(
            env=env, segmenter=segmenter, store=store, tracker=tracker,
            rng=rng, max_steps=max_steps, horizon=horizon,
            trace_path=trace_path,
        )
        print(
            f"  ep{ep}: len={result['length']} cause={result['cause_of_death']} "
            f"wood={result['final_inv'].get('wood', 0)} "
            f"took {time.time() - t0:.0f}s"
        )
        trace_files.append(trace_path)
    return trace_files


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def load_trace(path: Path) -> tuple[list[dict], dict]:
    """Return (tick_entries, episode_end_marker)."""
    entries: list[dict] = []
    end_marker: dict = {}
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("episode_end"):
                end_marker = obj
            else:
                entries.append(obj)
    return entries, end_marker


def aggregate(traces: list[tuple[list[dict], dict]]) -> None:
    """Print aggregate stats across all trace files."""
    total_ticks = 0
    lengths: list[int] = []
    causes: Counter = Counter()
    chosen_origins: Counter = Counter()
    n_cands_dist: list[int] = []
    first_actions: Counter = Counter()
    primitives: Counter = Counter()

    # Surprise accumulation per var
    surprise_abs: dict[str, list[float]] = defaultdict(list)
    surprise_signed: dict[str, list[float]] = defaultdict(list)

    # Rule / verification
    verify_count = 0
    verify_by_outcome: Counter = Counter()

    # Damage sources seen in chosen trajectories
    damage_sources: Counter = Counter()

    # Per-tick plan_origins breakdown — which origins WERE generated (not just picked)
    origins_generated: Counter = Counter()

    # Stuck-move detection: how often does move_* fail (prev_primitive_landed=False)
    move_attempts = 0
    move_failed = 0

    # Near-concept time: where does the agent stand
    near_time: Counter = Counter()

    # Visible concept frequency
    visible_freq: Counter = Counter()

    # Final 20 ticks before death per episode (last resort diagnostics)
    last_ticks_per_ep: list[list[dict]] = []

    for entries, end in traces:
        lengths.append(end.get("length", len(entries)))
        causes[end.get("cause_of_death", "unknown")] += 1
        for e in entries:
            total_ticks += 1
            chosen = e["chosen"]
            chosen_origins[chosen["origin"]] += 1
            n_cands_dist.append(e["n_candidates"])
            first_actions[chosen["first_action"] or "none"] += 1
            primitives[e["primitive"]] += 1
            near_time[e["near"]] += 1
            for v in e["visible"]:
                visible_freq[v] += 1
            for var, delta in e.get("surprise", {}).items():
                surprise_abs[var].append(abs(delta))
                surprise_signed[var].append(delta)
            if e.get("verified"):
                verify_count += 1
                verify_by_outcome[e.get("outcome") or "_none"] += 1
            for dp in chosen.get("damage_preview", []):
                damage_sources[f"{dp['source']} → {dp['var']}"] += dp["amount"]
            for origin, cnt in e.get("plan_origins", {}).items():
                origins_generated[origin] += cnt
            if e["primitive"].startswith("move_"):
                move_attempts += 1
                # prev_primitive_landed refers to the PREVIOUS move — so look at the NEXT entry
                # We can instead count via "self failed" by looking backward:
            # Separate pass for stuck detection
        # Count stuck moves by scanning pairs
        for i in range(1, len(entries)):
            prev = entries[i - 1]
            cur = entries[i]
            if prev["primitive"].startswith("move_") and not cur.get("prev_primitive_landed", True):
                move_failed += 1
        last_ticks_per_ep.append(entries[-20:] if len(entries) > 20 else entries)

    n_eps = len(traces)
    print("\n" + "=" * 70)
    print(f"AGGREGATE — {n_eps} episodes, {total_ticks} total ticks")
    print("=" * 70)
    print(f"  lengths: {lengths}  avg={np.mean(lengths):.0f}")
    print(f"  causes: {dict(causes)}")

    print("\n[PLAN GENERATION & SELECTION]")
    print(f"  avg candidates per tick: {np.mean(n_cands_dist):.1f}")
    print(f"    distribution: min={min(n_cands_dist)} p50={int(np.median(n_cands_dist))} max={max(n_cands_dist)}")
    print(f"  origins GENERATED (sum across ticks):")
    for origin, cnt in origins_generated.most_common():
        pct = 100 * cnt / max(1, sum(origins_generated.values()))
        print(f"    {origin:12s}: {cnt:5d} ({pct:5.1f}%)")
    print(f"  origins PICKED (chosen plan per tick):")
    for origin, cnt in chosen_origins.most_common():
        pct = 100 * cnt / max(1, total_ticks)
        print(f"    {origin:12s}: {cnt:5d} ({pct:5.1f}%)")
    print(f"  first action of chosen plan:")
    for act, cnt in first_actions.most_common(10):
        pct = 100 * cnt / max(1, total_ticks)
        print(f"    {act:12s}: {cnt:5d} ({pct:5.1f}%)")

    print("\n[PRIMITIVE EXECUTED]")
    for prim, cnt in primitives.most_common(10):
        pct = 100 * cnt / max(1, total_ticks)
        print(f"    {prim:14s}: {cnt:5d} ({pct:5.1f}%)")

    print("\n[MOVEMENT STUCK-RATE]")
    print(f"  move_* attempts: {move_attempts}")
    print(f"  move_* failed (pos unchanged): {move_failed} "
          f"({100 * move_failed / max(1, move_attempts):.1f}%)")

    print("\n[NEAR-CONCEPT TIME]")
    for concept, cnt in near_time.most_common(10):
        pct = 100 * cnt / max(1, total_ticks)
        print(f"    {concept:12s}: {cnt:5d} ({pct:5.1f}%)")

    print("\n[VISIBLE CONCEPTS (frequency across ticks)]")
    for concept, cnt in visible_freq.most_common(15):
        pct = 100 * cnt / max(1, total_ticks)
        print(f"    {concept:12s}: {cnt:5d} ({pct:5.1f}%)")

    print("\n[SURPRISE: |actual - predicted| per body var]")
    for var in ("health", "food", "drink", "energy"):
        absl = surprise_abs.get(var, [])
        sgnd = surprise_signed.get(var, [])
        if not absl:
            continue
        print(
            f"  {var:6s}: mean_abs={np.mean(absl):.3f} "
            f"max_abs={max(absl):.2f} "
            f"mean_signed={np.mean(sgnd):+.3f} "
            f"n>0.5={sum(1 for x in absl if x > 0.5)} "
            f"n>1.0={sum(1 for x in absl if x > 1.0)}"
        )

    print("\n[VERIFY_OUTCOME CALLS]")
    print(f"  total verified: {verify_count}")
    print(f"  by outcome label (top 10):")
    for label, cnt in verify_by_outcome.most_common(10):
        print(f"    {label:16s}: {cnt}")

    print("\n[DAMAGE SOURCES IN CHOSEN-PLAN TRAJECTORIES] (cumulative negative body_delta)")
    for src, total_amount in damage_sources.most_common(10):
        print(f"    {src:38s}: {total_amount:+.2f}")

    print("\n[LAST 5 TICKS OF EACH EPISODE]")
    for ep_idx, last in enumerate(last_ticks_per_ep):
        if not last:
            continue
        print(f"  --- ep{ep_idx} (final {min(5, len(last))} ticks) ---")
        for e in last[-5:]:
            body = e["body"]
            chosen = e["chosen"]
            dmg = chosen.get("damage_preview", [])
            if dmg:
                dmg_parts = [f"{d['source']}:{d['amount']:+.1f}" for d in dmg[:2]]
                dmg_str = " dmg=" + ",".join(dmg_parts)
            else:
                dmg_str = ""
            print(
                f"    s{e['step']:3d} "
                f"H{body.get('health', 0):.0f}F{body.get('food', 0):.0f}"
                f"D{body.get('drink', 0):.0f}E{body.get('energy', 0):.0f} "
                f"near={e['near']:10s} "
                f"{chosen['origin']:8s}→{e['primitive']:14s} "
                f"pred_end={chosen['predicted_final_body'].get('health'):.1f}H"
                f"{dmg_str}"
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--horizon", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=30,
                        help="warmup episodes (safe) to condition tracker; set 0 to skip")
    args = parser.parse_args()

    t_start = time.time()
    disable_rocm_conv()

    print("=" * 70)
    print("Stage 77a diagnostic trace")
    print("=" * 70)

    device = pick_device()
    segmenter = load_tile_segmenter(str(STAGE75_CHECKPOINT), device=device)

    store = ConceptStore()
    tb = CrafterTextbook("configs/crafter_textbook.yaml")
    tb.load_into(store)
    print(f"  Loaded {len(store.concepts)} concepts, "
          f"{sum(len(c.causal_links) for c in store.concepts.values())} rules, "
          f"{len(store.passive_rules)} passive rules")

    tracker = HomeostaticTracker()
    tracker.init_from_textbook(tb.body_block, store.passive_rules)

    if args.warmup > 0:
        phase_warmup(segmenter, store, tracker, horizon=args.horizon, n=args.warmup)

    print(
        f"\n[diag] {args.n_episodes} enemy-on episodes, "
        f"max_steps={args.max_steps}, horizon={args.horizon}"
    )
    trace_files = run_diagnostic_episodes(
        segmenter, store, tracker,
        n_episodes=args.n_episodes, max_steps=args.max_steps, horizon=args.horizon,
    )

    traces = [load_trace(p) for p in trace_files]
    aggregate(traces)

    print(f"\n  Total time: {(time.time() - t_start) / 60:.1f} min")
    print(f"  Trace files: {TRACE_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
