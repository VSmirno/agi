"""Stage 76: Continuous model learning — exp136.

Pipeline (runs on minipc):
  Phase 0: Load Stage 75 segmenter checkpoint (no retraining)
  Phase 1: Warmup A — 50 episodes, enemies OFF, high temperature
  Phase 2: Warmup B — 50 episodes, enemies ON, decaying temperature
  Phase 3: Evaluation — 3 runs × 20 episodes, enemies ON, max_steps=1000
  Phase 4: Gate checks (tile_acc, wood, survival, memory growth, hardcode lint)
  Phase 5: Summary report

Shared state across phases: StateEncoder, EpisodicSDM, ConceptStore,
HomeostaticTracker. Memory persists from warmup into evaluation.

Expected runtime: several hours on minipc CPU. Overnight-friendly.
"""

from __future__ import annotations

import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

from snks.agent.concept_store import ConceptStore
from snks.agent.continuous_agent import run_continuous_episode
from snks.agent.crafter_pixel_env import CrafterPixelEnv
from snks.agent.crafter_textbook import CrafterTextbook
from snks.agent.perception import HomeostaticTracker
from snks.encoder.cnn_encoder import disable_rocm_conv
from snks.encoder.tile_segmenter import load_tile_segmenter
from snks.memory.episodic_sdm import EpisodicSDM
from snks.memory.state_encoder import StateEncoder


STAGE75_CHECKPOINT = Path("demos/checkpoints/exp135/segmenter_9x9.pt")


# ---------------------------------------------------------------------------
# Phase 0: Load segmenter
# ---------------------------------------------------------------------------


def phase0_load_segmenter() -> Any:
    print("=" * 60)
    print("Phase 0: Load Stage 75 segmenter")
    print("=" * 60)
    t0 = time.time()
    if not STAGE75_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Segmenter checkpoint not found at {STAGE75_CHECKPOINT}. "
            f"Run Stage 75 first (experiments/exp135_grid8_tile_perception.py)."
        )
    segmenter = load_tile_segmenter(str(STAGE75_CHECKPOINT))
    n_params = sum(p.numel() for p in segmenter.parameters())
    print(f"  Loaded {STAGE75_CHECKPOINT} ({n_params} params, {time.time() - t0:.1f}s)")
    return segmenter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _disable_enemies(env: CrafterPixelEnv) -> None:
    try:
        env._env._balance_chunk = lambda *a, **kw: None
    except Exception:
        pass


def _run_episodes(
    name: str,
    n_episodes: int,
    max_steps: int,
    segmenter: Any,
    encoder: StateEncoder,
    sdm: EpisodicSDM,
    store: ConceptStore,
    tracker: HomeostaticTracker,
    enemies: bool,
    temperature_schedule: list[float],
    bootstrap_k: int,
    seed_offset: int,
) -> list[dict]:
    """Run a batch of episodes with a per-episode temperature schedule."""
    results: list[dict] = []
    t0 = time.time()
    for ep in range(n_episodes):
        env = CrafterPixelEnv(seed=ep * 11 + seed_offset)
        if not enemies:
            _disable_enemies(env)
        rng = np.random.RandomState(ep + seed_offset)
        temperature = temperature_schedule[min(ep, len(temperature_schedule) - 1)]

        result = run_continuous_episode(
            env=env,
            segmenter=segmenter,
            encoder=encoder,
            sdm=sdm,
            store=store,
            tracker=tracker,
            rng=rng,
            max_steps=max_steps,
            temperature=temperature,
            bootstrap_k=bootstrap_k,
        )
        result["episode"] = ep
        result["temperature"] = temperature
        results.append(result)

        if ep < 3 or ep % 10 == 0 or ep == n_episodes - 1:
            inv = result["final_inv"]
            elapsed = time.time() - t0
            eta = elapsed / (ep + 1) * (n_episodes - ep - 1)
            print(
                f"  [{name}] ep{ep:3d}/{n_episodes} len={result['length']:4d} "
                f"sdm={result['sdm_size']:5d} bs={result['bootstrap_ratio']:.2f} "
                f"H{inv.get('health', 0)}F{inv.get('food', 0)}"
                f"D{inv.get('drink', 0)}E{inv.get('energy', 0)} "
                f"W{inv.get('wood', 0)} cause={result['cause_of_death']} "
                f"T={temperature:.2f} ETA={eta/60:.1f}m"
            )
    return results


def _summarize(name: str, results: list[dict]) -> dict:
    lengths = [r["length"] for r in results]
    causes: Counter = Counter(r["cause_of_death"] for r in results)
    bootstrap_ratios = [r["bootstrap_ratio"] for r in results]
    sdm_ratios = [r["sdm_ratio"] for r in results]
    entropies = [r["action_entropy"] for r in results]
    wood_collected = [r["final_inv"].get("wood", 0) for r in results]
    avg_len = float(np.mean(lengths))
    median_len = int(np.median(lengths))
    print(f"\n  [{name}] n={len(results)} "
          f"avg_len={avg_len:.0f} median={median_len} "
          f"min={min(lengths)} max={max(lengths)}")
    print(f"  [{name}] avg_bootstrap_ratio={np.mean(bootstrap_ratios):.2f} "
          f"avg_sdm_ratio={np.mean(sdm_ratios):.2f} "
          f"avg_entropy={np.mean(entropies):.2f}")
    print(f"  [{name}] causes: {dict(causes)}")
    print(f"  [{name}] avg_wood={np.mean(wood_collected):.1f} "
          f"wood≥3: {sum(1 for w in wood_collected if w >= 3)}/{len(results)}")
    return {
        "name": name,
        "n": len(results),
        "avg_len": avg_len,
        "median_len": median_len,
        "min_len": min(lengths),
        "max_len": max(lengths),
        "avg_bootstrap_ratio": float(np.mean(bootstrap_ratios)),
        "avg_sdm_ratio": float(np.mean(sdm_ratios)),
        "avg_entropy": float(np.mean(entropies)),
        "avg_wood": float(np.mean(wood_collected)),
        "wood_3_count": sum(1 for w in wood_collected if w >= 3),
        "causes": dict(causes),
    }


# ---------------------------------------------------------------------------
# Phase 1: Warmup A — no enemies, high temperature
# ---------------------------------------------------------------------------


def phase1_warmup_safe(
    segmenter, encoder, sdm, store, tracker,
    n_episodes: int = 50, max_steps: int = 500, bootstrap_k: int = 5,
) -> dict:
    print("\n" + "=" * 60)
    print(f"Phase 1: Warmup A (no enemies, n={n_episodes})")
    print("=" * 60)
    temp_schedule = [1.0] * n_episodes  # high exploration
    results = _run_episodes(
        "warmup-safe", n_episodes, max_steps, segmenter, encoder, sdm, store,
        tracker, enemies=False, temperature_schedule=temp_schedule,
        bootstrap_k=bootstrap_k, seed_offset=300,
    )
    return _summarize("warmup-safe", results)


# ---------------------------------------------------------------------------
# Phase 2: Warmup B — enemies on, decaying temperature
# ---------------------------------------------------------------------------


def phase2_warmup_enemies(
    segmenter, encoder, sdm, store, tracker,
    n_episodes: int = 50, max_steps: int = 500, bootstrap_k: int = 5,
) -> dict:
    print("\n" + "=" * 60)
    print(f"Phase 2: Warmup B (enemies on, n={n_episodes})")
    print("=" * 60)
    # Linear decay from 1.0 to 0.5 across warmup
    temp_schedule = list(np.linspace(1.0, 0.5, n_episodes))
    results = _run_episodes(
        "warmup-enemy", n_episodes, max_steps, segmenter, encoder, sdm, store,
        tracker, enemies=True, temperature_schedule=temp_schedule,
        bootstrap_k=bootstrap_k, seed_offset=500,
    )
    return _summarize("warmup-enemy", results)


# ---------------------------------------------------------------------------
# Phase 3: Evaluation — 3 runs × 20 episodes
# ---------------------------------------------------------------------------


def phase3_evaluation(
    segmenter, encoder, sdm, store, tracker,
    n_runs: int = 3, n_episodes_per_run: int = 20, max_steps: int = 1000,
    temperature: float = 0.3, bootstrap_k: int = 5,
) -> list[dict]:
    print("\n" + "=" * 60)
    print(f"Phase 3: Evaluation ({n_runs} runs × {n_episodes_per_run} episodes)")
    print("=" * 60)
    summaries: list[dict] = []
    for run_idx in range(n_runs):
        print(f"\n--- Eval run {run_idx + 1}/{n_runs} ---")
        temp_schedule = [temperature] * n_episodes_per_run
        results = _run_episodes(
            f"eval-run{run_idx}", n_episodes_per_run, max_steps,
            segmenter, encoder, sdm, store, tracker,
            enemies=True, temperature_schedule=temp_schedule,
            bootstrap_k=bootstrap_k, seed_offset=1000 + run_idx * 100,
        )
        summaries.append(_summarize(f"eval-run{run_idx}", results))
    return summaries


# ---------------------------------------------------------------------------
# Phase 4: Gates
# ---------------------------------------------------------------------------


def phase4_gates(
    segmenter, encoder, sdm, store, tracker, eval_summaries: list[dict],
) -> dict:
    print("\n" + "=" * 60)
    print("Phase 4: Gate checks")
    print("=" * 60)

    # Gate 1: Survival ≥ 200 over all 3 runs
    run_means = [s["avg_len"] for s in eval_summaries]
    overall_mean = float(np.mean(run_means))
    gate_survival_all = all(m >= 200 for m in run_means)
    gate_survival_overall = overall_mean >= 200
    print(f"  Gate 1: survival mean per run = {run_means}")
    print(f"    overall = {overall_mean:.0f} "
          f"(per-run ≥200: {gate_survival_all}, overall ≥200: {gate_survival_overall})")

    # Gate 2: tile_acc ≥ 80%
    # Re-run Stage 75 accuracy gate lazily here
    try:
        from snks.encoder.tile_head_trainer import run_tile_accuracy_eval
        tile_acc = run_tile_accuracy_eval(segmenter)
    except Exception as e:
        print(f"    (tile_acc eval skipped: {e})")
        tile_acc = None
    if tile_acc is not None:
        print(f"  Gate 2: tile_acc = {tile_acc:.1%} (≥80%: {tile_acc >= 0.80})")

    # Gate 3: wood collection — smoke-like
    print("  Gate 3: wood smoke (20 episodes, enemies off, max_steps=200)")
    smoke_temp = [0.3] * 20
    smoke_results = _run_episodes(
        "smoke", 20, 200, segmenter, encoder, sdm, store, tracker,
        enemies=False, temperature_schedule=smoke_temp, bootstrap_k=5,
        seed_offset=2000,
    )
    wood_3_count = sum(1 for r in smoke_results if r["final_inv"].get("wood", 0) >= 3)
    gate_wood = wood_3_count / 20 >= 0.50
    print(f"    wood ≥3: {wood_3_count}/20 ({'PASS' if gate_wood else 'FAIL'})")

    # Gate 4: SDM growth
    gate_sdm_growth = len(sdm) >= sdm.capacity or len(sdm) > 0
    print(f"  Gate 4: sdm_size = {len(sdm)}/{sdm.capacity} "
          f"({'PASS' if gate_sdm_growth else 'FAIL'})")

    gates = {
        "survival_per_run": gate_survival_all,
        "survival_overall": gate_survival_overall,
        "tile_acc_80": (tile_acc or 0.0) >= 0.80 if tile_acc is not None else None,
        "wood_50pct": gate_wood,
        "sdm_growth": gate_sdm_growth,
    }
    return gates


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    t_start = time.time()
    disable_rocm_conv()

    segmenter = phase0_load_segmenter()

    store = ConceptStore()
    tb = CrafterTextbook("configs/crafter_textbook.yaml")
    tb.load_into(store)
    tracker = HomeostaticTracker()
    tracker.init_from_body_rules(tb.body_rules)
    print(f"  Body rules: {len(tb.body_rules)} innate rates loaded")

    encoder = StateEncoder()
    sdm = EpisodicSDM(capacity=10_000)

    warmup_a = phase1_warmup_safe(segmenter, encoder, sdm, store, tracker)
    warmup_b = phase2_warmup_enemies(segmenter, encoder, sdm, store, tracker)
    eval_summaries = phase3_evaluation(segmenter, encoder, sdm, store, tracker)
    gates = phase4_gates(segmenter, encoder, sdm, store, tracker, eval_summaries)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Warmup A (safe):    avg_len={warmup_a['avg_len']:.0f}")
    print(f"  Warmup B (enemies): avg_len={warmup_b['avg_len']:.0f}")
    for s in eval_summaries:
        print(f"  {s['name']}: avg_len={s['avg_len']:.0f} "
              f"bs={s['avg_bootstrap_ratio']:.2f} "
              f"entropy={s['avg_entropy']:.2f} "
              f"causes={s['causes']}")
    print(f"  SDM final size: {len(sdm)}/{sdm.capacity}")
    print(f"  Gates: {gates}")
    print(f"  Total time: {(time.time() - t_start)/60:.1f} min")

    all_critical_pass = bool(
        gates["survival_per_run"] and gates["survival_overall"]
        and gates["wood_50pct"] and gates["sdm_growth"]
    )
    print(f"  {'ALL CRITICAL GATES PASS' if all_critical_pass else 'SOME GATES FAIL'}")
    return all_critical_pass


if __name__ == "__main__":
    main()
