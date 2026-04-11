"""Stage 78c: MLP Residual Crafter integration + ablation + eval.

Purpose
-------
Stage 78b validated the ResidualBodyPredictor on a synthetic toy task
(PASS, conj_health_mse = 0.0064). But synthetic PASS ≠ real progress —
the only indicator that matters is Crafter with standard metrics:
survival mean, wood/coal/iron collection, achievements, action entropy.
This stage wires the residual into `ConceptStore.simulate_forward` and
trains it online from env rollouts, then runs an ablation:
`residual_off` (baseline, equivalent to Stage 77a) vs `residual_on`
(residual injected into planning + online SGD on observed body gap).

Both ablations run with IDENTICAL seeds so the delta is attributable
to the residual alone, not to env variance. They each get a fresh
ConceptStore + HomeostaticTracker (shared segmenter — read-only).

Pipeline per ablation
---------------------
  Phase 0 (shared): Load Stage 75 segmenter checkpoint
  Phase 1: Warmup A — 50 eps, enemies OFF (observation accumulation)
  Phase 2: Warmup B — 50 eps, enemies ON (conditional rate learning)
  Phase 3: Evaluation — 3 runs × 20 episodes, enemies ON, max_steps=1000
  Phase 4: Checkpoint residual (if enabled)
  Phase 5: Summarize + compare against Stage 77a baseline (eval mean 180)

Output
------
  _docs/stage78c_results.json  — full dict with per-phase/per-run metrics
  demos/checkpoints/exp138/residual_after_warmup_a.pt
  demos/checkpoints/exp138/residual_after_warmup_b.pt
  demos/checkpoints/exp138/residual_after_eval.pt

Usage
-----
    # Full run (overnight on minipc GPU):
    python experiments/stage78c_residual_crafter.py

    # Fast smoke (local CPU allowed for this flag ONLY):
    python experiments/stage78c_residual_crafter.py --fast

Stage 77a reference baseline (from docs/reports/stage-77a-report.md):
  warmup_a avg_len ≈ 222, warmup_b avg_len ≈ 203, eval avg_len ≈ 180,
  wood ≥3 rate = 0/20 in eval.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

from snks.agent.concept_store import ConceptStore
from snks.agent.crafter_pixel_env import CrafterPixelEnv
from snks.agent.crafter_textbook import CrafterTextbook
from snks.agent.mpc_agent import run_mpc_episode
from snks.agent.perception import HomeostaticTracker
from snks.encoder.cnn_encoder import disable_rocm_conv
from snks.encoder.tile_segmenter import load_tile_segmenter
from snks.learning.residual_predictor import ResidualBodyPredictor, ResidualConfig


STAGE75_CHECKPOINT = Path("demos/checkpoints/exp135/segmenter_9x9.pt")
CHECKPOINT_DIR = Path("demos/checkpoints/exp138")
RESULTS_PATH = Path("_docs/stage78c_results.json")

# Stage 77a reference numbers (from docs/reports/stage-77a-report.md).
STAGE77A_BASELINE = {
    "warmup_a_avg_len": 222.0,
    "warmup_b_avg_len": 203.0,
    "eval_avg_len": 180.0,
    "eval_wood_3_rate": 0.0,
}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _disable_enemies(env: CrafterPixelEnv) -> None:
    try:
        env._env._balance_chunk = lambda *a, **kw: None
    except Exception:
        pass


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _fresh_store_and_tracker() -> tuple[ConceptStore, HomeostaticTracker, CrafterTextbook]:
    """Each ablation gets its own store + tracker to avoid cross-contamination.
    Observation-driven rate learning in HomeostaticTracker must not leak between
    residual_off and residual_on runs, otherwise the delta is confounded.
    """
    store = ConceptStore()
    tb = CrafterTextbook("configs/crafter_textbook.yaml")
    tb.load_into(store)
    tracker = HomeostaticTracker()
    tracker.init_from_textbook(tb.body_block, store.passive_rules)
    return store, tracker, tb


# ---------------------------------------------------------------------------
# Episode runner (thin wrapper around run_mpc_episode)
# ---------------------------------------------------------------------------


def _run_episodes(
    name: str,
    n_episodes: int,
    max_steps: int,
    segmenter: Any,
    store: ConceptStore,
    tracker: HomeostaticTracker,
    enemies: bool,
    horizon: int,
    seed_offset: int,
    residual_predictor: Any,
    residual_optimizer: Any,
    residual_train: bool,
) -> list[dict]:
    """Run one batch of episodes. residual_* are passed straight through."""
    results: list[dict] = []
    t0 = time.time()
    for ep in range(n_episodes):
        env = CrafterPixelEnv(seed=ep * 11 + seed_offset)
        if not enemies:
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
            residual_predictor=residual_predictor,
            residual_optimizer=residual_optimizer,
            residual_train=residual_train,
        )
        result["episode"] = ep
        results.append(result)

        if ep < 3 or ep % 10 == 0 or ep == n_episodes - 1:
            inv = result["final_inv"]
            elapsed = time.time() - t0
            eta_min = elapsed / (ep + 1) * (n_episodes - ep - 1) / 60.0
            res_loss = result.get("residual_loss_mean")
            res_str = f" rl={res_loss:.4f}" if res_loss is not None else ""
            print(
                f"  [{name}] ep{ep:3d}/{n_episodes} len={result['length']:4d} "
                f"H{inv.get('health', 0)}F{inv.get('food', 0)}"
                f"D{inv.get('drink', 0)}E{inv.get('energy', 0)} "
                f"W{inv.get('wood', 0)} cause={result['cause_of_death']}"
                f"{res_str} ETA={eta_min:.1f}m"
            )
    return results


def _summarize(name: str, results: list[dict]) -> dict:
    lengths = [r["length"] for r in results]
    causes: Counter = Counter(r["cause_of_death"] for r in results)
    entropies = [r["action_entropy"] for r in results]
    wood_collected = [r["final_inv"].get("wood", 0) for r in results]
    coal_collected = [r["final_inv"].get("coal", 0) for r in results]
    iron_collected = [r["final_inv"].get("iron", 0) for r in results]
    avg_len = float(np.mean(lengths))
    median_len = int(np.median(lengths))
    std_len = float(np.std(lengths))

    residual_losses = [r.get("residual_loss_mean") for r in results if r.get("residual_loss_mean") is not None]
    residual_loss_mean = float(np.mean(residual_losses)) if residual_losses else None

    print(
        f"\n  [{name}] n={len(results)} avg_len={avg_len:.1f}±{std_len:.1f} "
        f"median={median_len} min={min(lengths)} max={max(lengths)}"
    )
    print(f"  [{name}] avg_entropy={np.mean(entropies):.2f}")
    print(f"  [{name}] causes: {dict(causes)}")
    print(
        f"  [{name}] wood avg={np.mean(wood_collected):.2f} "
        f"wood≥3: {sum(1 for w in wood_collected if w >= 3)}/{len(results)}"
    )
    print(
        f"  [{name}] coal avg={np.mean(coal_collected):.2f} "
        f"iron avg={np.mean(iron_collected):.2f}"
    )
    if residual_loss_mean is not None:
        print(f"  [{name}] residual_loss_mean={residual_loss_mean:.4f}")

    return {
        "name": name,
        "n": len(results),
        "avg_len": avg_len,
        "std_len": std_len,
        "median_len": median_len,
        "min_len": min(lengths),
        "max_len": max(lengths),
        "avg_entropy": float(np.mean(entropies)),
        "avg_wood": float(np.mean(wood_collected)),
        "wood_3_count": sum(1 for w in wood_collected if w >= 3),
        "wood_3_rate": float(sum(1 for w in wood_collected if w >= 3) / len(results)),
        "avg_coal": float(np.mean(coal_collected)),
        "coal_1_count": sum(1 for c in coal_collected if c >= 1),
        "avg_iron": float(np.mean(iron_collected)),
        "causes": dict(causes),
        "residual_loss_mean": residual_loss_mean,
    }


# ---------------------------------------------------------------------------
# One full ablation (warmup A/B + eval)
# ---------------------------------------------------------------------------


def run_one_ablation(
    label: str,
    residual_enabled: bool,
    segmenter: Any,
    n_warmup_a: int,
    n_warmup_b: int,
    n_eval_runs: int,
    n_eval_eps: int,
    warmup_max_steps: int,
    eval_max_steps: int,
    horizon: int,
    residual_lr: float,
    device: torch.device,
) -> dict:
    print("\n" + "#" * 70)
    print(f"# Ablation: {label}  (residual_enabled={residual_enabled})")
    print("#" * 70)
    t_start = time.time()

    store, tracker, tb = _fresh_store_and_tracker()
    print(
        f"  Store: {len(store.concepts)} concepts, "
        f"{sum(len(c.causal_links) for c in store.concepts.values())} action rules, "
        f"{len(store.passive_rules)} passive rules"
    )
    print(
        f"  Tracker: vital={list(tracker.vital_mins.keys())} "
        f"innate={list(tracker.innate_rates.keys())}"
    )

    residual_predictor: ResidualBodyPredictor | None = None
    residual_optimizer: Any = None
    if residual_enabled:
        residual_predictor = ResidualBodyPredictor(ResidualConfig()).to(device)
        residual_optimizer = torch.optim.Adam(residual_predictor.parameters(), lr=residual_lr)
        print(
            f"  Residual: {sum(p.numel() for p in residual_predictor.parameters())} params, "
            f"device={device}, lr={residual_lr}"
        )

    # --- Phase 1: Warmup A (enemies off) ---
    print(f"\n  [Phase 1] Warmup A — {n_warmup_a} eps, enemies OFF, max_steps={warmup_max_steps}")
    warmup_a_results = _run_episodes(
        f"{label}/warmup_a",
        n_warmup_a,
        warmup_max_steps,
        segmenter,
        store,
        tracker,
        enemies=False,
        horizon=horizon,
        seed_offset=300,
        residual_predictor=residual_predictor,
        residual_optimizer=residual_optimizer,
        residual_train=residual_enabled,
    )
    warmup_a_summary = _summarize(f"{label}/warmup_a", warmup_a_results)

    if residual_enabled and residual_predictor is not None:
        ckpt = CHECKPOINT_DIR / f"residual_{label}_after_warmup_a.pt"
        residual_predictor.save_state(str(ckpt))
        print(f"  Checkpoint: {ckpt}")

    # --- Phase 2: Warmup B (enemies on) ---
    print(f"\n  [Phase 2] Warmup B — {n_warmup_b} eps, enemies ON, max_steps={warmup_max_steps}")
    warmup_b_results = _run_episodes(
        f"{label}/warmup_b",
        n_warmup_b,
        warmup_max_steps,
        segmenter,
        store,
        tracker,
        enemies=True,
        horizon=horizon,
        seed_offset=500,
        residual_predictor=residual_predictor,
        residual_optimizer=residual_optimizer,
        residual_train=residual_enabled,
    )
    warmup_b_summary = _summarize(f"{label}/warmup_b", warmup_b_results)

    if residual_enabled and residual_predictor is not None:
        ckpt = CHECKPOINT_DIR / f"residual_{label}_after_warmup_b.pt"
        residual_predictor.save_state(str(ckpt))
        print(f"  Checkpoint: {ckpt}")

    # --- Phase 3: Evaluation ---
    print(f"\n  [Phase 3] Eval — {n_eval_runs} runs × {n_eval_eps} eps, enemies ON, max_steps={eval_max_steps}")
    eval_summaries: list[dict] = []
    all_eval_results: list[dict] = []
    for run_idx in range(n_eval_runs):
        print(f"\n  --- Eval run {run_idx + 1}/{n_eval_runs} ---")
        run_results = _run_episodes(
            f"{label}/eval_run{run_idx}",
            n_eval_eps,
            eval_max_steps,
            segmenter,
            store,
            tracker,
            enemies=True,
            horizon=horizon,
            seed_offset=1000 + run_idx * 100,
            residual_predictor=residual_predictor,
            residual_optimizer=residual_optimizer,
            residual_train=residual_enabled,
        )
        eval_summaries.append(_summarize(f"{label}/eval_run{run_idx}", run_results))
        all_eval_results.extend(run_results)

    eval_combined = _summarize(f"{label}/eval_combined", all_eval_results)

    if residual_enabled and residual_predictor is not None:
        ckpt = CHECKPOINT_DIR / f"residual_{label}_after_eval.pt"
        residual_predictor.save_state(str(ckpt))
        print(f"  Checkpoint: {ckpt}")

    elapsed_min = (time.time() - t_start) / 60.0
    print(f"\n  [{label}] total time: {elapsed_min:.1f} min")

    return {
        "label": label,
        "residual_enabled": residual_enabled,
        "warmup_a": warmup_a_summary,
        "warmup_b": warmup_b_summary,
        "eval_runs": eval_summaries,
        "eval_combined": eval_combined,
        "elapsed_min": elapsed_min,
    }


# ---------------------------------------------------------------------------
# Final comparison: off vs on vs Stage 77a baseline
# ---------------------------------------------------------------------------


def print_comparison(off: dict, on: dict) -> dict:
    print("\n" + "=" * 70)
    print("Stage 78c — Ablation comparison")
    print("=" * 70)

    def _row(label: str, phase_key: str, baseline: float | None) -> None:
        off_avg = off[phase_key]["avg_len"]
        on_avg = on[phase_key]["avg_len"]
        off_w3 = off[phase_key].get("wood_3_rate", 0.0)
        on_w3 = on[phase_key].get("wood_3_rate", 0.0)
        delta_on_vs_off = on_avg - off_avg
        base_str = ""
        if baseline is not None:
            delta_on_vs_base = on_avg - baseline
            delta_off_vs_base = off_avg - baseline
            base_str = (
                f" | Stage77a baseline={baseline:.0f} "
                f"(off Δ={delta_off_vs_base:+.1f}, on Δ={delta_on_vs_base:+.1f})"
            )
        print(
            f"  {label:14s} off={off_avg:6.1f} on={on_avg:6.1f} "
            f"(Δon-off={delta_on_vs_off:+.1f}) "
            f"wood≥3 off={off_w3*100:4.1f}% on={on_w3*100:4.1f}%{base_str}"
        )

    _row("warmup_a", "warmup_a", STAGE77A_BASELINE["warmup_a_avg_len"])
    _row("warmup_b", "warmup_b", STAGE77A_BASELINE["warmup_b_avg_len"])
    _row("eval", "eval_combined", STAGE77A_BASELINE["eval_avg_len"])

    # Residual loss summary
    on_eval_rl = on["eval_combined"].get("residual_loss_mean")
    if on_eval_rl is not None:
        print(f"  residual_loss_mean (on, eval) = {on_eval_rl:.4f}")

    eval_off_avg = off["eval_combined"]["avg_len"]
    eval_on_avg = on["eval_combined"]["avg_len"]
    baseline_eval = STAGE77A_BASELINE["eval_avg_len"]

    gate_passes = {
        "residual_on_beats_off": eval_on_avg > eval_off_avg,
        "residual_on_ge_baseline": eval_on_avg >= baseline_eval,
        "residual_on_meaningful_delta": abs(eval_on_avg - eval_off_avg) >= 5.0,
    }
    print(f"\n  Gates: {gate_passes}")

    return {
        "delta_on_minus_off_eval": eval_on_avg - eval_off_avg,
        "delta_on_minus_baseline_eval": eval_on_avg - baseline_eval,
        "delta_off_minus_baseline_eval": eval_off_avg - baseline_eval,
        "gates": gate_passes,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> bool:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Smoke run: tiny episode counts for local sanity, not a real eval",
    )
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--residual_lr", type=float, default=1e-3)
    parser.add_argument(
        "--only",
        choices=["off", "on", "both"],
        default="both",
        help="Run just one side of the ablation (debug/iteration)",
    )
    args = parser.parse_args()

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    t_total = time.time()
    disable_rocm_conv()
    device = _pick_device()
    print(f"Device: {device}")

    # Phase 0 (shared): segmenter
    print("\n[Phase 0] Load Stage 75 segmenter")
    if not STAGE75_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Segmenter checkpoint missing at {STAGE75_CHECKPOINT}. "
            f"Run Stage 75 (experiments/exp135_grid8_tile_perception.py) first."
        )
    from snks.encoder.tile_segmenter import pick_device as _tseg_device
    seg_device = _tseg_device()
    segmenter = load_tile_segmenter(str(STAGE75_CHECKPOINT), device=seg_device)
    print(f"  Loaded {STAGE75_CHECKPOINT}, segmenter device={seg_device}")

    # Episode budgets
    if args.fast:
        n_warmup_a = 3
        n_warmup_b = 3
        n_eval_runs = 1
        n_eval_eps = 3
        warmup_max_steps = 150
        eval_max_steps = 200
    else:
        n_warmup_a = 50
        n_warmup_b = 50
        n_eval_runs = 3
        n_eval_eps = 20
        warmup_max_steps = 500
        eval_max_steps = 1000

    print(
        f"Budget: warmup_a={n_warmup_a} warmup_b={n_warmup_b} "
        f"eval={n_eval_runs}×{n_eval_eps}"
    )
    print(f"Max steps: warmup={warmup_max_steps} eval={eval_max_steps}")

    report: dict[str, Any] = {
        "args": vars(args),
        "device": str(device),
        "budget": {
            "n_warmup_a": n_warmup_a,
            "n_warmup_b": n_warmup_b,
            "n_eval_runs": n_eval_runs,
            "n_eval_eps": n_eval_eps,
            "warmup_max_steps": warmup_max_steps,
            "eval_max_steps": eval_max_steps,
            "horizon": args.horizon,
        },
        "stage77a_baseline": STAGE77A_BASELINE,
    }

    off_report: dict | None = None
    on_report: dict | None = None

    if args.only in ("off", "both"):
        off_report = run_one_ablation(
            label="residual_off",
            residual_enabled=False,
            segmenter=segmenter,
            n_warmup_a=n_warmup_a,
            n_warmup_b=n_warmup_b,
            n_eval_runs=n_eval_runs,
            n_eval_eps=n_eval_eps,
            warmup_max_steps=warmup_max_steps,
            eval_max_steps=eval_max_steps,
            horizon=args.horizon,
            residual_lr=args.residual_lr,
            device=device,
        )
        report["residual_off"] = off_report

    if args.only in ("on", "both"):
        on_report = run_one_ablation(
            label="residual_on",
            residual_enabled=True,
            segmenter=segmenter,
            n_warmup_a=n_warmup_a,
            n_warmup_b=n_warmup_b,
            n_eval_runs=n_eval_runs,
            n_eval_eps=n_eval_eps,
            warmup_max_steps=warmup_max_steps,
            eval_max_steps=eval_max_steps,
            horizon=args.horizon,
            residual_lr=args.residual_lr,
            device=device,
        )
        report["residual_on"] = on_report

    if off_report is not None and on_report is not None:
        report["comparison"] = print_comparison(off_report, on_report)

    report["total_time_min"] = (time.time() - t_total) / 60.0
    print(f"\nTotal time: {report['total_time_min']:.1f} min")

    with open(RESULTS_PATH, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Results saved: {RESULTS_PATH}")

    if "comparison" in report:
        gates = report["comparison"]["gates"]
        return gates["residual_on_beats_off"] or gates["residual_on_meaningful_delta"]
    return True


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
