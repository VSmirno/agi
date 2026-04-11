"""Stage 79 — Surprise Accumulator + Rule Nursery on Crafter.

After Stage 78c partial-FAIL (residual_on hurts on enemies-on phases
because the encoding can only learn averaged corrections over
fingerprint collisions), Stage 79 adds explicit symbolic rule
induction. Each surprise observation feeds an L2/L1 bucket store; the
nursery emits candidate rules from saturated buckets, verifies them
over a held-out window, and promotes survivors to
`ConceptStore.learned_rules`. Promoted rules fire in Phase 7 of
`_apply_tick` for any subsequent simulate_forward call in the same
episode.

The synthetic conjunctive falsification test
(`tests/learning/test_nursery_synthetic_conjunctive.py`) already
showed the nursery can induce the canonical conjunctive sleep+
starvation rule from 500 surprise observations. This harness tests
whether the same mechanism, on real env rollouts, beats Stage 78c's
residual approach.

Pipeline per ablation
---------------------
  Phase 0 (shared): Load Stage 75 segmenter checkpoint
  Phase 1: Warmup A — 50 eps, enemies OFF
  Phase 2: Warmup B — 50 eps, enemies ON
  Phase 3: Eval — 3 runs × 20 episodes, enemies ON, max_steps=1000

Three ablations
---------------
  - nursery_off                 (baseline, equivalent to Stage 78c residual_off)
  - nursery_on                  (residual off, nursery on — headline test)
  - nursery_on_residual_on      (composition test, optional)

Output
------
  _docs/stage79_results.json  — full per-phase / per-run summary +
                                 nursery + accumulator stats
  _docs/stage79_learned_rules.jsonl  — one promoted rule per line for
                                        diagnostic inspection

Usage
-----
  python experiments/stage79_nursery_crafter.py            # full
  python experiments/stage79_nursery_crafter.py --fast     # smoke

  Or via minipc launcher:
    ./scripts/minipc-run.sh stage79_smoke "from stage79_nursery_crafter import main_fast; main_fast()"
    ./scripts/minipc-run.sh stage79       "from stage79_nursery_crafter import main_full; main_full()"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import asdict
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
from snks.learning.rule_nursery import RuleNursery
from snks.learning.surprise_accumulator import SurpriseAccumulator


STAGE75_CHECKPOINT = Path("demos/checkpoints/exp135/segmenter_9x9.pt")
RESULTS_PATH = Path("_docs/stage79_results.json")
LEARNED_RULES_PATH = Path("_docs/stage79_learned_rules.jsonl")

# Stage 77a Run 8 + Stage 78c reference numbers (for comparison)
STAGE77A_BASELINE = {
    "warmup_a_avg_len": 222.0,
    "warmup_b_avg_len": 203.0,
    "eval_avg_len": 180.0,
}
STAGE78C_REFERENCE = {
    "warmup_a_off": 215.9,
    "warmup_a_on": 259.5,
    "warmup_b_off": 182.3,
    "warmup_b_on": 159.8,
    "eval_off": 169.2,
    "eval_on": 152.1,
}


# ---------------------------------------------------------------------------
# Helpers
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


def _fresh_store_and_tracker() -> tuple[ConceptStore, HomeostaticTracker]:
    """Each ablation gets independent store + tracker so observation-driven
    learning does not leak between ablations. Same convention as Stage 78c."""
    store = ConceptStore()
    tb = CrafterTextbook("configs/crafter_textbook.yaml")
    tb.load_into(store)
    tracker = HomeostaticTracker()
    tracker.init_from_textbook(tb.body_block, store.passive_rules)
    return store, tracker


def _serialize_learned_rule(rule: Any) -> dict:
    """Convert a LearnedRule into a JSON-friendly dict for the dump."""
    return {
        "precondition": {
            "visible": sorted(rule.precondition.visible),
            "body_quartiles": list(rule.precondition.body_quartiles),
            "action": rule.precondition.action,
        },
        "effect": dict(rule.effect),
        "confidence": float(rule.confidence),
        "n_observations": int(rule.n_observations),
        "source": rule.source,
    }


# ---------------------------------------------------------------------------
# Episode runner — wraps run_mpc_episode with nursery threading
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
    surprise_accumulator: SurpriseAccumulator | None,
    rule_nursery: RuleNursery | None,
    residual_predictor: ResidualBodyPredictor | None,
    residual_optimizer: Any,
    residual_train: bool,
) -> list[dict]:
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
            surprise_accumulator=surprise_accumulator,
            rule_nursery=rule_nursery,
        )
        result["episode"] = ep
        results.append(result)

        if ep < 3 or ep % 10 == 0 or ep == n_episodes - 1:
            inv = result["final_inv"]
            elapsed = time.time() - t0
            eta_min = elapsed / (ep + 1) * (n_episodes - ep - 1) / 60.0
            ns = result.get("nursery_stats", {})
            ns_str = (
                f" rules={result.get('learned_rule_count', 0)} "
                f"ns_e={ns.get('emitted', 0)}p{ns.get('promoted', 0)}r{ns.get('rejected', 0)}"
                if rule_nursery is not None else ""
            )
            print(
                f"  [{name}] ep{ep:3d}/{n_episodes} len={result['length']:4d} "
                f"H{inv.get('health', 0)}F{inv.get('food', 0)}"
                f"D{inv.get('drink', 0)}E{inv.get('energy', 0)} "
                f"W{inv.get('wood', 0)} cause={result['cause_of_death']}"
                f"{ns_str} ETA={eta_min:.1f}m"
            )
    return results


def _summarize(name: str, results: list[dict]) -> dict:
    lengths = [r["length"] for r in results]
    causes: Counter = Counter(r["cause_of_death"] for r in results)
    entropies = [r["action_entropy"] for r in results]
    wood = [r["final_inv"].get("wood", 0) for r in results]
    coal = [r["final_inv"].get("coal", 0) for r in results]
    iron = [r["final_inv"].get("iron", 0) for r in results]
    avg_len = float(np.mean(lengths))
    median_len = int(np.median(lengths))
    std_len = float(np.std(lengths))

    final_rule_counts = [r.get("learned_rule_count", 0) for r in results]
    nursery_emitted = [r.get("nursery_stats", {}).get("emitted", 0) for r in results]
    nursery_promoted = [r.get("nursery_stats", {}).get("promoted", 0) for r in results]
    nursery_rejected = [r.get("nursery_stats", {}).get("rejected", 0) for r in results]

    print(
        f"\n  [{name}] n={len(results)} avg_len={avg_len:.1f}±{std_len:.1f} "
        f"median={median_len} min={min(lengths)} max={max(lengths)}"
    )
    print(f"  [{name}] avg_entropy={np.mean(entropies):.2f}")
    print(f"  [{name}] causes: {dict(causes)}")
    print(
        f"  [{name}] wood avg={np.mean(wood):.2f} ≥3:{sum(1 for w in wood if w >= 3)}/{len(results)}"
        f"  coal avg={np.mean(coal):.2f}  iron avg={np.mean(iron):.2f}"
    )
    if any(c > 0 for c in final_rule_counts):
        print(
            f"  [{name}] nursery: rules_per_ep_mean={np.mean(final_rule_counts):.2f}"
            f"  emitted/ep={np.mean(nursery_emitted):.2f}"
            f"  promoted/ep={np.mean(nursery_promoted):.2f}"
            f"  rejected/ep={np.mean(nursery_rejected):.2f}"
        )

    return {
        "name": name,
        "n": len(results),
        "avg_len": avg_len,
        "std_len": std_len,
        "median_len": median_len,
        "min_len": min(lengths),
        "max_len": max(lengths),
        "avg_entropy": float(np.mean(entropies)),
        "avg_wood": float(np.mean(wood)),
        "wood_3_count": sum(1 for w in wood if w >= 3),
        "wood_3_rate": float(sum(1 for w in wood if w >= 3) / len(results)),
        "avg_coal": float(np.mean(coal)),
        "coal_1_count": sum(1 for c in coal if c >= 1),
        "avg_iron": float(np.mean(iron)),
        "causes": dict(causes),
        "rules_per_ep_mean": float(np.mean(final_rule_counts)) if final_rule_counts else 0.0,
        "nursery_emitted_mean": float(np.mean(nursery_emitted)),
        "nursery_promoted_mean": float(np.mean(nursery_promoted)),
        "nursery_rejected_mean": float(np.mean(nursery_rejected)),
    }


# ---------------------------------------------------------------------------
# One full ablation
# ---------------------------------------------------------------------------


def run_one_ablation(
    label: str,
    nursery_enabled: bool,
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
    print(f"# Ablation: {label}  (nursery={nursery_enabled}, residual={residual_enabled})")
    print("#" * 70)
    t_start = time.time()

    store, tracker = _fresh_store_and_tracker()
    print(
        f"  Store: {len(store.concepts)} concepts, "
        f"{sum(len(c.causal_links) for c in store.concepts.values())} action rules, "
        f"{len(store.passive_rules)} passive rules"
    )

    # Stage 79 components
    accumulator: SurpriseAccumulator | None = None
    nursery: RuleNursery | None = None
    if nursery_enabled:
        accumulator = SurpriseAccumulator()
        nursery = RuleNursery()
        print("  Stage 79: SurpriseAccumulator + RuleNursery enabled")

    # Stage 78c residual (optional)
    residual_predictor: ResidualBodyPredictor | None = None
    residual_optimizer: Any = None
    if residual_enabled:
        residual_predictor = ResidualBodyPredictor(ResidualConfig()).to(device)
        residual_optimizer = torch.optim.Adam(residual_predictor.parameters(), lr=residual_lr)
        print(f"  Stage 78c residual: {sum(p.numel() for p in residual_predictor.parameters())} params, lr={residual_lr}")

    common_kwargs = dict(
        segmenter=segmenter,
        store=store,
        tracker=tracker,
        horizon=horizon,
        surprise_accumulator=accumulator,
        rule_nursery=nursery,
        residual_predictor=residual_predictor,
        residual_optimizer=residual_optimizer,
        residual_train=residual_enabled,
    )

    print(f"\n  [Phase 1] Warmup A — {n_warmup_a} eps, enemies OFF, max_steps={warmup_max_steps}")
    warmup_a_results = _run_episodes(
        f"{label}/warmup_a", n_warmup_a, warmup_max_steps,
        enemies=False, seed_offset=300, **common_kwargs,
    )
    warmup_a_summary = _summarize(f"{label}/warmup_a", warmup_a_results)

    print(f"\n  [Phase 2] Warmup B — {n_warmup_b} eps, enemies ON, max_steps={warmup_max_steps}")
    warmup_b_results = _run_episodes(
        f"{label}/warmup_b", n_warmup_b, warmup_max_steps,
        enemies=True, seed_offset=500, **common_kwargs,
    )
    warmup_b_summary = _summarize(f"{label}/warmup_b", warmup_b_results)

    print(f"\n  [Phase 3] Eval — {n_eval_runs} runs × {n_eval_eps} eps, enemies ON, max_steps={eval_max_steps}")
    eval_summaries: list[dict] = []
    all_eval_results: list[dict] = []
    for run_idx in range(n_eval_runs):
        print(f"\n  --- Eval run {run_idx + 1}/{n_eval_runs} ---")
        run_results = _run_episodes(
            f"{label}/eval_run{run_idx}", n_eval_eps, eval_max_steps,
            enemies=True, seed_offset=1000 + run_idx * 100, **common_kwargs,
        )
        eval_summaries.append(_summarize(f"{label}/eval_run{run_idx}", run_results))
        all_eval_results.extend(run_results)

    eval_combined = _summarize(f"{label}/eval_combined", all_eval_results)

    # Dump learned rules to JSONL for diagnostic inspection
    learned_dump_path = Path(f"_docs/stage79_learned_rules_{label}.jsonl")
    learned_dump_path.parent.mkdir(parents=True, exist_ok=True)
    with learned_dump_path.open("w") as f:
        for rule in store.learned_rules:
            f.write(json.dumps(_serialize_learned_rule(rule)) + "\n")
    print(f"  Learned rules dumped: {learned_dump_path} (n={len(store.learned_rules)})")

    elapsed_min = (time.time() - t_start) / 60.0
    print(f"\n  [{label}] total time: {elapsed_min:.1f} min, "
          f"final learned_rules count: {len(store.learned_rules)}")

    return {
        "label": label,
        "nursery_enabled": nursery_enabled,
        "residual_enabled": residual_enabled,
        "warmup_a": warmup_a_summary,
        "warmup_b": warmup_b_summary,
        "eval_runs": eval_summaries,
        "eval_combined": eval_combined,
        "elapsed_min": elapsed_min,
        "final_learned_rule_count": len(store.learned_rules),
    }


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def print_comparison(off: dict, on: dict, on_with_residual: dict | None) -> dict:
    print("\n" + "=" * 80)
    print("Stage 79 — Ablation comparison")
    print("=" * 80)

    def _fmt(label: str, phase_key: str, baseline_77a: float | None) -> None:
        off_avg = off[phase_key]["avg_len"]
        on_avg = on[phase_key]["avg_len"]
        delta_on_off = on_avg - off_avg
        b_str = ""
        if baseline_77a is not None:
            b_str = f" | 77a={baseline_77a:.0f} (off Δ={off_avg-baseline_77a:+.1f}, on Δ={on_avg-baseline_77a:+.1f})"
        if on_with_residual is not None:
            both_avg = on_with_residual[phase_key]["avg_len"]
            print(
                f"  {label:14s} off={off_avg:6.1f}  on={on_avg:6.1f}  on+res={both_avg:6.1f}"
                f"  Δ(on-off)={delta_on_off:+.1f}{b_str}"
            )
        else:
            print(
                f"  {label:14s} off={off_avg:6.1f}  on={on_avg:6.1f}"
                f"  Δ(on-off)={delta_on_off:+.1f}{b_str}"
            )

    _fmt("warmup_a", "warmup_a", STAGE77A_BASELINE["warmup_a_avg_len"])
    _fmt("warmup_b", "warmup_b", STAGE77A_BASELINE["warmup_b_avg_len"])
    _fmt("eval", "eval_combined", STAGE77A_BASELINE["eval_avg_len"])

    print("\n  Action entropy (smoking-gun proxy from Stage 78c):")
    print(
        f"    off: warmup_a {off['warmup_a']['avg_entropy']:.2f}  "
        f"warmup_b {off['warmup_b']['avg_entropy']:.2f}  "
        f"eval {off['eval_combined']['avg_entropy']:.2f}"
    )
    print(
        f"    on:  warmup_a {on['warmup_a']['avg_entropy']:.2f}  "
        f"warmup_b {on['warmup_b']['avg_entropy']:.2f}  "
        f"eval {on['eval_combined']['avg_entropy']:.2f}"
    )

    print(
        f"\n  Nursery (on, eval): "
        f"final_rules={on['final_learned_rule_count']}, "
        f"rules/ep_mean={on['eval_combined']['rules_per_ep_mean']:.2f}, "
        f"emitted/ep={on['eval_combined']['nursery_emitted_mean']:.2f}, "
        f"promoted/ep={on['eval_combined']['nursery_promoted_mean']:.2f}"
    )

    eval_off = off["eval_combined"]["avg_len"]
    eval_on = on["eval_combined"]["avg_len"]
    baseline_eval = STAGE77A_BASELINE["eval_avg_len"]

    gates = {
        "nursery_on_beats_off": eval_on > eval_off,
        "nursery_on_ge_77a_baseline": eval_on >= baseline_eval,
        "entropy_not_collapsed": (
            on["eval_combined"]["avg_entropy"]
            >= off["eval_combined"]["avg_entropy"] - 0.1
        ),
        "at_least_one_rule_per_ep": on["eval_combined"]["rules_per_ep_mean"] >= 1.0,
        "wood_3_at_least_5pct": on["eval_combined"]["wood_3_count"] >= 3,
    }
    print(f"\n  Gates: {gates}")

    return {
        "delta_on_minus_off_eval": eval_on - eval_off,
        "delta_on_minus_77a_eval": eval_on - baseline_eval,
        "gates": gates,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(
    fast: bool = False,
    horizon: int = 20,
    residual_lr: float = 1e-3,
    only: str = "both",
    test_residual_combo: bool = False,
) -> bool:
    """Entry point. Defaults run a full eval; pass fast=True for smoke."""
    if fast is False and horizon == 20 and residual_lr == 1e-3 and only == "both" and not test_residual_combo:
        if len(sys.argv) > 1:
            parser = argparse.ArgumentParser()
            parser.add_argument("--fast", action="store_true")
            parser.add_argument("--horizon", type=int, default=20)
            parser.add_argument("--residual_lr", type=float, default=1e-3)
            parser.add_argument("--only", choices=["off", "on", "both"], default="both")
            parser.add_argument("--with_residual", action="store_true")
            ns = parser.parse_args()
            fast = ns.fast
            horizon = ns.horizon
            residual_lr = ns.residual_lr
            only = ns.only
            test_residual_combo = ns.with_residual

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    t_total = time.time()
    disable_rocm_conv()
    device = _pick_device()
    print(f"Device: {device}")

    print("\n[Phase 0] Load Stage 75 segmenter")
    if not STAGE75_CHECKPOINT.exists():
        raise FileNotFoundError(f"Segmenter checkpoint missing at {STAGE75_CHECKPOINT}")
    from snks.encoder.tile_segmenter import pick_device as _seg_dev
    seg_device = _seg_dev()
    segmenter = load_tile_segmenter(str(STAGE75_CHECKPOINT), device=seg_device)
    print(f"  Loaded {STAGE75_CHECKPOINT}, segmenter device={seg_device}")

    if fast:
        n_warmup_a, n_warmup_b = 3, 3
        n_eval_runs, n_eval_eps = 1, 3
        warmup_max, eval_max = 150, 200
    else:
        n_warmup_a, n_warmup_b = 50, 50
        n_eval_runs, n_eval_eps = 3, 20
        warmup_max, eval_max = 500, 1000

    print(f"Budget: warmup_a={n_warmup_a} warmup_b={n_warmup_b} eval={n_eval_runs}×{n_eval_eps}")
    print(f"Max steps: warmup={warmup_max} eval={eval_max}")

    report: dict[str, Any] = {
        "args": {"fast": fast, "horizon": horizon, "residual_lr": residual_lr, "only": only,
                 "test_residual_combo": test_residual_combo},
        "device": str(device),
        "budget": {
            "n_warmup_a": n_warmup_a, "n_warmup_b": n_warmup_b,
            "n_eval_runs": n_eval_runs, "n_eval_eps": n_eval_eps,
            "warmup_max_steps": warmup_max, "eval_max_steps": eval_max,
            "horizon": horizon,
        },
        "stage77a_baseline": STAGE77A_BASELINE,
        "stage78c_reference": STAGE78C_REFERENCE,
    }

    off_report: dict | None = None
    on_report: dict | None = None
    on_residual_report: dict | None = None

    common = dict(
        segmenter=segmenter,
        n_warmup_a=n_warmup_a, n_warmup_b=n_warmup_b,
        n_eval_runs=n_eval_runs, n_eval_eps=n_eval_eps,
        warmup_max_steps=warmup_max, eval_max_steps=eval_max,
        horizon=horizon, residual_lr=residual_lr, device=device,
    )

    if only in ("off", "both"):
        off_report = run_one_ablation(
            label="nursery_off", nursery_enabled=False, residual_enabled=False, **common,
        )
        report["nursery_off"] = off_report

    if only in ("on", "both"):
        on_report = run_one_ablation(
            label="nursery_on", nursery_enabled=True, residual_enabled=False, **common,
        )
        report["nursery_on"] = on_report

    if test_residual_combo and only == "both":
        on_residual_report = run_one_ablation(
            label="nursery_on_residual_on", nursery_enabled=True, residual_enabled=True, **common,
        )
        report["nursery_on_residual_on"] = on_residual_report

    if off_report is not None and on_report is not None:
        report["comparison"] = print_comparison(off_report, on_report, on_residual_report)

    report["total_time_min"] = (time.time() - t_total) / 60.0
    print(f"\nTotal time: {report['total_time_min']:.1f} min")

    with open(RESULTS_PATH, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Results saved: {RESULTS_PATH}")

    if "comparison" in report:
        gates = report["comparison"]["gates"]
        return gates["nursery_on_beats_off"]
    return True


def main_fast() -> bool:
    return main(fast=True)


def main_full() -> bool:
    return main(fast=False)


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
