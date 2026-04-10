"""Stage 78b — MLP residual on the Stage 78a synthetic task.

Validates that a small MLP residual over a rules baseline can learn the
conjunctive rule the textbook does not contain. This is the unit test of
the residual-learner pattern Strategy v5 Branch A will use in Stage 80
for Crafter integration.

Setup:
  - Same dataset as Stage 78a (reused via import from stage78a_daf_spike_fair).
  - `textbook_rules_predict` = true rules MINUS the conjunctive sleep case.
  - Residual trained on `actual_delta - textbook_delta`.
  - Final prediction = textbook + residual.

Gate (Stage 78b unit):
  - conj_health_mse ≤ 0.0080 (Stage 78a baseline floor is 0.0072)
  - gen_health_mse  ≤ 0.0120 (Stage 78a baseline 0.0106)

Run locally on CPU (a few seconds — this is a unit test, not an
experiment; same dataset and model size as Stage 78a baseline which
ran in <2 s on GPU). On minipc for consistency with 78a run:

    ./scripts/minipc-run.sh stage78b "from stage78b_residual_synthetic import main; main()"

Or directly:

    .venv/bin/python experiments/stage78b_residual_synthetic.py
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

# Reuse Stage 78a synthetic task so the numerical comparison is apples-to-apples
from stage78a_daf_spike_fair import (
    ACTIONS,
    BODY_VARS,
    Sample,
    conjunctive_dataset,
    generate_dataset,
    true_body_delta,
)

from snks.learning.residual_predictor import ResidualBodyPredictor, ResidualConfig


# ---------------------------------------------------------------------------
# Textbook (rules WITHOUT the conjunctive case)
# ---------------------------------------------------------------------------


def textbook_rules_predict(
    visible: set[str], body: dict[str, float], action: str,
) -> dict[str, float]:
    """The textbook the residual sits on top of.

    Identical to stage78a_daf_spike_fair.true_body_delta except it does
    NOT know that sleep+(food==0 or drink==0) is harmful — the textbook
    applies the benign sleep rule unconditionally. The residual's job is
    to learn the correction for the conjunctive case without breaking
    the rest.
    """
    delta = {v: 0.0 for v in BODY_VARS}
    delta["food"] -= 0.04
    delta["drink"] -= 0.04
    delta["energy"] -= 0.02

    if action == "sleep":
        # Textbook: sleep is always beneficial. Missing the conjunctive branch.
        delta["energy"] += 0.2
        delta["health"] += 0.04

    if "skeleton" in visible:
        delta["health"] -= 0.4
    if "zombie" in visible:
        delta["health"] -= 0.5

    if action == "do_cow" and "cow" in visible:
        delta["food"] += 5.0
    if action == "do_water" and "water" in visible:
        delta["drink"] += 5.0

    return delta


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def sample_to_tensors(
    predictor: ResidualBodyPredictor,
    sample: Sample,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode a single sample into (fp, rules_delta, target_delta) tensors."""
    action_idx = ACTIONS.index(sample.action)
    fp = predictor.encode(sample.visible, sample.body, action_idx, device=device)
    rules = textbook_rules_predict(sample.visible, sample.body, sample.action)
    rules_t = torch.tensor([rules[v] for v in BODY_VARS], dtype=torch.float32, device=device)
    target_t = torch.tensor([sample.delta[v] for v in BODY_VARS], dtype=torch.float32, device=device)
    return fp, rules_t, target_t


def train_epoch(
    predictor: ResidualBodyPredictor,
    optimizer: torch.optim.Optimizer,
    samples: list[Sample],
    rng: np.random.RandomState,
    device: torch.device,
) -> float:
    predictor.train()
    losses: list[float] = []
    order = rng.permutation(len(samples))
    for idx in order:
        fp, rules_t, target_t = sample_to_tensors(predictor, samples[int(idx)], device)
        loss = predictor.residual_loss(fp, rules_t, target_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses))


def evaluate(
    predictor: ResidualBodyPredictor,
    samples: list[Sample],
    device: torch.device,
) -> dict[str, float | dict[str, float]]:
    predictor.eval()
    sq = {v: [] for v in BODY_VARS}
    residual_mags = {v: [] for v in BODY_VARS}
    with torch.no_grad():
        for s in samples:
            fp, rules_t, target_t = sample_to_tensors(predictor, s, device)
            residual = predictor.forward(fp)
            pred = rules_t + residual
            err = (pred - target_t).pow(2)
            for i, v in enumerate(BODY_VARS):
                sq[v].append(float(err[i].item()))
                residual_mags[v].append(float(residual[i].abs().item()))
    return {
        "overall_mse": float(np.mean([e for errs in sq.values() for e in errs])),
        "per_var_mse": {v: float(np.mean(es)) for v, es in sq.items()},
        "residual_abs_mean": {v: float(np.mean(m)) for v, m in residual_mags.items()},
    }


# ---------------------------------------------------------------------------
# Baselines for reference
# ---------------------------------------------------------------------------


def evaluate_rules_only(samples: list[Sample]) -> dict:
    """Just the textbook with no residual — the floor from which the residual improves."""
    sq = {v: [] for v in BODY_VARS}
    for s in samples:
        rules = textbook_rules_predict(s.visible, s.body, s.action)
        for v in BODY_VARS:
            sq[v].append((rules[v] - s.delta[v]) ** 2)
    return {
        "overall_mse": float(np.mean([e for errs in sq.values() for e in errs])),
        "per_var_mse": {v: float(np.mean(es)) for v, es in sq.items()},
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@dataclass
class GateResult:
    passed: bool
    conj_health_mse: float
    conj_health_gate: float
    gen_health_mse: float
    gen_health_gate: float


def check_gate(
    eval_general: dict, eval_conj: dict,
    conj_health_gate: float = 0.008, gen_health_gate: float = 0.012,
) -> GateResult:
    ch = eval_conj["per_var_mse"]["health"]
    gh = eval_general["per_var_mse"]["health"]
    return GateResult(
        passed=(ch <= conj_health_gate) and (gh <= gen_health_gate),
        conj_health_mse=ch,
        conj_health_gate=conj_health_gate,
        gen_health_mse=gh,
        gen_health_gate=gen_health_gate,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-train", type=int, default=1200)
    parser.add_argument("--n-test-general", type=int, default=300)
    parser.add_argument("--n-test-conj", type=int, default=200)
    parser.add_argument("--n-epochs", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output", type=str, default="_docs/stage78b_results.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device(args.device if args.device != "auto" else (
        "cuda" if torch.cuda.is_available() else "cpu"
    ))

    print("=" * 72)
    print("Stage 78b — MLP residual over textbook on Stage 78a synthetic task")
    print(f"  n_train={args.n_train} test_general={args.n_test_general} "
          f"test_conj={args.n_test_conj} epochs={args.n_epochs} hidden={args.hidden_dim}")
    print(f"  device={device}")
    print("=" * 72, flush=True)

    rng = np.random.RandomState(args.seed)
    train = generate_dataset(args.n_train, rng)
    test_general = generate_dataset(args.n_test_general, rng)
    test_conj = conjunctive_dataset(args.n_test_conj, rng)

    # Rules-only reference (the residual must improve on this)
    print("\n--- Rules-only baseline (textbook without conjunctive knowledge) ---")
    rules_gen = evaluate_rules_only(test_general)
    rules_conj = evaluate_rules_only(test_conj)
    print(f"  general overall_mse: {rules_gen['overall_mse']:.4f}")
    print(f"  general per_var: {rules_gen['per_var_mse']}")
    print(f"  conj    overall_mse: {rules_conj['overall_mse']:.4f}")
    print(f"  conj    per_var: {rules_conj['per_var_mse']}")
    print(f"  conj HEALTH mse: {rules_conj['per_var_mse']['health']:.4f}  "
          f"(this is what the residual must fix)", flush=True)

    # Build residual predictor
    print("\n--- Building residual predictor ---")
    cfg = ResidualConfig(
        n_visible_concepts=9,
        n_actions=len(ACTIONS),
        n_body_vars=len(BODY_VARS),
        body_buckets=10,
        hidden_dim=args.hidden_dim,
        concept_hash_active=30,
    )
    predictor = ResidualBodyPredictor(cfg).to(device)
    n_params = sum(p.numel() for p in predictor.parameters())
    print(f"  input_dim={predictor.input_dim} hidden={cfg.hidden_dim} "
          f"output={predictor.output_dim} n_params={n_params}", flush=True)

    optimizer = torch.optim.Adam(predictor.parameters(), lr=args.lr)

    # BEFORE training
    print("\n--- BEFORE training ---")
    ev_gen_before = evaluate(predictor, test_general, device)
    ev_con_before = evaluate(predictor, test_conj, device)
    print(f"  general health mse: {ev_gen_before['per_var_mse']['health']:.4f}")
    print(f"  conj    health mse: {ev_con_before['per_var_mse']['health']:.4f}")

    # Training loop
    print(f"\n--- TRAINING ({args.n_epochs} epochs) ---", flush=True)
    loss_curve: list[float] = []
    t0 = time.time()
    for epoch in range(args.n_epochs):
        epoch_loss = train_epoch(predictor, optimizer, train, rng, device)
        loss_curve.append(epoch_loss)
        elapsed = time.time() - t0
        if epoch % max(1, args.n_epochs // 10) == 0 or epoch == args.n_epochs - 1:
            ev_con_now = evaluate(predictor, test_conj, device)
            print(f"  epoch {epoch+1}/{args.n_epochs} train_mse={epoch_loss:.4f} "
                  f"conj_health_mse={ev_con_now['per_var_mse']['health']:.4f} "
                  f"elapsed={elapsed:.0f}s", flush=True)

    # AFTER training
    print("\n--- AFTER training ---")
    ev_gen = evaluate(predictor, test_general, device)
    ev_con = evaluate(predictor, test_conj, device)
    print(f"  general: overall_mse={ev_gen['overall_mse']:.4f} "
          f"per_var={ev_gen['per_var_mse']}")
    print(f"  conj:    overall_mse={ev_con['overall_mse']:.4f} "
          f"per_var={ev_con['per_var_mse']}")
    print(f"  conj HEALTH mse: {ev_con['per_var_mse']['health']:.4f}")
    print(f"  residual abs means: {ev_con['residual_abs_mean']}", flush=True)

    # Gate check
    gate = check_gate(ev_gen, ev_con)
    print("\n--- GATE ---")
    print(f"  conj_health_mse={gate.conj_health_mse:.4f}  "
          f"(gate ≤ {gate.conj_health_gate}) — "
          f"{'PASS' if gate.conj_health_mse <= gate.conj_health_gate else 'FAIL'}")
    print(f"  gen_health_mse={gate.gen_health_mse:.4f}  "
          f"(gate ≤ {gate.gen_health_gate}) — "
          f"{'PASS' if gate.gen_health_mse <= gate.gen_health_gate else 'FAIL'}")
    print(f"\n  STAGE 78b: {'PASS' if gate.passed else 'FAIL'}", flush=True)

    # Save results
    results = {
        "args": vars(args),
        "rules_only": {"general": rules_gen, "conj": rules_conj},
        "eval_general_before": ev_gen_before,
        "eval_conj_before": ev_con_before,
        "eval_general": ev_gen,
        "eval_conj": ev_con,
        "loss_curve": loss_curve,
        "gate": asdict(gate),
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  results → {out_path}")

    return 0 if gate.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
