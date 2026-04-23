"""Train Stage 90R local action evaluator on state-centered comparison data."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from stage90_quick_slice import _json_default

ROOT = Path(__file__).parent.parent
DOCS_DIR = ROOT / "_docs"
DEFAULT_DATASET_PATH = DOCS_DIR / "stage90r_local_dataset.json"
DEFAULT_CKPT_PATH = DOCS_DIR / "stage90r_local_evaluator.pt"
DEFAULT_EVAL_PATH = DOCS_DIR / "stage90r_local_evaluator_eval.json"


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _build_config(metadata: dict[str, Any]):
    from snks.agent.stage90r_local_model import LocalEvaluatorConfig

    return LocalEvaluatorConfig(
        viewport_rows=int(metadata["viewport_rows"]),
        viewport_cols=int(metadata["viewport_cols"]),
        n_classes=len(metadata["near_classes"]),
        n_body=len(metadata["body_keys"]),
        n_inventory=len(metadata["inventory_keys"]),
        n_actions=len(metadata["action_names"]),
        belief_state_dim=len(
            metadata.get("belief_state_feature_names", metadata.get("temporal_feature_names", []))
        ),
    )


def _run_epoch(model, loader, optimizer, device: torch.device) -> dict[str, float]:
    from torch.nn import functional as F

    from snks.agent.stage90r_local_model import masked_mse

    model.train(optimizer is not None)
    totals = {
        "loss": 0.0,
        "damage_mse": 0.0,
        "resource_mse": 0.0,
        "survival_bce": 0.0,
        "escape_mse": 0.0,
        "progress_mse": 0.0,
        "stall_bce": 0.0,
        "affordance_bce": 0.0,
        "threat_trend_mse": 0.0,
        "survival_acc": 0.0,
    }
    n_batches = 0

    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        preds = model(
            batch["class_ids"],
            batch["confidences"],
            batch["body"],
            batch["inventory"],
            batch["action"],
            batch["temporal"],
        )
        damage_mse = masked_mse(preds["pred_damage"], batch["damage"])
        resource_mse = masked_mse(preds["pred_resource_gain"], batch["resource_gain"])
        survival_bce = F.binary_cross_entropy_with_logits(
            preds["pred_survival_logit"],
            batch["survived"],
        )
        escape_mse = masked_mse(
            preds["pred_escape_delta"],
            batch["escape_delta"],
            batch["escape_mask"],
        )
        progress_mse = masked_mse(preds["pred_progress_delta"], batch["progress_delta"])
        stall_bce = F.binary_cross_entropy_with_logits(
            preds["pred_stall_risk_logit"],
            batch["stall_risk"],
        )
        affordance_bce = F.binary_cross_entropy_with_logits(
            preds["pred_affordance_persistence_logit"],
            batch["affordance_persistence"],
        )
        threat_trend_mse = masked_mse(preds["pred_threat_trend"], batch["threat_trend"])
        loss = (
            damage_mse
            + 0.5 * resource_mse
            + survival_bce
            + 0.25 * escape_mse
            + 0.5 * progress_mse
            + 0.35 * stall_bce
            + 0.25 * affordance_bce
            + 0.35 * threat_trend_mse
        )

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        survival_pred = (torch.sigmoid(preds["pred_survival_logit"]) >= 0.5).float()
        survival_target = (batch["survived"] >= 0.5).float()
        survival_acc = torch.mean((survival_pred == survival_target).float())

        totals["loss"] += float(loss.item())
        totals["damage_mse"] += float(damage_mse.item())
        totals["resource_mse"] += float(resource_mse.item())
        totals["survival_bce"] += float(survival_bce.item())
        totals["escape_mse"] += float(escape_mse.item())
        totals["progress_mse"] += float(progress_mse.item())
        totals["stall_bce"] += float(stall_bce.item())
        totals["affordance_bce"] += float(affordance_bce.item())
        totals["threat_trend_mse"] += float(threat_trend_mse.item())
        totals["survival_acc"] += float(survival_acc.item())
        n_batches += 1

    if n_batches == 0:
        return {key: 0.0 for key in totals}
    return {key: round(value / n_batches, 4) for key, value in totals.items()}


def _entropy(counter: Counter[str]) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counter.values():
        probability = count / total
        if probability > 0:
            entropy -= probability * math.log(probability)
    return entropy


def _empty_slice_report() -> dict[str, Any]:
    return {
        "n_states": 0,
        "top1_agreement": 0.0,
        "exact_top1_agreement": 0.0,
        "target_top1_tie_fraction": 0.0,
        "pairwise_preference_accuracy": 0.0,
        "predicted_top1_distribution": {},
        "predicted_top1_entropy": 0.0,
        "predicted_top1_normalized_entropy": 0.0,
        "dominant_action": None,
        "dominant_action_share": 0.0,
        "unique_top1_actions": 0,
    }


def _slice_report(
    *,
    n_states: int,
    top1_hits: int,
    exact_top1_hits: int,
    tie_states: int,
    pairwise_correct: int,
    pairwise_total: int,
    top1_counter: Counter[str],
    n_action_space: int,
) -> dict[str, Any]:
    if n_states <= 0:
        return _empty_slice_report()
    entropy = _entropy(top1_counter)
    max_entropy = math.log(max(n_action_space, 1)) if n_action_space > 1 else 1.0
    dominant_action, dominant_count = top1_counter.most_common(1)[0]
    return {
        "n_states": n_states,
        "top1_agreement": round(top1_hits / n_states, 4),
        "exact_top1_agreement": round(exact_top1_hits / n_states, 4),
        "target_top1_tie_fraction": round(tie_states / n_states, 4),
        "pairwise_preference_accuracy": round(pairwise_correct / max(pairwise_total, 1), 4),
        "predicted_top1_distribution": dict(sorted(top1_counter.items())),
        "predicted_top1_entropy": round(entropy, 4),
        "predicted_top1_normalized_entropy": round(entropy / max(max_entropy, 1e-6), 4),
        "dominant_action": dominant_action,
        "dominant_action_share": round(dominant_count / n_states, 4),
        "unique_top1_actions": len(top1_counter),
    }


def _anti_collapse_gate(ranking: dict[str, Any]) -> dict[str, Any]:
    overall = ranking["overall"]
    threat = ranking["regime_metrics"].get("hostile_contact_or_near", _empty_slice_report())
    resource = ranking["regime_metrics"].get("local_resource_facing", _empty_slice_report())

    checks: list[dict[str, Any]] = []

    def add_check(name: str, *, passed: bool, actual: Any, threshold: Any, reason: str) -> None:
        checks.append(
            {
                "name": name,
                "passed": bool(passed),
                "actual": actual,
                "threshold": threshold,
                "reason": reason,
            }
        )

    add_check(
        "dominant_action_share",
        passed=overall["dominant_action_share"] <= 0.7,
        actual=overall["dominant_action_share"],
        threshold="<= 0.70",
        reason="No single primitive should dominate almost all ranked states.",
    )
    add_check(
        "predicted_top1_entropy",
        passed=overall["predicted_top1_normalized_entropy"] >= 0.45,
        actual=overall["predicted_top1_normalized_entropy"],
        threshold=">= 0.45",
        reason="Predicted top-1 actions should retain meaningful diversity.",
    )
    add_check(
        "threat_slice_diversity",
        passed=(
            threat["n_states"] < 5 and threat["unique_top1_actions"] > 0
        ) or threat["unique_top1_actions"] >= 2,
        actual={
            "n_states": threat["n_states"],
            "unique_top1_actions": threat["unique_top1_actions"],
        },
        threshold=">= 2 unique actions when threat slice has enough support",
        reason="Threat slices should not collapse onto one reflex.",
    )
    add_check(
        "resource_slice_diversity",
        passed=(
            resource["n_states"] < 5 and resource["unique_top1_actions"] > 0
        ) or resource["unique_top1_actions"] >= 2,
        actual={
            "n_states": resource["n_states"],
            "unique_top1_actions": resource["unique_top1_actions"],
        },
        threshold=">= 2 unique actions when resource slice has enough support",
        reason="Resource-facing states should not default to generic movement.",
    )

    failed = [check for check in checks if not check["passed"]]
    if overall["n_states"] <= 0:
        status = "insufficient_coverage"
    elif failed:
        status = "fail"
    else:
        status = "pass"
    return {
        "status": status,
        "passed": status == "pass",
        "blocks_online_eval": status != "pass",
        "blocks_checkpoint_promotion": status != "pass",
        "checks": checks,
    }


def _evaluate_ranking(model, state_samples: list[dict[str, Any]], device: torch.device) -> dict[str, Any]:
    from snks.agent.stage90r_local_model import (
        compare_stage90r_target_labels,
        stage90r_action_utility,
        stage90r_target_order_key,
    )

    overall_top1_hits = 0
    overall_exact_top1_hits = 0
    overall_tie_states = 0
    overall_pairwise_correct = 0
    overall_pairwise_total = 0
    overall_top1_counter: Counter[str] = Counter()
    overall_states = 0

    regime_buckets = {
        "hostile_contact_or_near": {"top1_hits": 0, "exact_top1_hits": 0, "tie_states": 0, "pairwise_correct": 0, "pairwise_total": 0, "top1": Counter(), "states": 0},
        "local_resource_facing": {"top1_hits": 0, "exact_top1_hits": 0, "tie_states": 0, "pairwise_correct": 0, "pairwise_total": 0, "top1": Counter(), "states": 0},
        "low_vitals": {"top1_hits": 0, "exact_top1_hits": 0, "tie_states": 0, "pairwise_correct": 0, "pairwise_total": 0, "top1": Counter(), "states": 0},
        "neutral": {"top1_hits": 0, "exact_top1_hits": 0, "tie_states": 0, "pairwise_correct": 0, "pairwise_total": 0, "top1": Counter(), "states": 0},
    }
    explanatory_examples: list[dict[str, Any]] = []
    counterfactual_states = 0

    for state in state_samples:
        candidates = list(state.get("candidate_actions", []))
        if len(candidates) < 2:
            continue
        if int(state.get("comparison_coverage", {}).get("n_counterfactual_actions", 0)) >= 2:
            counterfactual_states += 1

        observation = state["observation"]
        class_ids = torch.tensor([observation["viewport_class_ids"]] * len(candidates), dtype=torch.long, device=device)
        confidences = torch.tensor([observation["viewport_confidences"]] * len(candidates), dtype=torch.float32, device=device)
        body = torch.tensor([observation["body_vector"]] * len(candidates), dtype=torch.float32, device=device)
        inventory = torch.tensor([observation["inventory_vector"]] * len(candidates), dtype=torch.float32, device=device)
        belief_state = torch.tensor(
            [
                observation.get(
                    "belief_state_vector",
                    observation.get("temporal_vector", []),
                )
            ]
            * len(candidates),
            dtype=torch.float32,
            device=device,
        )
        action = torch.tensor([candidate["action_index"] for candidate in candidates], dtype=torch.long, device=device)

        try:
            preds = model(class_ids, confidences, body, inventory, action, belief_state)
        except TypeError:
            preds = model(class_ids, confidences, body, inventory, action)
        pred_scores = stage90r_action_utility(**preds).detach().cpu().tolist()
        target_keys = [stage90r_target_order_key(candidate["label"]) for candidate in candidates]

        pred_best_idx = max(range(len(candidates)), key=lambda idx: float(pred_scores[idx]))
        pred_best_action = str(candidates[pred_best_idx]["action"])
        best_target_key = max(target_keys)
        tied_target_best_indices = [
            idx
            for idx, key in enumerate(target_keys)
            if key == best_target_key
        ]
        target_best_idx = tied_target_best_indices[0]
        target_best_action = str(candidates[target_best_idx]["action"])
        exact_top1_hit = pred_best_idx == target_best_idx
        top1_hit = pred_best_idx in tied_target_best_indices
        tie_state = len(tied_target_best_indices) > 1

        pairwise_correct = 0
        pairwise_total = 0
        for left in range(len(candidates)):
            for right in range(left + 1, len(candidates)):
                target_cmp = compare_stage90r_target_labels(
                    candidates[left]["label"],
                    candidates[right]["label"],
                )
                if target_cmp == 0:
                    continue
                pred_delta = float(pred_scores[left]) - float(pred_scores[right])
                pairwise_total += 1
                if pred_delta != 0.0 and (pred_delta > 0) == (target_cmp > 0):
                    pairwise_correct += 1

        overall_states += 1
        overall_top1_hits += int(top1_hit)
        overall_exact_top1_hits += int(exact_top1_hit)
        overall_tie_states += int(tie_state)
        overall_pairwise_correct += pairwise_correct
        overall_pairwise_total += pairwise_total
        overall_top1_counter[pred_best_action] += 1

        regime_keys = list(state.get("regime_labels", [])) or ["neutral"]
        derived_regimes = set(regime_keys)
        if "hostile_contact" in derived_regimes or "hostile_near" in derived_regimes:
            derived_regimes.add("hostile_contact_or_near")
        for regime in regime_buckets:
            if regime not in derived_regimes:
                continue
            bucket = regime_buckets[regime]
            bucket["states"] += 1
            bucket["top1_hits"] += int(top1_hit)
            bucket["exact_top1_hits"] += int(exact_top1_hit)
            bucket["tie_states"] += int(tie_state)
            bucket["pairwise_correct"] += pairwise_correct
            bucket["pairwise_total"] += pairwise_total
            bucket["top1"][pred_best_action] += 1

        if len(explanatory_examples) < 5:
            explanatory_examples.append(
                {
                    "state_id": int(state.get("state_id", -1)),
                    "primary_regime": state.get("primary_regime", "neutral"),
                    "predicted_best_action": pred_best_action,
                    "target_best_action": target_best_action,
                    "tied_target_best_actions": [
                        str(candidates[idx]["action"])
                        for idx in tied_target_best_indices
                    ],
                    "candidates": [
                        {
                            "action": candidate["action"],
                            "predicted_score": round(float(pred_scores[idx]), 4),
                            "target_order_key": [round(float(value), 4) for value in target_keys[idx]],
                            "pred_damage": round(float(preds["pred_damage"][idx].item()), 4),
                            "pred_resource_gain": round(float(preds["pred_resource_gain"][idx].item()), 4),
                            "pred_survival_prob": round(float(torch.sigmoid(preds["pred_survival_logit"][idx]).item()), 4),
                            "pred_escape_delta": round(float(preds["pred_escape_delta"][idx].item()), 4),
                            "pred_progress_delta": round(float(preds["pred_progress_delta"][idx].item()), 4),
                            "pred_stall_risk": round(float(torch.sigmoid(preds["pred_stall_risk_logit"][idx]).item()), 4),
                            "pred_affordance_persistence": round(
                                float(torch.sigmoid(preds["pred_affordance_persistence_logit"][idx]).item()),
                                4,
                            ),
                            "pred_threat_trend": round(float(preds["pred_threat_trend"][idx].item()), 4),
                            "target_label": candidate["label"],
                        }
                        for idx, candidate in enumerate(candidates)
                    ],
                }
            )

    report = {
        "overall": _slice_report(
            n_states=overall_states,
            top1_hits=overall_top1_hits,
            exact_top1_hits=overall_exact_top1_hits,
            tie_states=overall_tie_states,
            pairwise_correct=overall_pairwise_correct,
            pairwise_total=overall_pairwise_total,
            top1_counter=overall_top1_counter,
            n_action_space=6,
        ),
        "regime_metrics": {
            regime: _slice_report(
                n_states=bucket["states"],
                top1_hits=bucket["top1_hits"],
                exact_top1_hits=bucket["exact_top1_hits"],
                tie_states=bucket["tie_states"],
                pairwise_correct=bucket["pairwise_correct"],
                pairwise_total=bucket["pairwise_total"],
                top1_counter=bucket["top1"],
                n_action_space=6,
            )
            for regime, bucket in regime_buckets.items()
        },
        "counterfactual_support": {
            "n_states_with_counterfactual_comparison": counterfactual_states,
            "fraction": round(counterfactual_states / max(overall_states, 1), 3),
        },
        "explanatory_examples": explanatory_examples,
    }
    report["anti_collapse_gate"] = _anti_collapse_gate(report)
    return report


def _selection_score(ranking: dict[str, Any]) -> float:
    overall = ranking["overall"]
    return round(
        float(overall["top1_agreement"])
        + (0.5 * float(overall["pairwise_preference_accuracy"]))
        - (0.25 * float(overall["dominant_action_share"])),
        4,
    )


def _checkpoint_priority(
    *,
    valid_loss: float,
    selection_score: float,
) -> tuple[float, float]:
    """Prefer calibrated gate-passing checkpoints before ranking margin.

    Ranking-only selection can promote overfit models whose discrete ordering looks
    strong on a particular split while the underlying heads are poorly calibrated
    for online utility use. Lower validation loss wins first; ranking score only
    breaks ties.
    """
    return (-float(valid_loss), float(selection_score))


def _count_counterfactual_states(state_samples: list[dict[str, Any]]) -> int:
    return sum(
        1
        for state in state_samples
        if int(state.get("comparison_coverage", {}).get("n_counterfactual_actions", 0)) >= 2
    )


def main() -> None:
    from snks.agent.stage90r_local_model import (
        LocalActionEvaluator,
        Stage90RLocalDataset,
        collate_local_samples,
        flatten_state_samples,
        load_local_dataset,
        split_samples_by_episode,
        training_rows_from_payload,
    )
    from snks.agent.stage90r_local_policy import build_state_centered_training_examples

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--checkpoint-out", type=Path, default=DEFAULT_CKPT_PATH)
    parser.add_argument("--eval-out", type=Path, default=DEFAULT_EVAL_PATH)
    args = parser.parse_args()

    payload = load_local_dataset(args.dataset)
    base_samples = list(payload.get("samples", []))
    if not base_samples:
        base_samples = training_rows_from_payload(payload)
    if not base_samples:
        raise ValueError(f"Dataset has no usable samples: {args.dataset}")

    train_base_samples, valid_base_samples = split_samples_by_episode(
        base_samples,
        train_ratio=args.train_ratio,
    )
    train_state_samples = build_state_centered_training_examples(train_base_samples)
    valid_state_samples = build_state_centered_training_examples(valid_base_samples)
    train_samples = flatten_state_samples(train_state_samples)
    valid_samples = flatten_state_samples(valid_state_samples)
    if not train_samples or not valid_samples:
        raise ValueError("State-centered split produced an empty train or validation set")

    config = _build_config(payload["metadata"])
    model = LocalActionEvaluator(config)
    device = _device()
    model.to(device)

    train_loader = DataLoader(
        Stage90RLocalDataset(train_samples),
        batch_size=min(args.batch_size, max(1, len(train_samples))),
        shuffle=True,
        collate_fn=collate_local_samples,
    )
    valid_loader = DataLoader(
        Stage90RLocalDataset(valid_samples),
        batch_size=min(args.batch_size, max(1, len(valid_samples))),
        shuffle=False,
        collate_fn=collate_local_samples,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history: list[dict[str, Any]] = []
    best_valid_loss: float | None = None
    best_selection_score: float | None = None
    best_checkpoint_priority: tuple[float, float] | None = None
    best_ranking_report: dict[str, Any] | None = None
    best_blocked_candidate: dict[str, Any] | None = None
    checkpoint_saved = False
    for epoch in range(args.epochs):
        train_metrics = _run_epoch(model, train_loader, optimizer, device)
        with torch.no_grad():
            valid_metrics = _run_epoch(model, valid_loader, None, device)
            valid_ranking = _evaluate_ranking(model, valid_state_samples, device)
        selection_score = _selection_score(valid_ranking)
        history.append(
            {
                "epoch": epoch + 1,
                "train": train_metrics,
                "valid": valid_metrics,
                "valid_ranking": valid_ranking,
                "selection_score": selection_score,
            }
        )
        print(
            f"epoch {epoch + 1:02d}/{args.epochs} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"valid_loss={valid_metrics['loss']:.4f} "
            f"valid_top1={valid_ranking['overall']['top1_agreement']:.3f} "
            f"gate={valid_ranking['anti_collapse_gate']['status']}"
        )
        gate_passed = bool(valid_ranking["anti_collapse_gate"]["passed"])
        candidate_priority = _checkpoint_priority(
            valid_loss=valid_metrics["loss"],
            selection_score=selection_score,
        )
        if gate_passed and (
            best_checkpoint_priority is None
            or candidate_priority > best_checkpoint_priority
        ):
            best_checkpoint_priority = candidate_priority
            best_selection_score = selection_score
            best_valid_loss = valid_metrics["loss"]
            best_ranking_report = valid_ranking
            args.checkpoint_out.parent.mkdir(exist_ok=True)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "config": config.__dict__,
                    "dataset_path": str(args.dataset),
                    "metadata": payload["metadata"],
                    "offline_ranking": valid_ranking,
                    "selection_score": selection_score,
                    "offline_gate": valid_ranking["anti_collapse_gate"],
                },
                args.checkpoint_out,
            )
            checkpoint_saved = True
        elif not gate_passed and (
            best_blocked_candidate is None
            or selection_score > float(best_blocked_candidate["selection_score"])
        ):
            best_blocked_candidate = {
                "epoch": epoch + 1,
                "selection_score": selection_score,
                "valid_loss": valid_metrics["loss"],
                "offline_gate": valid_ranking["anti_collapse_gate"],
            }

    report = {
        "stage": "stage90r_local_evaluator_train",
        "dataset_path": str(args.dataset),
        "checkpoint_path": str(args.checkpoint_out) if checkpoint_saved else None,
        "config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "train_ratio": args.train_ratio,
        },
        "dataset_summary": {
            "n_samples": len(base_samples),
            "n_train_samples": len(train_samples),
            "n_valid_samples": len(valid_samples),
            "n_train_episodes": len({(s["seed"], s["episode_id"]) for s in train_base_samples}),
            "n_valid_episodes": len({(s["seed"], s["episode_id"]) for s in valid_base_samples}),
            "n_train_state_samples": len(train_state_samples),
            "n_valid_state_samples": len(valid_state_samples),
            "n_train_counterfactual_ready_states": _count_counterfactual_states(train_state_samples),
            "n_valid_counterfactual_ready_states": _count_counterfactual_states(valid_state_samples),
        },
        "metadata": payload["metadata"],
        "history": history,
        "best_valid_loss": best_valid_loss,
        "best_selection_score": best_selection_score,
        "best_valid_ranking": best_ranking_report,
        "checkpoint_saved": checkpoint_saved,
        "training_outcome": "checkpoint_promoted" if checkpoint_saved else "blocked_offline_gate",
        "best_blocked_candidate": best_blocked_candidate,
    }
    args.eval_out.parent.mkdir(exist_ok=True)
    args.eval_out.write_text(json.dumps(report, indent=2, default=_json_default))
    if checkpoint_saved:
        print(f"saved checkpoint: {args.checkpoint_out}")
    else:
        print("offline gate blocked checkpoint promotion; no checkpoint saved")
    print(f"saved eval report: {args.eval_out}")


if __name__ == "__main__":
    main()
