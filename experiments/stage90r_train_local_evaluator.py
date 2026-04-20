"""Train Stage 90R local action evaluator on viewport-first dataset."""

from __future__ import annotations

import argparse
import json
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
        loss = damage_mse + 0.5 * resource_mse + survival_bce + 0.25 * escape_mse

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        survival_pred = (torch.sigmoid(preds["pred_survival_logit"]) >= 0.5).float()
        survival_acc = torch.mean((survival_pred == batch["survived"]).float())

        totals["loss"] += float(loss.item())
        totals["damage_mse"] += float(damage_mse.item())
        totals["resource_mse"] += float(resource_mse.item())
        totals["survival_bce"] += float(survival_bce.item())
        totals["escape_mse"] += float(escape_mse.item())
        totals["survival_acc"] += float(survival_acc.item())
        n_batches += 1

    if n_batches == 0:
        return {key: 0.0 for key in totals}
    return {key: round(value / n_batches, 4) for key, value in totals.items()}


def main() -> None:
    from snks.agent.stage90r_local_model import (
        LocalActionEvaluator,
        Stage90RLocalDataset,
        collate_local_samples,
        load_local_dataset,
        split_samples_by_episode,
    )

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
    samples = list(payload.get("samples", []))
    if not samples:
        raise ValueError(f"Dataset has no samples: {args.dataset}")

    train_samples, valid_samples = split_samples_by_episode(samples, train_ratio=args.train_ratio)
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
    best_valid = None
    for epoch in range(args.epochs):
        train_metrics = _run_epoch(model, train_loader, optimizer, device)
        with torch.no_grad():
            valid_metrics = _run_epoch(model, valid_loader, None, device)
        history.append(
            {
                "epoch": epoch + 1,
                "train": train_metrics,
                "valid": valid_metrics,
            }
        )
        print(
            f"epoch {epoch + 1:02d}/{args.epochs} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"valid_loss={valid_metrics['loss']:.4f} "
            f"valid_surv_acc={valid_metrics['survival_acc']:.3f}"
        )
        if best_valid is None or valid_metrics["loss"] < best_valid:
            best_valid = valid_metrics["loss"]
            args.checkpoint_out.parent.mkdir(exist_ok=True)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "config": config.__dict__,
                    "dataset_path": str(args.dataset),
                    "metadata": payload["metadata"],
                },
                args.checkpoint_out,
            )

    report = {
        "stage": "stage90r_local_evaluator_train",
        "dataset_path": str(args.dataset),
        "checkpoint_path": str(args.checkpoint_out),
        "config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "train_ratio": args.train_ratio,
        },
        "dataset_summary": {
            "n_samples": len(samples),
            "n_train_samples": len(train_samples),
            "n_valid_samples": len(valid_samples),
            "n_train_episodes": len({(s["seed"], s["episode_id"]) for s in train_samples}),
            "n_valid_episodes": len({(s["seed"], s["episode_id"]) for s in valid_samples}),
        },
        "metadata": payload["metadata"],
        "history": history,
        "best_valid_loss": best_valid,
    }
    args.eval_out.parent.mkdir(exist_ok=True)
    args.eval_out.write_text(json.dumps(report, indent=2, default=_json_default))
    print(f"saved checkpoint: {args.checkpoint_out}")
    print(f"saved eval report: {args.eval_out}")


if __name__ == "__main__":
    main()
