"""Checkpoint-only linear separability of Push-local event/no-change proxies."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch

from experiments.exp145_physics_transfer import SOURCE_LAYOUTS, _collect
from experiments.exp146_temporal_mpc_physics import (
    ProgressJournal, _exit_code, _positive, _progress_interval,
)
from experiments.exp149_replay_coverage import _audit_counts
from experiments.exp150_residual_dynamics import PROTOCOL as RESIDUAL_PROTOCOL
from experiments.exp151_event_balanced_dynamics import (
    FIXED_CONFIG, FIXED_CORPUS, _load_residual_checkpoint,
)
from snks.env.core_types import Episode
from snks.pipeline import core_experiment as core


TASKS = {"interact_rgb_changed": 3, "forward_rgb_changed": 2}
PROTOCOL = {
    "episodes_per_layout": 512, "collection_steps": 64,
    "probe_updates": 400, "probe_batch_size": 256, "probe_seed": 152, "z_dim": 256,
}
LEARNING_RATE = 0.01
THRESHOLDS = {
    "ordered_balanced_accuracy_min": 0.80,
    "each_class_recall_min": 0.70,
    "ordered_minus_shuffled_balanced_accuracy_min": 0.20,
}
DEFAULT_CHECKPOINT = Path(
    "output_to_user/core/exp150-residual-dynamics-001/residual_checkpoint.pt"
)


def _split_episodes(
    episodes: dict[str, list[Episode]], episodes_per_layout: int,
) -> tuple[dict[str, list[Episode]], dict[str, list[Episode]], int]:
    """Use exp146's cutoff, covering every episode on its respective side."""
    cutoff = round(0.75 * episodes_per_layout)
    if not 0 < cutoff < episodes_per_layout:
        raise ValueError("split requires nonempty train and heldout episode sets")
    uids = [ep.uid for rows in episodes.values() for ep in rows]
    if len(uids) != len(set(uids)):
        raise ValueError("duplicate episode UIDs could overlap the probe split")
    if any(len(rows) != episodes_per_layout for rows in episodes.values()):
        raise ValueError("every layout must contain episodes_per_layout episodes")
    return ({name: rows[:cutoff] for name, rows in episodes.items()},
            {name: rows[cutoff:] for name, rows in episodes.items()}, cutoff)


def _collect_corpus(args: argparse.Namespace, deadline: float, journal: ProgressJournal):
    episodes: dict[str, list[Episode]] = {name: [] for name in SOURCE_LAYOUTS}
    total = len(episodes) * args.episodes_per_layout
    completed = 0
    journal.update("collect", 0, total)
    for offset in range(args.episodes_per_layout):
        for index, (name, (layout, _actions)) in enumerate(SOURCE_LAYOUTS.items()):
            core._check_deadline(deadline, f"collect/{name}/{offset}")
            seed = 10000 + index * 100000 + offset
            episodes[name].append(_collect(name, layout, seed, args.collection_steps))
            completed += 1
            if completed % args.collection_log_every == 0 or completed == total:
                journal.update("collect", completed, total, layout=name, offset=offset)
    train, heldout, cutoff = _split_episodes(episodes, args.episodes_per_layout)
    journal.update("corpus_counts", 0, 1)
    coverage = _audit_counts(episodes)
    counts = {
        "episodes": completed,
        "transitions": sum(len(ep.transitions) for rows in episodes.values() for ep in rows),
        "natural_terminals_by_layout": {
            name: sum(bool(ep.transitions and ep.transitions[-1].terminated) for ep in rows)
            for name, rows in episodes.items()
        },
        "natural_terminals_fit_cutoff_by_layout": {
            name: sum(bool(ep.transitions and ep.transitions[-1].terminated) for ep in rows)
            for name, rows in train.items()
        },
        "rgb_changing_interact_transitions": coverage["rgb_changing_interact_transitions"],
        "episodes_with_rgb_changing_interact": coverage["episodes_with_rgb_changing_interact"],
        "action_counts": {
            action: {key: row[key] for key in ("total", "rgb_changed", "rgb_no_change")}
            for action, row in coverage["actions"].items()
        },
    }
    fixed = args.episodes_per_layout == 512 and args.collection_steps == 64
    if fixed and counts != FIXED_CORPUS:
        raise AssertionError(f"fixed exp149/151 corpus mismatch: {counts}")
    journal.update("corpus_counts", 1, 1, fixed_corpus_verified=fixed)
    return train, heldout, {
        "observed_counts": counts, "fixed_corpus_verified": fixed,
        "source_layouts_insertion_order": list(SOURCE_LAYOUTS),
        "seed_scheme": "10000 + layout_index * 100000 + offset",
        "collection_policy": "exp145._collect: random.Random(seed + 145000).randrange(5)",
        "collection_interleaved_by_offset": True,
        "fit_cutoff_per_layout": cutoff,
        "episode_uids": {
            split: {name: [ep.uid for ep in rows] for name, rows in by_layout.items()}
            for split, by_layout in (("train", train), ("heldout", heldout))
        },
        "split_scope": "episode-disjoint supervised probe fit/heldout; encoder saw entire corpus",
    }


@torch.no_grad()
def _encode_task(
    encoder: torch.nn.Module, episodes: list[Episode], action: int, batch_size: int,
    device: torch.device | str, deadline: float, journal: ProgressJournal, split: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode only current RGB; future RGB supplies the diagnostic target only."""
    transitions = [t for ep in episodes for t in ep.transitions if t.action == action]
    if not transitions:
        raise ValueError(f"no action-{action} transitions in {split}")
    labels = torch.tensor([
        int(not np.array_equal(t.before.rgb, t.after.rgb)) for t in transitions
    ], dtype=torch.long)
    encoded = []
    journal.update("probe_encode", 0, len(transitions), action=action, split=split)
    for start in range(0, len(transitions), batch_size):
        core._check_deadline(deadline, f"encode/{split}/{action}/{start}")
        chunk = transitions[start:start + batch_size]
        rgb = torch.as_tensor(np.stack([t.before.rgb for t in chunk]), device=device)
        z = encoder(rgb.float() / 255)
        if not torch.isfinite(z).all():
            raise FloatingPointError("non-finite frozen encoder features")
        encoded.append(z.detach().cpu())
        journal.update("probe_encode", start + len(chunk), len(transitions),
                       action=action, split=split)
    return torch.cat(encoded), labels


def _balanced_indices(labels: np.ndarray, batch_size: int, rng: np.random.Generator) -> np.ndarray:
    """Deterministic 50:50 class draws with replacement for each probe update."""
    if batch_size < 2 or batch_size % 2:
        raise ValueError("50:50 sampling requires a positive even batch_size")
    pools = [np.flatnonzero(labels == value) for value in (0, 1)]
    if any(len(pool) == 0 for pool in pools):
        raise ValueError("probe sampling requires both classes")
    indices = np.concatenate([rng.choice(pool, batch_size // 2, replace=True) for pool in pools])
    rng.shuffle(indices)
    return indices


def _class_counts(labels: torch.Tensor) -> dict[str, int]:
    return {str(value): int((labels == value).sum()) for value in (0, 1)}


def _metrics(labels: torch.Tensor, logits: torch.Tensor) -> dict[str, Any]:
    """Score the fixed probability >= 0.5 decision against unbalanced real labels."""
    counts = _class_counts(labels)
    if not all(counts.values()):
        raise ValueError("probe metrics require both classes in the evaluation split")
    if not torch.isfinite(logits).all():
        raise FloatingPointError("non-finite probe predictions")
    predicted = logits >= 0
    confusion = {
        "tn": int(((labels == 0) & ~predicted).sum()),
        "fp": int(((labels == 0) & predicted).sum()),
        "fn": int(((labels == 1) & ~predicted).sum()),
        "tp": int(((labels == 1) & predicted).sum()),
    }
    recalls = {"0": confusion["tn"] / counts["0"], "1": confusion["tp"] / counts["1"]}
    minority = min((0, 1), key=lambda value: (counts[str(value)], value))
    majority = 1 - minority
    return {
        "class_counts": counts, "confusion": confusion, "recall_by_class": recalls,
        "balanced_accuracy": (recalls["0"] + recalls["1"]) / 2,
        "minority_class": minority, "majority_class": majority,
        "minority_recall": recalls[str(minority)], "majority_recall": recalls[str(majority)],
        "minority_basis": "evaluation split class counts; ties choose class 0",
    }


def _fit_task(
    train: tuple[torch.Tensor, torch.Tensor], heldout: tuple[torch.Tensor, torch.Tensor],
    args: argparse.Namespace, action: int, device: torch.device,
    deadline: float, journal: ProgressJournal,
) -> dict[str, Any]:
    train_x, train_y = (value.to(device) for value in train)
    heldout_x, heldout_y = (value.to(device) for value in heldout)
    # Normalization is learned on probe-training episodes alone and is linear.
    mean = train_x.mean(0)
    scale = train_x.std(0, unbiased=False).clamp_min(1e-6)
    train_x, heldout_x = (train_x - mean) / scale, (heldout_x - mean) / scale
    shuffle_seed = args.probe_seed + 10000 + action
    permutation = np.random.default_rng(shuffle_seed).permutation(len(train_y))
    shuffled_y = train_y[torch.as_tensor(permutation, device=device)]
    result: dict[str, Any] = {
        "action": action, "train_counts": _class_counts(train_y),
        "heldout_counts": _class_counts(heldout_y),
        "shuffled_train_counts": _class_counts(shuffled_y),
        "shuffle_seed": shuffle_seed,
        "shuffled_labels_differ_from_original": int((train_y != shuffled_y).sum()),
    }
    for arm, labels in (("ordered", train_y), ("shuffled", shuffled_y)):
        head = torch.nn.Linear(train_x.shape[1], 1, device=device)
        torch.nn.init.zeros_(head.weight)
        torch.nn.init.zeros_(head.bias)
        optimizer = torch.optim.Adam(head.parameters(), lr=LEARNING_RATE)
        sample_seed = args.probe_seed + action
        rng = np.random.default_rng(sample_seed)
        cpu_labels = labels.cpu().numpy()
        sampled = np.zeros(2, dtype=np.int64)
        first_loss = last_loss = None
        journal.update("probe_fit", 0, args.probe_updates, action=action, arm=arm)
        for update in range(args.probe_updates):
            core._check_deadline(deadline, f"probe_fit/{action}/{arm}/{update}")
            indices = _balanced_indices(cpu_labels, args.probe_batch_size, rng)
            sampled += np.bincount(cpu_labels[indices], minlength=2)
            index = torch.as_tensor(indices, device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = head(train_x[index]).squeeze(-1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels[index].float())
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite logistic probe loss")
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach())
            if first_loss is None:
                first_loss = last_loss
            if (update + 1) % 50 == 0 or update + 1 == args.probe_updates:
                journal.update("probe_fit", update + 1, args.probe_updates,
                               action=action, arm=arm, loss=last_loss)
        with torch.no_grad():
            result[arm] = {
                "updates": args.probe_updates, "sampling_seed": sample_seed,
                "sampled_class_counts": {str(value): int(sampled[value]) for value in (0, 1)},
                "loss_first": first_loss, "loss_last": last_loss,
                "train_fit_labels": _metrics(labels, head(train_x).squeeze(-1)),
                "heldout": _metrics(heldout_y, head(heldout_x).squeeze(-1)),
            }
        journal.update("probe_metrics", 1, 1, action=action, arm=arm,
                       heldout_balanced_accuracy=result[arm]["heldout"]["balanced_accuracy"])
    return result


def _task_signal(ordered: dict[str, Any], shuffled: dict[str, Any]) -> bool:
    accuracy = ordered["balanced_accuracy"]
    control = shuffled["balanced_accuracy"]
    recalls = [ordered["recall_by_class"][str(value)] for value in (0, 1)]
    return bool(
        all(math.isfinite(value) for value in (accuracy, control, *recalls))
        and accuracy >= THRESHOLDS["ordered_balanced_accuracy_min"]
        and min(recalls) >= THRESHOLDS["each_class_recall_min"]
        and accuracy >= control + THRESHOLDS["ordered_minus_shuffled_balanced_accuracy_min"]
    )


def _outcome_label(signals: dict[str, bool], exact_protocol: bool) -> str:
    if not exact_protocol:
        return "non_preregistered_protocol"
    interact, forward = signals["interact_rgb_changed"], signals["forward_rgb_changed"]
    if interact and forward:
        return "representation_signal_evidence"
    if not interact and forward:
        return "contact_representation_bottleneck_evidence"
    if not interact and not forward:
        return "encoder_objective_bottleneck_evidence"
    return "mixed_or_inconclusive"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out", type=Path, required=True)
    for name, default in PROTOCOL.items():
        parser.add_argument(f"--{name.replace('_', '-')}", type=_positive, default=default)
    parser.add_argument("--encode-batch-size", type=_positive, default=256)
    parser.add_argument("--collection-log-every", type=_positive, default=32)
    parser.add_argument("--max-seconds", type=_positive, default=600)
    parser.add_argument("--progress-interval", type=_progress_interval, default=30)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.episodes_per_layout < 4:
        parser.error("at least 4 episodes per layout are required for the probe split")
    if args.probe_batch_size < 2 or args.probe_batch_size % 2:
        parser.error("probe batch size must be positive and even")
    args.out.mkdir(parents=True, exist_ok=False)
    deadline = time.monotonic() + args.max_seconds
    manifest = {
        "argv": list(sys.orig_argv), "cwd": str(Path.cwd()),
        "analysis_git_head": core._git_commit(), "checkpoint_git_head": None,
        "checkpoint": str(args.checkpoint), "budgets": core._jsonable(vars(args)),
        "fixed_protocol": PROTOCOL, "thresholds": THRESHOLDS,
        "status": "running", "exit_code": None, "exit_status": None,
    }
    core._write_json(args.out / "manifest.json", manifest)
    with ProgressJournal(args.out / "progress.jsonl", args.progress_interval) as journal:
        try:
            journal.update("initialize", 0, 1, operation="safe_residual_checkpoint_load")
            model, temporal_probe, head, metadata = _load_residual_checkpoint(args.checkpoint)
            del temporal_probe
            manifest.update(checkpoint_git_head=head, checkpoint_metadata=metadata)
            core._write_json(args.out / "manifest.json", manifest)
            if metadata["z_dim"] != args.z_dim:
                raise ValueError("checkpoint z_dim does not match --z-dim")
            device = torch.device(metadata["device"])
            core._seed_everything(args.probe_seed)
            journal.update("initialize", 1, 1, device=str(device))
            train, heldout, corpus = _collect_corpus(args, deadline, journal)
            matching = {
                "probe_defaults": all(getattr(args, key) == value for key, value in PROTOCOL.items()),
                "fixed_corpus": corpus["fixed_corpus_verified"],
                "checkpoint_training_budgets": all(metadata["budgets"].get(key) == value
                                                    for key, value in RESIDUAL_PROTOCOL.items()),
                "checkpoint_training_config": metadata["config"] == FIXED_CONFIG,
                "checkpoint_uniform_residual": metadata["event_balanced"] is False,
            }
            exact = all(matching.values())
            manifest["protocol_match"] = matching
            core._write_json(args.out / "manifest.json", manifest)
            tasks = {}
            for name, action in TASKS.items():
                encoded = {}
                for split, rows in (("train", train), ("heldout", heldout)):
                    encoded[split] = _encode_task(
                        model.encoder, [ep for layout in SOURCE_LAYOUTS for ep in rows[layout]],
                        action, args.encode_batch_size, device, deadline, journal, split,
                    )
                tasks[name] = _fit_task(encoded["train"], encoded["heldout"], args,
                                        action, device, deadline, journal)
            signals = {name: _task_signal(task["ordered"]["heldout"], task["shuffled"]["heldout"])
                       for name, task in tasks.items()}
            outcome = _outcome_label(signals, exact)
            journal.update("artifacts", 0, 1)
            core._check_deadline(deadline, "artifacts")
            results = {
                "status": "completed", "claim": "bounded frozen-encoder linear probe diagnostic",
                "exact_protocol": exact, "protocol_match": matching, "corpus": corpus,
                "tasks": tasks, "thresholds": THRESHOLDS, "task_signal_thresholds_met": signals,
                "representation_signal_evidence": exact and all(signals.values()),
                "outcome_label": outcome,
                "next_hypothesis": {
                    "representation_signal_evidence": "transition-state conditioning/gating",
                    "contact_representation_bottleneck_evidence": "contact representation",
                    "encoder_objective_bottleneck_evidence": "encoder/objective",
                    "mixed_or_inconclusive": "mixed diagnostic; no single bottleneck isolated",
                    "non_preregistered_protocol": "complete the preregistered protocol before interpretation",
                }[outcome],
                "controls": {
                    "encoder_frozen": True, "input": "z(before.rgb) only",
                    "target": "before/after RGB inequality, within the selected action",
                    "head": "single linear logistic probe per task and control",
                    "initialization": "identical zero weights and bias",
                    "normalization": "train-only mean/std per latent dimension; std floor 1e-6",
                    "optimizer": "Adam", "learning_rate": LEARNING_RATE,
                    "sampling": "50:50 classes with replacement",
                    "control": "one fixed train-label permutation; heldout labels remain real",
                    "probability_threshold": 0.5, "early_stopping": False,
                    "encoder_updates": 0, "goal_success_evaluated": False, "push2_run": False,
                },
                "limitations": [
                    "Raw RGB-change labels are Push-local proxies, not generic semantic event labels.",
                    "Probe supervision is diagnostic only; no predictor or controller is trained with it.",
                    "Heldout episodes are unseen by supervised probes, but were in the frozen encoder training corpus.",
                    "One seed, one linear head, and fixed updates; failure is not proof that information is absent or nonlinearly inaccessible.",
                    "A reduced-budget smoke cannot provide preregistered representation-signal evidence.",
                    "Diagnostic bottleneck labels express next hypotheses, not AGI, JEPA, concept, or transfer proof.",
                ],
                "artifacts": {"manifest": "manifest.json", "progress": "progress.jsonl",
                              "external_log": "run.log"},
            }
            core._write_json(args.out / "results.json", results)
            manifest.update(status="completed", exit_code=0, exit_status=0)
            core._write_json(args.out / "manifest.json", manifest)
            journal.update("artifacts", 1, 1, outcome_label=outcome)
            return 0
        except BaseException as error:
            code = _exit_code(error)
            core._write_json(args.out / "manifest.json", {
                **manifest, "status": "failed", "exit_code": code, "exit_status": code,
                "error": f"{type(error).__name__}: {error}",
            })
            raise


if __name__ == "__main__":
    raise SystemExit(main())
