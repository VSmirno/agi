"""Focused contracts for the exp167 checkpoint-only 2x2 oracle swap."""

from __future__ import annotations

import importlib
import importlib.util

import torch


def _experiment():
    name = "experiments.exp167_hurdle_oracle_swap"
    assert importlib.util.find_spec(name) is not None, "exp167 audit is missing"
    return importlib.import_module(name)


def test_oracle_swap_arm_algebra_includes_false_positive_zero():
    exp = _experiment()
    logits = torch.tensor([[1.0, -1.0, 1.0]])
    conditional = torch.tensor([[0.2, 0.3, 0.4]])
    oracle = torch.tensor([[0.0, 0.6, 0.7]])

    arms = exp.oracle_swap_gates(logits, conditional, oracle)

    torch.testing.assert_close(arms["PP"], torch.tensor([[0.2, 0.0, 0.4]]))
    torch.testing.assert_close(arms["PO"], torch.tensor([[0.0, 0.0, 0.7]]))
    torch.testing.assert_close(arms["OP"], torch.tensor([[0.0, 0.3, 0.4]]))
    torch.testing.assert_close(arms["OO"], oracle)
    assert arms["PO"][0, 0].item() == 0.0


def test_interpretation_truth_table_localizes_the_failed_component():
    exp = _experiment()

    assert exp.interpret_swap(False, True, True)[0] == "atom_bottleneck"
    assert exp.interpret_swap(True, False, True)[0] == "conditional_bottleneck"
    assert exp.interpret_swap(False, False, True)[0] == "both_components_fail"
    assert exp.interpret_swap(True, True, True)[0] == "error_interaction"
    assert exp.interpret_swap(True, True, False)[0] == "invalid_oracle_audit"


def test_defaults_lock_checkpoints_references_and_observability():
    exp = _experiment()
    args = exp.build_parser().parse_args(["--out", "run"])

    assert args.checkpoint == exp.DEFAULT_CHECKPOINT
    assert args.exp166_reference == exp.DEFAULT_EXP166_REFERENCE
    assert args.exp159_reference == exp.DEFAULT_EXP159_REFERENCE
    assert args.progress_interval == 30
