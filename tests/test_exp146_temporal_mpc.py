from __future__ import annotations

import json
import sys
from pathlib import Path
import time
from types import SimpleNamespace

import torch

sys.path.append(str(Path(__file__).resolve().parents[1] / "experiments"))

from experiments.exp146_temporal_mpc_physics import (
    ProgressJournal,
    TerminationNeutralModel,
    _late_fork_audit,
    _rank_fork_rows,
    _reconstruct_model_root,
    build_parser,
)
from snks.agent.core_world_model import LatentState, Prediction


class _StubModel:
    def __init__(self):
        self.schemas = {"stub": (2, 0)}
        self.initial_state = object()
        self.next_state = object()
        self.prediction = Prediction(
            next_state=self.next_state,
            terminated_prob=torch.tensor([0.25, 0.75]),
            uncertainty=torch.tensor([0.1, 0.2]),
            member_z=torch.tensor([[[1.0]], [[3.0]]]),
        )

    def initial(self, observation):
        self.initial_observation = observation
        return self.initial_state

    def step(self, state, actions):
        self.step_inputs = (state, actions)
        return self.prediction


def test_termination_neutral_model_preserves_dynamics_but_zeros_termination():
    underlying = _StubModel()
    model = TerminationNeutralModel(underlying)
    observation = object()
    state = object()
    actions = torch.tensor([0, 1], dtype=torch.long)

    assert model.schemas is underlying.schemas
    assert model.initial(observation) is underlying.initial_state
    assert underlying.initial_observation is observation

    prediction = model.step(state, actions)

    assert underlying.step_inputs == (state, actions)
    assert prediction.next_state is underlying.prediction.next_state
    assert prediction.uncertainty is underlying.prediction.uncertainty
    assert prediction.member_z is underlying.prediction.member_z
    torch.testing.assert_close(
        prediction.terminated_prob,
        torch.zeros_like(underlying.prediction.terminated_prob),
    )


def test_progress_journal_flushes_updates_and_final_status_to_jsonl_and_stdout(
    tmp_path, capsys
):
    output_path = tmp_path / "progress.jsonl"

    with ProgressJournal(output_path) as journal:
        journal.update("collect", completed=1, total=3, loss=0.25)

        update_record = json.loads(output_path.read_text().strip())
        stdout_record = json.loads(capsys.readouterr().out.strip())

        assert update_record == stdout_record
        assert update_record["stage"] == "collect"
        assert update_record["completed"] == 1
        assert update_record["total"] == 3
        assert update_record["loss"] == 0.25
        assert update_record["elapsed_seconds"] >= 0.0

        journal.close(status="completed")

    records = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert records[-1]["status"] == "completed"
    assert records[-1]["elapsed_seconds"] >= update_record["elapsed_seconds"]


def test_late_fork_audit_is_opt_in():
    parser = build_parser()

    default = parser.parse_args(["--config", "config.yaml", "--out", "run"])
    enabled = parser.parse_args(
        ["--config", "config.yaml", "--out", "run", "--late-fork-audit"]
    )

    assert default.late_fork_audit is False
    assert enabled.late_fork_audit is True


class _PrefixModel:
    def initial(self, observation):
        value = float(observation)
        return LatentState(
            z=torch.tensor([[value]]),
            sensors=torch.tensor([[value + 10.0]]),
            sensor_mask=torch.tensor([[True]]),
            hidden=torch.tensor([[0.0]]),
            schema="stub",
        )

    def step(self, state, actions):
        next_state = LatentState(
            z=state.z + 100.0,
            sensors=state.sensors + 100.0,
            sensor_mask=torch.tensor([[False]]),
            hidden=state.hidden + actions[:, None],
            schema=state.schema,
        )
        return Prediction(
            next_state=next_state,
            terminated_prob=torch.zeros(len(actions)),
            uncertainty=torch.zeros(len(actions)),
            member_z=next_state.z[None],
        )


class _PrefixAdapter:
    def __init__(self):
        self.actions = []

    def reset(self, seed):
        assert seed == 20000
        return 1.0

    def step(self, action):
        self.actions.append(action)
        return SimpleNamespace(
            after=1.0 + len(self.actions),
            terminated=False,
            truncated=False,
        )

    def diagnostic_snapshot(self):
        return {"actions": list(self.actions)}


def test_reconstruct_model_root_teacher_forces_observations_but_not_hidden():
    state, diagnostic = _reconstruct_model_root(
        _PrefixModel(), _PrefixAdapter(), 20000, (0, 3, 2, 3, 2)
    )

    torch.testing.assert_close(state.z, torch.tensor([[6.0]]))
    torch.testing.assert_close(state.sensors, torch.tensor([[16.0]]))
    torch.testing.assert_close(state.sensor_mask, torch.tensor([[True]]))
    torch.testing.assert_close(state.hidden, torch.tensor([[10.0]]))
    assert diagnostic == {"actions": [0, 3, 2, 3, 2]}


def test_fork_ranks_use_lower_cost_then_lexicographic_actions():
    rows = [
        {"actions": [1, 0, 0], "actual_ordered_cost": 0.0},
        {"actions": [0, 0, 1], "actual_ordered_cost": 1.0},
        {"actions": [0, 1, 0], "actual_ordered_cost": 0.0},
    ]

    _rank_fork_rows(rows, ("actual_ordered_cost",))

    assert [row["actual_ordered_rank"] for row in rows] == [2, 3, 1]


class _AuditModel:
    schemas = {"grid-v1": (5, 1)}

    def initial(self, observation):
        z = torch.tensor(
            [[float(torch.as_tensor(observation.rgb).float().mean())]]
        )
        return LatentState(
            z=z,
            sensors=torch.as_tensor(observation.sensors)[None],
            sensor_mask=torch.as_tensor(observation.sensor_mask)[None],
            hidden=torch.zeros((1, 1)),
            schema=observation.schema,
        )

    def step(self, state, actions):
        delta = actions[:, None].float() / 10.0
        next_state = LatentState(
            z=state.z + delta,
            sensors=state.sensors,
            sensor_mask=state.sensor_mask,
            hidden=state.hidden + delta,
            schema=state.schema,
        )
        return Prediction(
            next_state=next_state,
            terminated_prob=torch.ones(len(actions)),
            uncertainty=torch.zeros(len(actions)),
            member_z=next_state.z[None],
        )


class _AuditProbe:
    def __call__(self, anchor, target, _horizon):
        return -(anchor - target).square().mean(-1)


def test_late_fork_audit_writes_all_ranked_rows(tmp_path):
    rows_path = tmp_path / "forks.jsonl"
    with ProgressJournal(tmp_path / "progress.jsonl") as journal:
        summary = _late_fork_audit(
            _AuditModel(),
            _AuditProbe(),
            SimpleNamespace(max_model_calls=128),
            time.monotonic() + 30,
            journal,
            rows_path,
        )

    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    assert len(rows) == 125
    assert all(
        set(row) >= {
            "actions",
            "actual_ordered_rank",
            "predicted_ordered_rank",
            "actual_raw_rank",
            "predicted_raw_rank",
        }
        for row in rows
    )
    assert summary["protocol"]["prefix"] == [0, 3, 2, 3, 2]
    assert summary["protocol"]["fork_count"] == 125
    assert set(summary["beam"]) == {"ordered", "raw"}
