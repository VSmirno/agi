from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1] / "experiments"))

from experiments.exp146_temporal_mpc_physics import (
    ProgressJournal,
    TerminationNeutralModel,
)
from snks.agent.core_world_model import Prediction


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
