"""Real-episode learning loop with a small, inspectable evaluation boundary."""

from dataclasses import dataclass, replace
import hashlib

import torch

from snks.agent.core_agent import CoreAgent
from snks.env.core_types import Episode, Mode
from snks.learning.core_replay import SequenceReplay
from snks.learning.core_trainer import CoreTrainer, tensorize
from snks.pipeline.core_tasks import TaskCase, resolve_goal, score_episode


@dataclass
class EpisodeResult:
    episode: Episode
    steps: int
    agent_failed: bool
    infrastructure_failed: bool
    audit: list[dict]
    success: bool = False
    model_calls: int = 0


def model_digest(model: torch.nn.Module) -> str:
    """Fingerprint weights and buffers, not a mutable environment instance."""
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def run_episode(adapter, agent: CoreAgent, case: TaskCase, mode: Mode,
                replay: SequenceReplay, trainer: CoreTrainer | None,
                *, exploration: float | None = None, updates: int = 1) -> EpisodeResult:
    """Only complete real episodes enter replay; evaluation cannot train."""
    allowed = {Mode.TRAIN: {"train"}, Mode.ADAPT: {"adapt"},
               Mode.EVALUATE: {"validation", "test", "zero_shot_test"}}
    if case.split not in allowed[mode]:
        raise PermissionError(f"{mode.value} cannot read {case.split}")
    if mode == Mode.EVALUATE and trainer is not None:
        raise PermissionError("evaluation must not own a trainer")
    evaluation = mode == Mode.EVALUATE
    before = (model_digest(agent.model), replay.manifest()) if evaluation else None
    agent.model.eval()
    obs = adapter.reset(case.seed)
    steps = adapter.reset_transitions
    if steps >= case.max_steps:
        raise ValueError("reset exhausted episode transition budget")
    goal = resolve_goal(case, obs)
    agent.start(obs, goal)
    audit = [{"step": steps, "sensors": obs.sensors.tolist(),
              "sensor_mask": obs.sensor_mask.tolist(),
              "diagnostic": adapter.diagnostic_snapshot()}]
    transitions, calls, failed = [], 0, False
    fraction = 0.0 if evaluation else agent.config.exploration_fraction
    if exploration is not None:
        if evaluation and exploration != 0.0:
            raise ValueError("evaluation exploration must be zero")
        fraction = exploration
    while steps < case.max_steps:
        try:
            action = agent.act(fraction)
            calls += agent.last_model_calls
            if not 0 <= action < len(adapter.actions.names):
                raise ValueError("agent returned invalid primitive")
        except (FloatingPointError, ValueError, RuntimeError) as error:
            # A model/action failure stays in the denominator; no legacy fallback.
            failed = True
            audit.append({"step": steps, "agent_failure": str(error)})
            break
        transition = adapter.step(action)
        steps += 1
        if steps == case.max_steps and not transition.terminated:
            transition = replace(transition, truncated=True)
        transitions.append(transition)
        try:
            agent.observe(transition)
        except (FloatingPointError, ValueError, RuntimeError) as error:
            failed = True
            audit.append({"step": steps, "agent_failure": str(error)})
            break
        audit.append({"step": steps, "action": action,
                      "sensors": transition.after.sensors.tolist(),
                      "sensor_mask": transition.after.sensor_mask.tolist(),
                      "diagnostic": adapter.diagnostic_snapshot(),
                      "candidates": agent.last_trace})
        if transition.terminated or transition.truncated:
            break
    episode = Episode(case.uid, case.split, case.family, case.ruleset, tuple(transitions))
    success = not failed and score_episode(case, audit)
    if evaluation:
        if before != (model_digest(agent.model), replay.manifest()):
            raise RuntimeError("evaluation mutated model or replay")
    elif not failed and transitions:
        replay.append(episode, mode)
        if trainer is not None:
            for _ in range(updates):
                samples = replay.sample(agent.config.batch_size, agent.config.train_horizon,
                                        agent.config.burn_in, agent.config.recent_fraction,
                                        schema=obs.schema,
                                        salient_fraction=agent.config.salient_fraction)
                batch = tensorize(samples, agent.config.burn_in, agent.state.z.device)
                trainer.update(batch, mode)
    return EpisodeResult(episode, steps, failed, False, audit, success, calls)
