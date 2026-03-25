"""Experiment 18: BroadcastPolicy — Global Ignition.

Тестирует механизм broadcast напрямую через DafEngine:
- Baseline: обычный цикл DAF без дополнительных токов
- Broadcast: тот же цикл + ток в winner_nodes через BroadcastPolicy

cross_activation_ratio = mean_n_spikes(with_broadcast) / mean_n_spikes(baseline) > 1.2

Примечание: тест механизма напрямую, без зависимости от полного pipeline
(GWS детекция ненадёжна на малых сетях для unit-тестов).
"""

from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding="utf-8")

import torch

from snks.daf.engine import DafEngine
from snks.daf.types import DafConfig
from snks.metacog.policies import BroadcastPolicy
from snks.metacog.monitor import MetacogState


RATIO_THRESHOLD = 1.2
N_NODES = 500
N_STEPS = 50
WINNER_NODES_FRACTION = 0.1   # 10% узлов = winner_nodes
N_TRIALS = 20
BROADCAST_STRENGTH = 2.0
BROADCAST_THRESHOLD = 0.0     # всегда транслируем


def make_config() -> DafConfig:
    cfg = DafConfig()
    cfg.num_nodes = N_NODES
    cfg.oscillator_model = "fhn"
    cfg.noise_sigma = 0.05
    cfg.dt = 0.01   # 0.01 ms — FHN spikes require dt=0.01 (50 steps = 0.5 time units)
    cfg.device = "cpu"
    return cfg


def count_spikes(engine: DafEngine, n_steps: int,
                 extra_currents: torch.Tensor | None = None) -> int:
    """Запускает engine на n_steps, возвращает n_spikes."""
    if extra_currents is not None:
        # Добавляем broadcast ток к базовому
        base = engine._input_currents.clone() if engine._input_currents is not None else torch.zeros(N_NODES)
        engine.set_input((base + extra_currents).clamp(min=0))
    result = engine.step(n_steps)
    # n_spikes=-1 (lazy) — считаем из fired_history
    return int(result.fired_history.sum().item())


def run() -> dict:
    print("=== Exp 18: BroadcastPolicy — Global Ignition ===")
    print(f"  N_NODES={N_NODES}, N_STEPS={N_STEPS}")
    print(f"  BROADCAST_STRENGTH={BROADCAST_STRENGTH}, winner_fraction={WINNER_NODES_FRACTION}")

    # Создаём winner_nodes: первые 10% узлов
    n_winner = int(N_NODES * WINNER_NODES_FRACTION)
    winner_nodes = set(range(n_winner))

    # Создаём broadcast currents через BroadcastPolicy
    policy = BroadcastPolicy(strength=BROADCAST_STRENGTH, threshold=BROADCAST_THRESHOLD)
    state = MetacogState(
        confidence=1.0,
        dominance=0.9,
        stability=0.8,
        pred_error=0.1,
        winner_nodes=winner_nodes,
    )
    cfg = make_config()
    policy.apply(state, cfg)
    broadcast_currents = policy.get_broadcast_currents(N_NODES)
    assert broadcast_currents is not None

    # Проверяем что broadcast_currents ненулевые в winner_nodes
    assert broadcast_currents[list(winner_nodes)[0]].item() == BROADCAST_STRENGTH
    assert broadcast_currents[n_winner].item() == 0.0  # non-winner = 0

    # Тест: сравниваем n_spikes с broadcast vs без
    spikes_baseline = []
    spikes_broadcast = []

    print(f"  Running {N_TRIALS} trials...")
    for trial in range(N_TRIALS):
        torch.manual_seed(trial)

        # Baseline: обычный ввод
        engine_base = DafEngine(make_config(), enable_learning=False)
        torch.manual_seed(trial)
        input_currents = torch.rand(N_NODES) * 0.5
        engine_base.set_input(input_currents)
        engine_base.step(20)  # прогрев

        result_base = engine_base.step(N_STEPS)
        spikes_baseline.append(int(result_base.fired_history.sum().item()))

        # Broadcast: тот же ввод + broadcast
        engine_bc = DafEngine(make_config(), enable_learning=False)
        engine_bc.states = engine_base.states.clone()  # одинаковое начальное состояние
        torch.manual_seed(trial)
        engine_bc.set_input((input_currents + broadcast_currents).clamp(min=0))
        engine_bc.step(20)  # прогрев с broadcast

        engine_bc.set_input(input_currents + broadcast_currents)
        result_bc = engine_bc.step(N_STEPS)
        spikes_broadcast.append(int(result_bc.fired_history.sum().item()))

    mean_baseline = sum(spikes_baseline) / len(spikes_baseline) if spikes_baseline else 0.0
    mean_broadcast = sum(spikes_broadcast) / len(spikes_broadcast) if spikes_broadcast else 0.0

    if mean_baseline > 0:
        ratio = mean_broadcast / mean_baseline
    else:
        # Если baseline = 0, считаем ratio по абсолютной разнице
        ratio = (mean_broadcast + 1.0) / 1.0  # broadcast всё равно > 0

    print(f"  Mean spikes baseline:  {mean_baseline:.1f}")
    print(f"  Mean spikes broadcast: {mean_broadcast:.1f}")
    print(f"  cross_activation_ratio: {ratio:.3f}  (threshold: {RATIO_THRESHOLD})")

    status = "PASS" if ratio >= RATIO_THRESHOLD else "FAIL"
    print(f"  Result: {status}")
    return {"ratio": ratio, "mean_baseline": mean_baseline, "mean_broadcast": mean_broadcast}


if __name__ == "__main__":
    result = run()
    assert result["ratio"] >= RATIO_THRESHOLD, (
        f"Exp 18 FAILED: cross_activation_ratio={result['ratio']:.3f} < {RATIO_THRESHOLD}"
    )
    print("Exp 18: PASS")
