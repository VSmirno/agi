"""Experiment 16b: Confidence Ratio Gate (Stage 9 debt closure).

Обучаем HACPredictionEngine на повторяющемся паттерне (self-loop A→A):
memory ≈ bind(A, A), predict_next(A) ≈ unbind(A, bind(A,A)) ≈ A → PE ≈ 0.

Для noise: predict_next(noise) даёт случайный вектор ≠ noise → PE ≈ 0.5.

Gate: mean_confidence(focused) / mean_confidence(noise) > 1.5
"""

from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding="utf-8")

import torch

from snks.dcam.hac import HACEngine
from snks.daf.hac_prediction import HACPredictionEngine
from snks.daf.types import HACPredictionConfig, MetacogConfig
from snks.gws.workspace import GWSState
from snks.metacog.monitor import MetacogMonitor
from snks.sks.embedder import SKSEmbedder


RATIO_THRESHOLD = 1.5
N_NODES = 500
HAC_DIM = 512
CORE_SIZE = 80
N_TRAIN = 60     # повторений для обучения self-loop
N_TEST = 20      # тестовых семплов


def make_focused_cluster(n_nodes: int = N_NODES, seed: int = 42) -> set[int]:
    torch.manual_seed(seed)
    return set(range(CORE_SIZE))  # фиксированное ядро


def compute_confidence_from_pe(monitor: MetacogMonitor, winner_pe: float) -> float:
    """Вычисляет confidence через MetacogMonitor.update() с заданным winner_pe."""
    class _Proxy:
        mean_prediction_error = 0.1
        winner_pe: float
    proxy = _Proxy()
    proxy.winner_pe = winner_pe

    gws = GWSState(
        winner_id=0,
        winner_nodes=set(range(10)),
        winner_size=10,
        winner_score=10.0,
        dominance=0.9,
    )
    state = monitor.update(gws, proxy)
    return state.confidence


def run() -> dict:
    print("=== Exp 16b: Confidence Ratio Gate (Stage 9) ===")
    print(f"  HAC_DIM={HAC_DIM}, N_TRAIN={N_TRAIN}")

    hac = HACEngine(dim=HAC_DIM)
    embedder = SKSEmbedder(n_nodes=N_NODES, hac_dim=HAC_DIM, device="cpu")

    # Создаём focused embedding (один стабильный паттерн)
    cluster = make_focused_cluster()
    focused_emb = embedder.embed({0: cluster})[0]
    print(f"  Focused embedding norm: {focused_emb.norm():.3f}")

    # Обучаем self-loop: A → A (observe одним и тем же embedding)
    cfg = HACPredictionConfig(memory_decay=0.99)  # медленное затухание для стабильной памяти
    predictor = HACPredictionEngine(hac, cfg)

    print(f"  Training self-loop A→A ({N_TRAIN} steps)...")
    for _ in range(N_TRAIN):
        predictor.observe({0: focused_emb})

    # Проверяем качество self-loop предсказания
    pred_focused = predictor.predict_next({0: focused_emb})
    if pred_focused is not None:
        self_sim = hac.similarity(pred_focused, focused_emb)
        print(f"  Self-loop predict_next(A) ≈ A? cosine={self_sim:.3f}")
    else:
        print("  WARNING: no prediction (memory empty)")
        return {"ratio": 0.0, "mean_focused": 0.0, "mean_noise": 0.0}

    # MetacogMonitor: gamma=1.0, confidence = 1 - winner_pe
    monitor = MetacogMonitor(MetacogConfig(alpha=0.0, beta=0.0, gamma=1.0))

    # Focused: вычисляем winner_pe для известного паттерна
    focused_pes = []
    for i in range(N_TEST):
        # Небольшой шум вокруг focused (вариации того же концепта)
        torch.manual_seed(i)
        noisy_cluster = set(range(CORE_SIZE)) | {CORE_SIZE + i % 20}
        emb_var = embedder.embed({0: noisy_cluster})[0]
        pred = predictor.predict_next({0: emb_var})
        if pred is not None:
            pe = predictor.compute_winner_pe(pred, emb_var)
            focused_pes.append(pe)

    # Noise: случайные embeddings
    noise_pes = []
    for i in range(N_TEST):
        torch.manual_seed(i + 10000)
        noise_emb = hac.random_vector()
        pred = predictor.predict_next({0: noise_emb})
        if pred is not None:
            pe = predictor.compute_winner_pe(pred, noise_emb)
            noise_pes.append(pe)

    # Вычисляем confidence через MetacogMonitor
    focused_conf = [compute_confidence_from_pe(monitor, pe) for pe in focused_pes]
    noise_conf = [compute_confidence_from_pe(monitor, pe) for pe in noise_pes]

    mean_focused = sum(focused_conf) / len(focused_conf) if focused_conf else 0.0
    mean_noise = sum(noise_conf) / len(noise_conf) if noise_conf else 1.0
    ratio = mean_focused / mean_noise if mean_noise > 1e-6 else 0.0

    print(f"  mean winner_pe (focused): {sum(focused_pes)/len(focused_pes):.3f}" if focused_pes else "  no focused PEs")
    print(f"  mean winner_pe (noise):   {sum(noise_pes)/len(noise_pes):.3f}" if noise_pes else "  no noise PEs")
    print(f"  mean_confidence(focused): {mean_focused:.3f}")
    print(f"  mean_confidence(noise):   {mean_noise:.3f}")
    print(f"  ratio:                    {ratio:.3f}  (threshold: {RATIO_THRESHOLD})")

    status = "PASS" if ratio >= RATIO_THRESHOLD else "FAIL"
    print(f"  Result: {status}")
    return {"ratio": ratio, "mean_focused": mean_focused, "mean_noise": mean_noise}


if __name__ == "__main__":
    result = run()
    assert result["ratio"] >= RATIO_THRESHOLD, (
        f"Exp 16b FAILED: ratio={result['ratio']:.3f} < {RATIO_THRESHOLD}"
    )
    print("Exp 16b: PASS")
