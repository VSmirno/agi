"""Experiment 22: Hierarchical prediction accuracy (Stage 10).

Проверяет, что L2 predictor (meta-level) точнее предсказывает на горизонте 5 циклов,
чем L1 predictor на горизонте 1.

Метрика:
- l1_accuracy = доля предсказаний L1 с cosine_sim > 0.6
- l2_accuracy = доля предсказаний L2 с cosine_sim > 0.6 (горизонт 5)
- Gate: l2_accuracy > l1_accuracy + 0.05

Подход: синтетический паттерн A->B->C->A->... (3 класса).
L1 предсказывает следующий шаг в пространстве raw embeddings.
L2 предсказывает шаг+5 в пространстве meta-embeddings.

Поскольку meta-embedding — EWA с decay=0.8, после ~5 шагов он стабилизируется
на «лицо» текущего аттрактора, делая предсказание горизонта 5 более надёжным.
"""

from __future__ import annotations

import sys

sys.stdout.reconfigure(encoding="utf-8")

import torch
import torch.nn.functional as F

from snks.daf.hac_prediction import HACPredictionEngine
from snks.daf.types import HACPredictionConfig, HierarchicalConfig
from snks.dcam.hac import HACEngine
from snks.sks.meta_embedder import MetaEmbedder

HORIZON = 5
COSINE_THRESHOLD = 0.6
GATE_MARGIN = 0.05


def run(
    device: str = "cpu",
    hac_dim: int = 256,
    n_train: int = 200,
    n_test: int = 100,
) -> dict:
    """Run Exp 22: Hierarchical prediction accuracy.

    Args:
        device: torch device string (always uses CPU for HAC).
        hac_dim: dimensionality of HAC vectors.
        n_train: number of training steps.
        n_test: number of test steps.

    Returns:
        Dict with keys: passed, l1_accuracy, l2_accuracy, margin.
    """
    print("=== Exp 22: Hierarchical Prediction Accuracy (Stage 10) ===")
    print(f"  device={device}, hac_dim={hac_dim}, n_train={n_train}, n_test={n_test}")

    # HAC всегда на CPU — per spec
    hac = HACEngine(dim=hac_dim, device=torch.device("cpu"))

    # Три центра кластеров (фиксированный seed)
    torch.manual_seed(42)
    centers = [torch.randn(hac_dim) for _ in range(3)]
    centers = [c / c.norm() for c in centers]

    def make_embed(step: int, noise: float = 0.05) -> torch.Tensor:
        """Создаёт детерминированный embedding для шага `step`."""
        torch.manual_seed(step * 17 + 3)
        e = centers[step % 3] + noise * torch.randn(hac_dim)
        return (e / e.norm()).to(torch.device("cpu"))

    # Инициализация предикторов и meta-embedder
    pred_cfg = HACPredictionConfig(memory_decay=0.95, enabled=True)
    hier_cfg = HierarchicalConfig(meta_decay=0.8)

    hac_l1 = HACPredictionEngine(hac, pred_cfg)
    hac_l2 = HACPredictionEngine(hac, pred_cfg)
    me = MetaEmbedder(hac, hier_cfg)

    # -----------------------------------------------------------------------
    # Предзаполнение future_metas: независимый MetaEmbedder, прогнанный по
    # всем шагам train + test + horizon заранее. Используется для получения
    # «целевого» meta_embed на шаге t+HORIZON во время теста.
    # -----------------------------------------------------------------------
    me_future = MetaEmbedder(hac, hier_cfg)
    future_metas: list[torch.Tensor | None] = []
    for t in range(n_train + n_test + HORIZON + 1):
        emb = {0: make_embed(t)}
        m = me_future.update(emb)
        future_metas.append(m.clone() if m is not None else None)

    # -----------------------------------------------------------------------
    # Фаза обучения
    # -----------------------------------------------------------------------
    print("  Training...")
    for t in range(n_train):
        emb = {0: make_embed(t)}
        hac_l1.observe(emb)

        meta_t = me.update(emb)
        if meta_t is not None:
            hac_l2.observe({"meta": meta_t.clone()})

    # -----------------------------------------------------------------------
    # Фаза теста
    # -----------------------------------------------------------------------
    print("  Testing...")
    l1_correct = 0
    l2_correct = 0
    l1_total = 0
    l2_total = 0

    for t in range(n_test):
        abs_t = n_train + t
        emb = {0: make_embed(abs_t)}

        # --- L1: предсказать следующий raw embedding (горизонт 1) ---
        pred_l1 = hac_l1.predict_next(emb)
        if pred_l1 is not None:
            actual_next = make_embed(abs_t + 1)
            sim = F.cosine_similarity(
                pred_l1.unsqueeze(0), actual_next.unsqueeze(0)
            ).item()
            l1_correct += int(sim > COSINE_THRESHOLD)
            l1_total += 1

        # observe текущий шаг, чтобы память L1 продолжала обновляться
        hac_l1.observe(emb)

        # --- L2: предсказать meta_embed через HORIZON шагов ---
        meta_t = me.update(emb)
        if meta_t is not None:
            pred_l2 = hac_l2.predict_next({"meta": meta_t.clone()})
            if pred_l2 is not None:
                # Целевой meta_embed — из предзаполненного future_metas
                future_idx = abs_t + HORIZON
                if future_idx < len(future_metas) and future_metas[future_idx] is not None:
                    actual_future: torch.Tensor = future_metas[future_idx]  # type: ignore[assignment]
                    sim_l2 = F.cosine_similarity(
                        pred_l2.unsqueeze(0), actual_future.unsqueeze(0)
                    ).item()
                    l2_correct += int(sim_l2 > COSINE_THRESHOLD)
                    l2_total += 1

            # observe для L2 тоже продолжаем
            hac_l2.observe({"meta": meta_t.clone()})

    # -----------------------------------------------------------------------
    # Итоги
    # -----------------------------------------------------------------------
    l1_accuracy = l1_correct / l1_total if l1_total > 0 else 0.0
    l2_accuracy = l2_correct / l2_total if l2_total > 0 else 0.0
    margin = l2_accuracy - l1_accuracy
    passed = bool(margin > GATE_MARGIN)

    print(f"  L1 accuracy : {l1_accuracy:.3f}  ({l1_correct}/{l1_total})")
    print(f"  L2 accuracy : {l2_accuracy:.3f}  ({l2_correct}/{l2_total})")
    print(f"  Margin      : {margin:+.3f}  (gate: >{GATE_MARGIN})")
    print(f"  Result      : {'PASS' if passed else 'FAIL'}")

    return {
        "passed": passed,
        "l1_accuracy": l1_accuracy,
        "l2_accuracy": l2_accuracy,
        "margin": margin,
    }


if __name__ == "__main__":
    _device = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    result = run(device=_device)
    print(result)
    sys.exit(0 if result["passed"] else 1)
