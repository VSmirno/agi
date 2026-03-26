"""Experiment 21: Meta-embedding stability (Stage 10).

Проверяет, что MetaEmbedder создаёт различимые представления для разных классов shapes.

Метрика:
    intra_sim = mean cosine_similarity пар внутри одного класса
    inter_sim = mean cosine_similarity пар из разных классов
    Gate: mean(intra_sim) > mean(inter_sim) + 0.05

Дизайн:
    10 классов, 50 последовательностей каждого, 20 шагов каждая.
    Если основной критерий не проходит → ablation по meta_decay ∈ {0.5, 0.7, 0.8, 0.9}.
"""

from __future__ import annotations

import sys

sys.stdout.reconfigure(encoding="utf-8")

import random

import torch
from torch import Tensor

from snks.dcam.hac import HACEngine
from snks.daf.types import HierarchicalConfig
from snks.sks.meta_embedder import MetaEmbedder

MARGIN_THRESHOLD = 0.05
ABLATION_DECAYS = [0.5, 0.7, 0.8, 0.9]
N_INTER_PAIRS = 200


def _generate_class_embeddings(
    hac_dim: int,
    n_classes: int,
    n_sequences: int,
    seq_len: int,
) -> list[list[list[Tensor]]]:
    """Генерирует embeddings для всех классов, последовательностей и шагов.

    Returns:
        embeddings[class_id][seq_id][step] = Tensor(hac_dim,)
    """
    all_embeddings: list[list[list[Tensor]]] = []
    for c in range(n_classes):
        torch.manual_seed(c * 1000)
        class_center = torch.randn(hac_dim)
        class_center = class_center / class_center.norm()

        seq_list: list[list[Tensor]] = []
        for s in range(n_sequences):
            step_list: list[Tensor] = []
            for t in range(seq_len):
                torch.manual_seed(c * 1000 + s * 100 + t)
                noise = torch.randn(hac_dim) * 0.1
                embed = class_center + noise
                embed = embed / embed.norm()
                step_list.append(embed)
            seq_list.append(step_list)
        all_embeddings.append(seq_list)

    return all_embeddings


def _collect_meta_embeds(
    hac: HACEngine,
    config: HierarchicalConfig,
    all_embeddings: list[list[list[Tensor]]],
    n_classes: int,
    n_sequences: int,
    seq_len: int,
) -> list[list[Tensor]]:
    """Прогоняет MetaEmbedder по всем последовательностям.

    Returns:
        meta_embeds[class_id][seq_id] = Tensor(hac_dim,)
    """
    meta_embedder = MetaEmbedder(hac=hac, config=config)
    result: list[list[Tensor]] = []

    for c in range(n_classes):
        class_metas: list[Tensor] = []
        for s in range(n_sequences):
            meta_embedder.reset()
            for t in range(seq_len):
                embed = all_embeddings[c][s][t]
                meta_embedder.update({0: embed})
            final_meta = meta_embedder.get_meta_embed()
            assert final_meta is not None, f"meta_embed is None after {seq_len} steps"
            class_metas.append(final_meta.detach().clone())
        result.append(class_metas)

    return result


def _cosine_sim(a: Tensor, b: Tensor) -> float:
    """Косинусное сходство двух единичных векторов."""
    return float(torch.dot(a, b).clamp(-1.0, 1.0).item())


def _compute_intra_inter(
    meta_embeds: list[list[Tensor]],
    n_classes: int,
    n_sequences: int,
    n_inter_pairs: int,
    rng: random.Random,
) -> tuple[float, float]:
    """Вычисляет mean intra-class и mean inter-class cosine similarity.

    Returns:
        (mean_intra_sim, mean_inter_sim)
    """
    # Intra: все пары (s1, s2) внутри каждого класса
    intra_sims: list[float] = []
    for c in range(n_classes):
        for s1 in range(n_sequences):
            for s2 in range(s1 + 1, n_sequences):
                sim = _cosine_sim(meta_embeds[c][s1], meta_embeds[c][s2])
                intra_sims.append(sim)

    # Inter: случайные пары из разных классов
    inter_sims: list[float] = []
    for _ in range(n_inter_pairs):
        c1 = rng.randrange(n_classes)
        c2 = rng.randrange(n_classes - 1)
        if c2 >= c1:
            c2 += 1
        s1 = rng.randrange(n_sequences)
        s2 = rng.randrange(n_sequences)
        sim = _cosine_sim(meta_embeds[c1][s1], meta_embeds[c2][s2])
        inter_sims.append(sim)

    mean_intra = sum(intra_sims) / len(intra_sims) if intra_sims else 0.0
    mean_inter = sum(inter_sims) / len(inter_sims) if inter_sims else 0.0
    return mean_intra, mean_inter


def _evaluate_decay(
    hac: HACEngine,
    meta_decay: float,
    all_embeddings: list[list[list[Tensor]]],
    n_classes: int,
    n_sequences: int,
    seq_len: int,
    rng: random.Random,
) -> tuple[float, float, float]:
    """Оценивает конкретное значение meta_decay.

    Returns:
        (mean_intra_sim, mean_inter_sim, margin)
    """
    config = HierarchicalConfig(enabled=True, meta_decay=meta_decay)
    meta_embeds = _collect_meta_embeds(
        hac, config, all_embeddings, n_classes, n_sequences, seq_len
    )
    mean_intra, mean_inter = _compute_intra_inter(
        meta_embeds, n_classes, n_sequences, N_INTER_PAIRS, rng
    )
    margin = mean_intra - mean_inter
    return mean_intra, mean_inter, margin


def run(
    device: str = "cpu",
    hac_dim: int = 256,
    n_classes: int = 10,
    n_sequences: int = 50,
    seq_len: int = 20,
) -> dict:
    """Запускает эксперимент meta-embedding stability.

    Args:
        device: устройство torch (HAC работает на CPU).
        hac_dim: размерность HAC векторов.
        n_classes: количество классов.
        n_sequences: количество последовательностей на класс.
        seq_len: длина каждой последовательности (шагов).

    Returns:
        Словарь с ключами: passed, mean_intra_sim, mean_inter_sim, margin, best_decay.
    """
    # HAC всегда на CPU (ограничение архитектуры)
    hac_device = torch.device("cpu")
    hac = HACEngine(dim=hac_dim, device=hac_device)

    rng = random.Random(42)

    # Генерация embeddings один раз (независимо от decay)
    all_embeddings = _generate_class_embeddings(hac_dim, n_classes, n_sequences, seq_len)

    # Основной прогон с default meta_decay=0.8
    default_decay = 0.8
    mean_intra, mean_inter, margin = _evaluate_decay(
        hac, default_decay, all_embeddings, n_classes, n_sequences, seq_len, rng
    )

    passed = margin > MARGIN_THRESHOLD
    best_decay = default_decay

    if not passed:
        # Ablation по meta_decay
        best_margin = margin
        for decay in ABLATION_DECAYS:
            rng_ablation = random.Random(42)
            intra_a, inter_a, margin_a = _evaluate_decay(
                hac, decay, all_embeddings, n_classes, n_sequences, seq_len, rng_ablation
            )
            if margin_a > best_margin:
                best_margin = margin_a
                best_decay = decay
                mean_intra = intra_a
                mean_inter = inter_a
                margin = margin_a

        passed = margin > MARGIN_THRESHOLD

    return {
        "passed": passed,
        "mean_intra_sim": float(mean_intra),
        "mean_inter_sim": float(mean_inter),
        "margin": float(margin),
        "best_decay": float(best_decay),
    }


if __name__ == "__main__":
    device = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    result = run(device=device)
    print(result)
    sys.exit(0 if result["passed"] else 1)
