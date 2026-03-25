"""Experiment 19: SKS Embedding Quality.

Проверяет, что HAC-embeddings СКС образуют семантически значимые кластеры.

Подход: симулируем, что разные стимулы одной категории активируют похожие
(перекрывающиеся) наборы узлов — это свойство обученной ДАП-сети. Embedder
должен давать похожие векторы для близких кластеров и разные для далёких.

Gate: NMI(NN-кластеры, true labels) > 0.7
"""

from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding="utf-8")

import torch
import numpy as np
from sklearn.metrics import normalized_mutual_info_score

from snks.sks.embedder import SKSEmbedder


NMI_THRESHOLD = 0.7
N_NODES = 2000
HAC_DIM = 256
N_CATEGORIES = 5
N_VARIATIONS = 8      # вариации одной категории
CORE_SIZE = 80        # общее ядро кластера одной категории
VARIATION_SIZE = 20   # дополнительные узлы вариации
K_NEIGHBORS = 3


def make_category_clusters(n_nodes: int = N_NODES) -> tuple[list[set[int]], list[int]]:
    """Создаёт синтетические кластеры с известной категориальной структурой.

    Каждая категория имеет 'ядро' из CORE_SIZE узлов.
    Вариации = ядро + случайные VARIATION_SIZE узлов.
    Между категориями нет перекрытия ядер.
    """
    torch.manual_seed(42)
    clusters, labels = [], []

    # Разбиваем узлы на N_CATEGORIES непересекающихся диапазонов
    nodes_per_cat = n_nodes // N_CATEGORIES
    for cat in range(N_CATEGORIES):
        base = cat * nodes_per_cat
        core_nodes = set(range(base, base + CORE_SIZE))

        for var in range(N_VARIATIONS):
            torch.manual_seed(cat * 1000 + var)
            # Вариация = ядро + немного случайных узлов из того же диапазона
            extra = set(
                torch.randint(base + CORE_SIZE,
                              min(base + nodes_per_cat, n_nodes),
                              (VARIATION_SIZE,)).tolist()
            )
            cluster = core_nodes | extra
            clusters.append(cluster)
            labels.append(cat)

    return clusters, labels


def run() -> float:
    print("=== Exp 19: SKS Embedding Quality ===")
    print(f"  N_NODES={N_NODES}, HAC_DIM={HAC_DIM}")
    print(f"  {N_CATEGORIES} categories × {N_VARIATIONS} variations")
    print(f"  Core size: {CORE_SIZE}, variation extra: {VARIATION_SIZE}")

    embedder = SKSEmbedder(n_nodes=N_NODES, hac_dim=HAC_DIM, device="cpu")
    clusters, labels = make_category_clusters(N_NODES)
    N = len(clusters)

    # Embed all clusters
    embeddings_list = []
    for i, cluster in enumerate(clusters):
        result = embedder.embed({i: cluster})
        embeddings_list.append(result[i])

    emb_matrix = torch.stack(embeddings_list)  # (N, HAC_DIM)

    # Pairwise cosine similarities
    normed = emb_matrix / emb_matrix.norm(dim=1, keepdim=True).clamp(min=1e-8)
    sim_matrix = (normed @ normed.T).numpy()  # (N, N)

    # K-NN classification
    pred_labels = []
    for i in range(N):
        sims = sim_matrix[i].copy()
        sims[i] = -1  # exclude self
        nn_indices = np.argsort(sims)[-K_NEIGHBORS:]
        nn_labels = [labels[j] for j in nn_indices]
        vote = max(set(nn_labels), key=nn_labels.count)
        pred_labels.append(vote)

    nmi = normalized_mutual_info_score(labels, pred_labels)
    accuracy = sum(p == t for p, t in zip(pred_labels, labels)) / N

    print(f"  N embeddings: {N}")
    print(f"  NN-classification accuracy: {accuracy:.3f}")
    print(f"  NMI (NN-кластеры vs true): {nmi:.3f}  (threshold: {NMI_THRESHOLD})")

    # Показать внутри- и межкатегорийное сходство
    intra_sims, inter_sims = [], []
    for i in range(N):
        for j in range(i + 1, N):
            s = float(sim_matrix[i, j])
            if labels[i] == labels[j]:
                intra_sims.append(s)
            else:
                inter_sims.append(s)

    if intra_sims and inter_sims:
        print(f"  Intra-category cosine sim: {np.mean(intra_sims):.3f} ± {np.std(intra_sims):.3f}")
        print(f"  Inter-category cosine sim: {np.mean(inter_sims):.3f} ± {np.std(inter_sims):.3f}")

    status = "PASS" if nmi >= NMI_THRESHOLD else "FAIL"
    print(f"  Result: {status}")
    return nmi


if __name__ == "__main__":
    nmi = run()
    assert nmi >= NMI_THRESHOLD, f"Exp 19 FAILED: NMI={nmi:.3f} < {NMI_THRESHOLD}"
    print("Exp 19: PASS")
