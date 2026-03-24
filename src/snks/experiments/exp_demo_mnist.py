"""СНКС Demo: Unsupervised MNIST Learning.

Демонстрация: система учится различать цифры 0-9 без каких-либо меток,
используя только FHN-осцилляторы + STDP.

Генерирует самодостаточный HTML-отчёт с t-SNE, тепловой картой кластеров
и примерами изображений.

Usage:
    python -m snks.experiments.exp_demo_mnist
    python -m snks.experiments.exp_demo_mnist --nodes 50000 --per-class 200 --epochs 5
"""

from __future__ import annotations

import argparse
import base64
import json
import time
from io import BytesIO
from pathlib import Path

import numpy as np
import torch

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from snks.daf.types import DafConfig, EncoderConfig, PipelineConfig, PredictionConfig, SKSConfig
from snks.data.mnist import MnistLoader
from snks.pipeline.runner import Pipeline
from snks.sks.metrics import compute_nmi

# --------------------------------------------------------------------------- #
# Визуальные константы                                                         #
# --------------------------------------------------------------------------- #

DIGIT_NAMES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
COLORS = [
    "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
    "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9",
]

# --------------------------------------------------------------------------- #
# Конфигурация                                                                 #
# --------------------------------------------------------------------------- #

def make_config(device: str, num_nodes: int) -> PipelineConfig:
    return PipelineConfig(
        daf=DafConfig(
            num_nodes=num_nodes,
            avg_degree=30,
            oscillator_model="fhn",
            coupling_strength=0.05,
            dt=0.01,
            noise_sigma=0.005,
            fhn_I_base=0.0,
            device=device,
        ),
        encoder=EncoderConfig(
            sdr_size=8192,
            pool_h=8,
            pool_w=8,
            sdr_sparsity=0.04,
            sdr_current_strength=1.0,
        ),
        sks=SKSConfig(
            top_k=min(num_nodes // 2, 5000),
            dbscan_eps=0.3,
            dbscan_min_samples=5,
            min_cluster_size=5,
            coherence_mode="rate",
        ),
        prediction=PredictionConfig(),
        steps_per_cycle=200,
        device=device,
    )

# --------------------------------------------------------------------------- #
# Обучение с отслеживанием                                                     #
# --------------------------------------------------------------------------- #

def train_and_collect(
    pipeline: Pipeline,
    images: torch.Tensor,
    labels: torch.Tensor,
    epochs: int,
    n_classes: int,
) -> dict:
    """Обучение с записью NMI/PE по эпохам и сбором firing-rate векторов."""
    n_images = images.shape[0]
    n_nodes = pipeline.engine.config.num_nodes

    nmi_history: list[float] = []
    pe_history: list[float] = []
    sks_history: list[float] = []

    firing_rates_final = np.zeros((n_images, n_nodes), dtype=np.float32)
    labels_final = np.zeros(n_images, dtype=np.int64)

    for epoch in range(epochs):
        t0 = time.time()
        perm = torch.randperm(n_images)
        epoch_pe: list[float] = []
        epoch_sks: list[int] = []
        epoch_rates = np.zeros((n_images, n_nodes), dtype=np.float32)

        for i, idx in enumerate(perm):
            result = pipeline.perception_cycle(images[idx])
            epoch_pe.append(result.mean_prediction_error)
            epoch_sks.append(result.n_sks)

            fh = pipeline.engine.get_fired_history()
            if fh is not None:
                epoch_rates[i] = fh.float().mean(dim=0).cpu().numpy()

            if (i + 1) % 200 == 0 or (i + 1) == n_images:
                speed = (i + 1) / max(time.time() - t0, 1e-6)
                print(
                    f"  Epoch {epoch + 1}/{epochs}  [{i + 1}/{n_images}]"
                    f"  PE={np.mean(epoch_pe[-50:]):.3f}"
                    f"  SKS={np.mean(epoch_sks[-50:]):.1f}"
                    f"  {speed:.1f} img/s"
                )

        # NMI для этой эпохи
        true_epoch = labels[perm].numpy()
        km = KMeans(n_clusters=n_classes, n_init=5, random_state=42)
        pred = km.fit_predict(epoch_rates)
        nmi = float(compute_nmi(pred, true_epoch))

        mean_pe = float(np.mean(epoch_pe))
        mean_sks = float(np.mean(epoch_sks))
        nmi_history.append(nmi)
        pe_history.append(mean_pe)
        sks_history.append(mean_sks)
        print(
            f"  >>> Epoch {epoch + 1}: NMI={nmi:.4f}"
            f"  PE={mean_pe:.4f}  SKS={mean_sks:.1f}"
            f"  t={time.time() - t0:.1f}s"
        )

        if epoch == epochs - 1:
            firing_rates_final = epoch_rates
            labels_final = true_epoch
            # Сохраняем perm чтобы потом правильно показать изображения
            perm_last = perm.numpy()

    # Финальная кластеризация (больше итераций)
    km_final = KMeans(n_clusters=n_classes, n_init=20, random_state=42)
    cluster_labels = km_final.fit_predict(firing_rates_final)

    return {
        "nmi_history": nmi_history,
        "pe_history": pe_history,
        "sks_history": sks_history,
        "firing_rates": firing_rates_final,
        "true_labels": labels_final,
        "cluster_labels": cluster_labels,
        "perm_last": perm_last,   # perm-индексы последней эпохи
    }

# --------------------------------------------------------------------------- #
# Визуализация                                                                 #
# --------------------------------------------------------------------------- #

def _fig_to_b64(fig) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110,
                facecolor=fig.get_facecolor())
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return data


def plot_tsne(
    firing_rates: np.ndarray,
    true_labels: np.ndarray,
    cluster_labels: np.ndarray,
    n_classes: int,
    class_names: list[str],
) -> str:
    print("[VIZ] PCA (50 components)...")
    n_components = min(50, firing_rates.shape[1], firing_rates.shape[0] - 1)
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(firing_rates)
    var = pca.explained_variance_ratio_.sum()
    print(f"[VIZ] PCA объясняет {var:.1%} дисперсии")

    print("[VIZ] t-SNE (n_iter=1000)...")
    perp = min(30, len(firing_rates) // 5)
    tsne = TSNE(n_components=2, perplexity=perp, n_iter=1000,
                random_state=42, verbose=0)
    X_2d = tsne.fit_transform(X_pca)
    print("[VIZ] t-SNE готово")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor="#0d0d1a")

    for ax, lbl, title in [
        (axes[0], true_labels, "Истинные цифры (метки не использовались)"),
        (axes[1], cluster_labels, "Кластеры СНКС (самоорганизация)"),
    ]:
        ax.set_facecolor("#0a0a18")
        for c in range(n_classes):
            m = lbl == c
            ax.scatter(
                X_2d[m, 0], X_2d[m, 1],
                c=COLORS[c % len(COLORS)],
                label=class_names[c] if c < len(class_names) else str(c),
                alpha=0.75, s=18, linewidths=0,
            )
        ax.set_title(title, color="white", fontsize=12, pad=10)
        ax.tick_params(colors="#555")
        for sp in ax.spines.values():
            sp.set_edgecolor("#2a2a4e")
        ax.grid(True, alpha=0.08, color="white")
        ax.legend(
            title="Класс", title_fontsize=8, fontsize=8,
            framealpha=0.25, labelcolor="white",
            facecolor="#1a1a2e", edgecolor="#333",
            loc="best", ncol=2,
        )

    fig.suptitle(
        "t-SNE: пространство представлений СНКС",
        color="white", fontsize=14, y=1.01,
    )
    plt.tight_layout()
    return _fig_to_b64(fig)


def plot_learning_curves(nmi_history: list, pe_history: list) -> str:
    epochs = list(range(1, len(nmi_history) + 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), facecolor="#0d0d1a")

    for ax in (ax1, ax2):
        ax.set_facecolor("#0a0a18")
        for sp in ax.spines.values():
            sp.set_edgecolor("#2a2a4e")
        ax.tick_params(colors="#666")
        ax.grid(True, alpha=0.1, color="white")
        ax.set_xlabel("Эпоха", color="#888")

    ax1.plot(epochs, nmi_history, "o-", color="#4ECDC4", lw=2.5,
             markersize=9, markerfacecolor="white", zorder=5)
    ax1.axhline(0.6, color="#FF6B6B", ls="--", alpha=0.7, label="Gate > 0.6")
    ax1.fill_between(epochs, nmi_history, alpha=0.12, color="#4ECDC4")
    ax1.set_ylabel("NMI", color="#888")
    ax1.set_title("Качество кластеризации (NMI)", color="white", fontsize=11)
    ax1.set_ylim(0, 1.0)
    ax1.legend(facecolor="#1a1a2e", edgecolor="#444", labelcolor="white", fontsize=9)

    ax2.plot(epochs, pe_history, "o-", color="#F7DC6F", lw=2.5,
             markersize=9, markerfacecolor="white", zorder=5)
    ax2.fill_between(epochs, pe_history, alpha=0.12, color="#F7DC6F")
    ax2.set_ylabel("Mean PE", color="#888")
    ax2.set_title("Ошибка предсказания (↓ = система учится)", color="white", fontsize=11)

    plt.tight_layout()
    return _fig_to_b64(fig)


def plot_heatmap(
    true_labels: np.ndarray,
    cluster_labels: np.ndarray,
    n_classes: int,
    class_names: list[str],
) -> str:
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for t, c in zip(true_labels, cluster_labels):
        matrix[c % n_classes, int(t)] += 1

    row_sums = matrix.sum(axis=1, keepdims=True)
    norm = matrix / np.maximum(row_sums, 1)

    fig, ax = plt.subplots(figsize=(9, 7), facecolor="#0d0d1a")
    ax.set_facecolor("#0a0a18")
    im = ax.imshow(norm, cmap="Blues", aspect="auto", vmin=0, vmax=1)

    for i in range(n_classes):
        for j in range(n_classes):
            v = norm[i, j]
            ax.text(j, i, f"{v:.0%}", ha="center", va="center",
                    fontsize=8, color="white" if v < 0.5 else "#0d1a2e",
                    fontweight="bold" if v > 0.3 else "normal")

    names_short = [n[:4] for n in class_names[:n_classes]]
    ax.set_xticks(range(n_classes))
    ax.set_xticklabels(names_short, color="#aaa", fontsize=10)
    ax.set_yticks(range(n_classes))
    ax.set_yticklabels([f"C{i}" for i in range(n_classes)], color="#aaa", fontsize=9)
    ax.set_xlabel("Истинный класс", color="#aaa", fontsize=11)
    ax.set_ylabel("Кластер СНКС", color="#aaa", fontsize=11)
    ax.set_title("Состав кластеров (% изображений истинного класса)",
                 color="white", fontsize=11, pad=12)
    for sp in ax.spines.values():
        sp.set_edgecolor("#2a2a4e")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Доля", color="#888", fontsize=9)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#888", fontsize=8)
    plt.tight_layout()
    return _fig_to_b64(fig)


def plot_cluster_samples(
    images: torch.Tensor,
    true_labels: np.ndarray,
    cluster_labels: np.ndarray,
    n_classes: int,
    class_names: list[str],
    perm_last: np.ndarray | None = None,
    n_cols: int = 8,
) -> str:
    """perm_last: если задан, images[perm_last[i]] соответствует cluster_labels[i]."""
    fig, axes = plt.subplots(
        n_classes, n_cols + 1,
        figsize=(n_cols * 1.25 + 1.6, n_classes * 1.25),
        facecolor="#0d0d1a",
    )

    for cid in range(n_classes):
        mask = np.where(cluster_labels == cid)[0]
        counts = np.bincount(true_labels[mask].astype(int), minlength=n_classes)
        dom = int(counts.argmax())
        purity = counts[dom] / max(len(mask), 1)

        ax_l = axes[cid, 0]
        ax_l.set_facecolor("#131328")
        dom_name = class_names[dom] if dom < len(class_names) else str(dom)
        ax_l.text(0.5, 0.65, f"C{cid}", ha="center", va="center",
                  fontsize=10, color=COLORS[cid % len(COLORS)], fontweight="bold")
        ax_l.text(0.5, 0.25, f"≈{dom_name}\n{purity:.0%}",
                  ha="center", va="center", fontsize=7, color="#9999bb")
        ax_l.axis("off")

        for j in range(n_cols):
            ax = axes[cid, j + 1]
            ax.set_facecolor("#0a0a18")
            if j < len(mask):
                # perm_last[i] — индекс в исходном массиве images
                img_idx = int(perm_last[mask[j]]) if perm_last is not None else int(mask[j])
                img = images[img_idx].numpy()
                ax.imshow(img, cmap="gray", interpolation="nearest")
                td = int(true_labels[mask[j]])
                for sp in ax.spines.values():
                    sp.set_edgecolor(COLORS[td % len(COLORS)])
                    sp.set_linewidth(1.8)
            else:
                for sp in ax.spines.values():
                    sp.set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(
        "Примеры изображений по кластерам  "
        "(рамка = истинный класс, подпись = доминирующий класс + чистота)",
        color="white", fontsize=10, y=1.01,
    )
    plt.tight_layout(pad=0.25)
    return _fig_to_b64(fig)


def compute_purity_table(
    true_labels: np.ndarray,
    cluster_labels: np.ndarray,
    n_classes: int,
    class_names: list[str],
) -> list[dict]:
    rows = []
    for cid in range(n_classes):
        mask = np.where(cluster_labels == cid)[0]
        if len(mask) == 0:
            rows.append({"cluster": cid, "size": 0, "dominant": "—", "purity": 0.0})
            continue
        counts = np.bincount(true_labels[mask].astype(int), minlength=n_classes)
        dom = int(counts.argmax())
        purity = counts[dom] / len(mask)
        rows.append({
            "cluster": cid,
            "size": int(len(mask)),
            "dominant": class_names[dom] if dom < len(class_names) else str(dom),
            "dominant_idx": dom,
            "purity": float(purity),
        })
    return rows

# --------------------------------------------------------------------------- #
# HTML Report                                                                  #
# --------------------------------------------------------------------------- #

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: #080818;
  color: #d0d0f0;
  font-family: 'Segoe UI', 'Inter', system-ui, monospace;
  line-height: 1.6;
}
.header {
  background: linear-gradient(135deg, #0d1f4a 0%, #1a0a3e 60%, #0d2a3a 100%);
  padding: 40px 60px 32px;
  border-bottom: 2px solid #1e1e5e;
}
.header h1 {
  font-size: 2em;
  font-weight: 800;
  background: linear-gradient(90deg, #4ECDC4, #45B7D1, #BB8FCE);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin-bottom: 10px;
}
.header p { color: #7799bb; font-size: 0.95em; margin: 4px 0; }
.header .sub { color: #5566aa; font-size: 0.85em; margin-top: 8px; font-style: italic; }
.metrics {
  display: flex; gap: 14px; padding: 22px 60px;
  background: #0a0a18; border-bottom: 1px solid #181838; flex-wrap: wrap;
}
.mc {
  background: #111128; border: 1px solid #242450; border-radius: 10px;
  padding: 14px 22px; min-width: 130px; text-align: center;
}
.mc .val {
  font-size: 1.85em; font-weight: 700;
  background: linear-gradient(90deg, #4ECDC4, #45B7D1);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  background-clip: text;
}
.mc.pass .val { background: linear-gradient(90deg, #4ECDC4, #96CEB4);
  -webkit-background-clip: text; background-clip: text; }
.mc .lbl { font-size: 0.72em; color: #6677aa; text-transform: uppercase;
  letter-spacing: 0.06em; margin-top: 4px; }
.content { max-width: 1400px; margin: 0 auto; padding: 30px 60px; }
.card {
  background: #0e0e22; border: 1px solid #202048; border-radius: 12px;
  padding: 26px; margin-bottom: 26px;
  box-shadow: 0 4px 24px rgba(0,0,0,0.5);
}
.card h2 { font-size: 1.1em; color: #7799cc; margin-bottom: 6px; font-weight: 600; }
.card .desc { color: #556688; font-size: 0.85em; margin-bottom: 14px;
  padding-bottom: 12px; border-bottom: 1px solid #181838; }
.card img { width: 100%; border-radius: 6px; border: 1px solid #1a1a3e; }
.two { display: grid; grid-template-columns: 1fr 1fr; gap: 22px; }
table { width: 100%; border-collapse: collapse; margin-top: 6px; }
th { background: #111128; color: #7799cc; font-size: 0.78em; text-transform: uppercase;
  letter-spacing: 0.05em; padding: 8px 12px; border-bottom: 1px solid #252545; text-align: left; }
td { padding: 7px 12px; font-size: 0.87em; border-bottom: 1px solid #141430; color: #9999cc; }
tr:hover td { background: #111130; }
.badge {
  display: inline-block; padding: 2px 10px; border-radius: 20px;
  font-size: 0.75em; font-weight: 700; letter-spacing: 0.05em; margin-left: 8px;
}
.badge-pass { background: #1a3a1a; color: #4ECDC4; border: 1px solid #2a5a2a; }
.badge-fail { background: #3a1a1a; color: #FF6B6B; border: 1px solid #5a2a2a; }
.cfg {
  background: #070714; border: 1px solid #1a1a3e; border-radius: 8px;
  padding: 16px; font-family: monospace; font-size: 0.8em; color: #7788bb;
  overflow-x: auto; white-space: pre;
}
.footer { text-align: center; padding: 22px; color: #333355; font-size: 0.78em;
  border-top: 1px solid #141430; margin-top: 36px; }
.purity-bar { display: inline-block; height: 10px; border-radius: 3px;
  background: #4ECDC4; margin-left: 8px; vertical-align: middle; }
"""

_HTML = """<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<title>СНКС Demo — Unsupervised {mode}</title>
<style>{css}</style>
</head>
<body>
<div class="header">
  <h1>СНКС: Самоорганизующаяся Нейросеть</h1>
  <p>Распознавание <strong>{mode}</strong> <em>без меток</em> — FHN-осцилляторы + STDP</p>
  <p>Ни одной пары (входные данные, метка) не было предоставлено при обучении.</p>
  <div class="sub">Архитектура: {num_nodes:,} FHN-осцилляторов · STDP (Spike-Timing-Dependent Plasticity)
  · Rate-based SKS detection · Без backpropagation · Без градиентного спуска</div>
</div>
<div class="metrics">
  <div class="mc {nmi_cls}"><div class="val">{nmi:.3f}</div><div class="lbl">NMI финальный</div></div>
  <div class="mc"><div class="val">{num_nodes:,}</div><div class="lbl">Осцилляторов</div></div>
  <div class="mc"><div class="val">{n_images:,}</div><div class="lbl">Изображений</div></div>
  <div class="mc"><div class="val">{epochs}</div><div class="lbl">Эпох</div></div>
  <div class="mc"><div class="val">{n_classes}</div><div class="lbl">Классов</div></div>
  <div class="mc"><div class="val">{avg_purity:.0%}</div><div class="lbl">Средняя чистота</div></div>
  <div class="mc"><div class="val">{train_time}</div><div class="lbl">Время обучения</div></div>
</div>
<div class="content">

<div class="card">
  <h2>Пространство представлений (t-SNE)
    <span class="badge {badge_cls}">{nmi_status}</span>
  </h2>
  <div class="desc">
    Каждая точка — одно изображение в 2D-пространстве (через PCA → t-SNE).
    <b>Слева:</b> окраска по истинным меткам — система их <em>никогда не видела</em>.
    <b>Справа:</b> кластеры, сформированные СНКС самостоятельно.
    Совпадение паттернов = система научилась осмысленным группам без учителя.
  </div>
  <img src="data:image/png;base64,{tsne_b64}" alt="t-SNE">
</div>

<div class="two">
  <div class="card">
    <h2>Прогресс обучения</h2>
    <div class="desc">NMI растёт — осцилляторы специализируются через STDP.
      PE (ошибка предсказания) снижается по мере формирования паттернов.</div>
    <img src="data:image/png;base64,{nmi_b64}" alt="Learning curves">
  </div>
  <div class="card">
    <h2>Состав кластеров</h2>
    <div class="desc">Строки = кластеры СНКС, столбцы = истинные классы.
      Диагональный паттерн указывает на специализацию кластеров.</div>
    <img src="data:image/png;base64,{heatmap_b64}" alt="Heatmap">
  </div>
</div>

<div class="card">
  <h2>Примеры изображений по кластерам</h2>
  <div class="desc">Каждая строка — один кластер СНКС. Рамка = истинный класс.
    Подпись = доминирующий класс (чистота %).
    Хорошая кластеризация: в строке преобладают изображения одного класса.</div>
  <img src="data:image/png;base64,{samples_b64}" alt="Samples">
</div>

<div class="two">
  <div class="card">
    <h2>Чистота кластеров</h2>
    <div class="desc">Для каждого кластера: доминирующий класс и % изображений этого класса.</div>
    <table>
      <tr><th>Кластер</th><th>Размер</th><th>Доминирует</th><th>Чистота</th></tr>
      {purity_rows}
    </table>
  </div>
  <div class="card">
    <h2>Конфигурация системы</h2>
    <div class="desc">Параметры данного запуска.</div>
    <div class="cfg">{config_json}</div>
  </div>
</div>

</div>
<div class="footer">
  СНКС MVP · {timestamp} · FHN+STDP · PyTorch · No backpropagation · No labels · NMI={nmi:.4f}
</div>
</body>
</html>"""


def build_report(
    results: dict,
    images: torch.Tensor,
    n_classes: int,
    class_names: list[str],
    epochs: int,
    num_nodes: int,
    train_time_sec: float,
    config_dict: dict,
    mode: str,
    output_path: Path,
) -> None:
    nmi = results["nmi_history"][-1]
    purity_rows_data = compute_purity_table(
        results["true_labels"], results["cluster_labels"], n_classes, class_names
    )
    avg_purity = float(np.mean([r["purity"] for r in purity_rows_data]))

    print("[VIZ] t-SNE...")
    tsne_b64 = plot_tsne(
        results["firing_rates"], results["true_labels"], results["cluster_labels"],
        n_classes, class_names,
    )
    print("[VIZ] кривые обучения...")
    nmi_b64 = plot_learning_curves(results["nmi_history"], results["pe_history"])
    print("[VIZ] тепловая карта...")
    heatmap_b64 = plot_heatmap(
        results["true_labels"], results["cluster_labels"], n_classes, class_names
    )
    print("[VIZ] примеры изображений...")
    samples_b64 = plot_cluster_samples(
        images, results["true_labels"], results["cluster_labels"], n_classes, class_names,
        perm_last=results.get("perm_last"),
    )

    if train_time_sec < 60:
        train_time_str = f"{train_time_sec:.0f}s"
    elif train_time_sec < 3600:
        train_time_str = f"{train_time_sec / 60:.1f}m"
    else:
        train_time_str = f"{train_time_sec / 3600:.1f}h"

    purity_rows_html = ""
    for r in purity_rows_data:
        bar_w = int(r["purity"] * 80)
        purity_rows_html += (
            f"<tr><td>C{r['cluster']}</td>"
            f"<td>{r['size']}</td>"
            f"<td>{r['dominant']}</td>"
            f"<td>{r['purity']:.0%}"
            f"<span class='purity-bar' style='width:{bar_w}px'></span></td></tr>"
        )

    nmi_pass = nmi > 0.6
    html = _HTML.format(
        css=_CSS,
        mode=mode,
        nmi=nmi,
        nmi_cls="pass" if nmi_pass else "",
        nmi_status="PASS ✓" if nmi_pass else "FAIL",
        badge_cls="badge-pass" if nmi_pass else "badge-fail",
        num_nodes=num_nodes,
        n_images=images.shape[0],
        epochs=epochs,
        n_classes=n_classes,
        avg_purity=avg_purity,
        train_time=train_time_str,
        tsne_b64=tsne_b64,
        nmi_b64=nmi_b64,
        heatmap_b64=heatmap_b64,
        samples_b64=samples_b64,
        purity_rows=purity_rows_html,
        config_json=json.dumps(config_dict, indent=2),
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    output_path.write_text(html, encoding="utf-8")
    print(f"[VIZ] Отчёт сохранён: {output_path}")

# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def run(
    device: str = "auto",
    num_nodes: int = 50000,
    n_per_class: int = 200,
    epochs: int = 5,
    output_dir: str = "demo_output",
) -> float:
    from snks.device import get_device

    if device == "auto":
        device = str(get_device())

    print(f"[DEMO] СНКС MNIST Demo")
    print(f"[DEMO] Device: {device}  Nodes: {num_nodes:,}")
    print(f"[DEMO] {n_per_class}/class × 10 = {n_per_class * 10} images  Epochs: {epochs}")

    if not HAS_MPL:
        print("[DEMO] ОШИБКА: установите matplotlib: pip install matplotlib")
        return 0.0

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Данные
    print("[DEMO] Загрузка MNIST...")
    loader = MnistLoader(data_root="data/", target_size=64, seed=42)
    images, labels = loader.load("train", n_per_class=n_per_class)
    n_classes = int(labels.max().item()) + 1
    print(f"[DEMO] Загружено {images.shape[0]} изображений, {n_classes} классов")

    # Pipeline
    config = make_config(device, num_nodes)
    pipeline = Pipeline(config)

    # Обучение
    t0 = time.time()
    results = train_and_collect(pipeline, images, labels, epochs, n_classes)
    train_time = time.time() - t0

    final_nmi = results["nmi_history"][-1]
    print(f"\n[DEMO] Обучение завершено за {train_time:.1f}s")
    print(f"[DEMO] Финальный NMI: {final_nmi:.4f}  {'PASS' if final_nmi > 0.6 else 'FAIL'}")

    config_dict = {
        "device": device,
        "num_nodes": num_nodes,
        "n_per_class": n_per_class,
        "total_images": images.shape[0],
        "epochs": epochs,
        "oscillator": "FHN (FitzHugh-Nagumo)",
        "learning": "STDP",
        "coupling_strength": 0.05,
        "dt": 0.01,
        "steps_per_cycle": 200,
        "sdr_size": 8192,
        "detection": "rate-based (mean+3σ)",
        "final_nmi": float(final_nmi),
        "nmi_history": [float(x) for x in results["nmi_history"]],
        "train_time_sec": float(train_time),
    }

    build_report(
        results=results,
        images=images,
        n_classes=n_classes,
        class_names=DIGIT_NAMES[:n_classes],
        epochs=epochs,
        num_nodes=num_nodes,
        train_time_sec=train_time,
        config_dict=config_dict,
        mode="рукописных цифр MNIST",
        output_path=out / "mnist_report.html",
    )

    # Метрики JSON
    metrics = {
        "final_nmi": float(final_nmi),
        "nmi_history": [float(x) for x in results["nmi_history"]],
        "pe_history": [float(x) for x in results["pe_history"]],
        "sks_history": [float(x) for x in results["sks_history"]],
        "train_time_sec": float(train_time),
        "gate_pass": float(final_nmi) > 0.6,
    }
    (out / "mnist_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"\n[DEMO] Результаты:")
    print(f"  HTML:    {out / 'mnist_report.html'}")
    print(f"  Metrics: {out / 'mnist_metrics.json'}")

    return final_nmi


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="СНКС MNIST Visual Demo")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--nodes", type=int, default=50000)
    parser.add_argument("--per-class", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--output", default="demo_output")
    args = parser.parse_args()

    run(
        device=args.device,
        num_nodes=args.nodes,
        n_per_class=args.per_class,
        epochs=args.epochs,
        output_dir=args.output,
    )
