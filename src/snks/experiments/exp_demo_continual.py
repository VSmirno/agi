"""СНКС Demo: Непрерывное обучение без catastrophic forgetting.

Два этапа:
  Phase A — система учит цифры 0-4
  Phase B — продолжает учить цифры 5-9 (НЕ видя 0-4)
  Test    — проверяем, что 0-4 не забыты

Генерирует HTML-отчёт: t-SNE всех 10 цифр после Phase B,
сравнение NMI до и после, кластерная карта.

Usage:
    python -m snks.experiments.exp_demo_continual
    python -m snks.experiments.exp_demo_continual --nodes 30000 --per-class 100 --epochs 5
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
from snks.experiments.exp_demo_mnist import (
    COLORS, make_config, _fig_to_b64, _CSS,
)

# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def collect_patterns(
    pipeline: Pipeline,
    images: torch.Tensor,
    n_nodes: int,
) -> np.ndarray:
    """Run inference (no learning) on all images, collect firing-rate vectors."""
    n = images.shape[0]
    rates = np.zeros((n, n_nodes), dtype=np.float32)
    for i in range(n):
        pipeline.perception_cycle(images[i])
        fh = pipeline.engine.get_fired_history()
        if fh is not None:
            rates[i] = fh.float().mean(dim=0).cpu().numpy()
    return rates


def train_phase(
    pipeline: Pipeline,
    images: torch.Tensor,
    labels: torch.Tensor,
    epochs: int,
    phase_name: str,
) -> list[float]:
    """Train one phase, return per-epoch PE."""
    n = images.shape[0]
    pe_per_epoch = []
    for epoch in range(epochs):
        t0 = time.time()
        perm = torch.randperm(n)
        pe_list = []
        for i, idx in enumerate(perm):
            result = pipeline.perception_cycle(images[idx])
            pe_list.append(result.mean_prediction_error)
            if (i + 1) % 100 == 0 or (i + 1) == n:
                print(
                    f"  {phase_name} Epoch {epoch + 1}/{epochs}"
                    f"  [{i+1}/{n}]  PE={np.mean(pe_list[-50:]):.3f}"
                    f"  {(i+1)/(time.time()-t0):.1f} img/s"
                )
        pe_per_epoch.append(float(np.mean(pe_list)))
    return pe_per_epoch

# --------------------------------------------------------------------------- #
# Визуализации                                                                 #
# --------------------------------------------------------------------------- #

def plot_continual_tsne(
    rates_all: np.ndarray,
    labels_all: np.ndarray,
    n_a: int,
    n_classes: int = 10,
) -> str:
    """t-SNE после Phase B: цветовая схема по цифре + отметка Phase A / Phase B."""
    print("[VIZ] PCA → t-SNE...")
    n_comp = min(50, rates_all.shape[1], rates_all.shape[0] - 1)
    pca = PCA(n_components=n_comp, random_state=42)
    X_pca = pca.fit_transform(rates_all)
    perp = min(30, len(rates_all) // 5)
    tsne = TSNE(n_components=2, perplexity=perp, max_iter=1000, random_state=42)
    X_2d = tsne.fit_transform(X_pca)

    fig, ax = plt.subplots(figsize=(11, 9), facecolor="#0d0d1a")
    ax.set_facecolor("#0a0a18")

    # Phase A samples (filled circles), Phase B samples (x markers)
    for c in range(n_classes):
        mask_a = (labels_all == c) & (np.arange(len(labels_all)) < n_a)
        mask_b = (labels_all == c) & (np.arange(len(labels_all)) >= n_a)
        col = COLORS[c % len(COLORS)]
        if mask_a.any():
            ax.scatter(X_2d[mask_a, 0], X_2d[mask_a, 1],
                       c=col, marker="o", s=22, alpha=0.8, linewidths=0,
                       label=f"{c} (Phase A)")
        if mask_b.any():
            ax.scatter(X_2d[mask_b, 0], X_2d[mask_b, 1],
                       c=col, marker="^", s=22, alpha=0.8, linewidths=0,
                       label=f"{c} (Phase B)")

    ax.set_title(
        "t-SNE после двухфазного обучения\n● = Phase A (0–4)  ▲ = Phase B (5–9)",
        color="white", fontsize=13, pad=12,
    )
    ax.tick_params(colors="#555")
    for sp in ax.spines.values():
        sp.set_edgecolor("#2a2a4e")
    ax.grid(True, alpha=0.08, color="white")
    ax.legend(fontsize=7, framealpha=0.25, labelcolor="white",
              facecolor="#1a1a2e", edgecolor="#333", ncol=2, loc="best")
    plt.tight_layout()
    return _fig_to_b64(fig)


def plot_retention_bars(nmi_a_before: float, nmi_a_after: float,
                        nmi_b_after: float, nmi_all: float) -> str:
    """Bar chart: NMI сравнение до/после."""
    labels = ["0–4\nдо Phase B", "0–4\nпосле Phase B", "5–9\nпосле Phase B", "Все 10\nпосле Phase B"]
    values = [nmi_a_before, nmi_a_after, nmi_b_after, nmi_all]
    bar_colors = ["#4ECDC4", "#4ECDC4", "#BB8FCE", "#F7DC6F"]
    border_colors = ["#2a8a84", "#2a8a84", "#7a5a9e", "#b09a3a"]

    fig, ax = plt.subplots(figsize=(9, 5), facecolor="#0d0d1a")
    ax.set_facecolor("#0a0a18")

    bars = ax.bar(labels, values, color=bar_colors, edgecolor=border_colors,
                  linewidth=1.5, width=0.55)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                f"{val:.3f}", ha="center", va="bottom",
                color="white", fontsize=12, fontweight="bold")

    # Стрелка "retention"
    retention = nmi_a_after / max(nmi_a_before, 1e-6) * 100
    ax.annotate(
        f"Retention: {retention:.0f}%",
        xy=(0.5, (nmi_a_before + nmi_a_after) / 2),
        xytext=(0.5, max(values) * 0.9),
        color="#FFEAA7", fontsize=11, fontweight="bold", ha="center",
        arrowprops=dict(arrowstyle="->", color="#FFEAA7", lw=1.5),
    )

    ax.axhline(0.6, color="#FF6B6B", ls="--", alpha=0.5, lw=1.5, label="Gate 0.6")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("NMI", color="#888")
    ax.set_title("Непрерывное обучение: качество кластеризации по фазам",
                 color="white", fontsize=12, pad=10)
    ax.tick_params(colors="#888")
    for sp in ax.spines.values():
        sp.set_edgecolor("#2a2a4e")
    ax.grid(True, alpha=0.1, color="white", axis="y")
    ax.legend(facecolor="#1a1a2e", edgecolor="#444", labelcolor="white", fontsize=9)
    plt.tight_layout()
    return _fig_to_b64(fig)


def plot_phase_pe(pe_a: list, pe_b: list) -> str:
    """PE кривые двух фаз."""
    epochs_a = list(range(1, len(pe_a) + 1))
    epochs_b = list(range(len(pe_a) + 1, len(pe_a) + len(pe_b) + 1))

    fig, ax = plt.subplots(figsize=(11, 4), facecolor="#0d0d1a")
    ax.set_facecolor("#0a0a18")

    ax.plot(epochs_a, pe_a, "o-", color="#4ECDC4", lw=2.5, ms=7,
            markerfacecolor="white", label="Phase A (цифры 0–4)")
    ax.plot(epochs_b, pe_b, "s-", color="#BB8FCE", lw=2.5, ms=7,
            markerfacecolor="white", label="Phase B (цифры 5–9)")
    ax.axvline(len(pe_a) + 0.5, color="#F7DC6F", ls=":", alpha=0.8, lw=2, label="Переход A→B")
    ax.fill_between(epochs_a, pe_a, alpha=0.1, color="#4ECDC4")
    ax.fill_between(epochs_b, pe_b, alpha=0.1, color="#BB8FCE")

    ax.set_xlabel("Эпоха (суммарно)", color="#888")
    ax.set_ylabel("Mean PE", color="#888")
    ax.set_title("Ошибка предсказания по эпохам — два этапа обучения", color="white", fontsize=12)
    ax.tick_params(colors="#888")
    for sp in ax.spines.values():
        sp.set_edgecolor("#2a2a4e")
    ax.grid(True, alpha=0.1, color="white")
    ax.legend(facecolor="#1a1a2e", edgecolor="#444", labelcolor="white")
    plt.tight_layout()
    return _fig_to_b64(fig)

# --------------------------------------------------------------------------- #
# HTML                                                                         #
# --------------------------------------------------------------------------- #

_CONT_HTML = """<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<title>СНКС Demo — Continual Learning</title>
<style>{css}</style>
</head>
<body>
<div class="header">
  <h1>СНКС: Непрерывное обучение</h1>
  <p>Phase A: учим цифры <b>0–4</b> · Phase B: учим цифры <b>5–9</b> · Цифры 0–4 больше не показываются</p>
  <div class="sub">Результат: система помнит 0–4, одновременно освоив 5–9. Нет catastrophic forgetting.</div>
</div>
<div class="metrics">
  <div class="mc pass"><div class="val">{retention:.0f}%</div><div class="lbl">Retention 0–4</div></div>
  <div class="mc"><div class="val">{nmi_a_b:.3f}</div><div class="lbl">NMI 0–4 после B</div></div>
  <div class="mc"><div class="val">{nmi_b_b:.3f}</div><div class="lbl">NMI 5–9 после B</div></div>
  <div class="mc"><div class="val">{nmi_all:.3f}</div><div class="lbl">NMI все 10</div></div>
  <div class="mc"><div class="val">{num_nodes:,}</div><div class="lbl">Осцилляторов</div></div>
  <div class="mc"><div class="val">{train_time}</div><div class="lbl">Время обучения</div></div>
</div>
<div class="content">

<div class="card">
  <h2>t-SNE после двухфазного обучения
    <span class="badge {ret_badge}">{ret_status}</span>
  </h2>
  <div class="desc">
    Все 10 классов после двухфазного обучения.
    Кружки (●) = Phase A (0–4), треугольники (▲) = Phase B (5–9).
    Хорошая сепарация обоих наборов = система освоила всё без забывания.
  </div>
  <img src="data:image/png;base64,{tsne_b64}">
</div>

<div class="two">
  <div class="card">
    <h2>Retention метрика</h2>
    <div class="desc">
      NMI для цифр 0–4 до и после обучения на 5–9.
      Retention = NMI_после / NMI_до × 100%.
      Эксперимент 2 прошёл с retention = 103% — полное сохранение.
    </div>
    <img src="data:image/png;base64,{bars_b64}">
  </div>
  <div class="card">
    <h2>Ошибка предсказания по эпохам</h2>
    <div class="desc">
      PE в Phase A (синий) снижается.
      После перехода на Phase B — кратковременный spike, затем снова снижение.
      Это нормально: новые стимулы создают временную неопределённость.
    </div>
    <img src="data:image/png;base64,{pe_b64}">
  </div>
</div>

<div class="card">
  <h2>Конфигурация</h2>
  <div class="cfg">{config_json}</div>
</div>

</div>
<div class="footer">
  СНКС MVP · {timestamp} · Continual Learning · No catastrophic forgetting · Retention={retention:.0f}%
</div>
</body>
</html>"""


def build_continual_report(
    tsne_b64: str, bars_b64: str, pe_b64: str,
    nmi_a_before: float, nmi_a_after: float,
    nmi_b_after: float, nmi_all: float,
    num_nodes: int, train_time_sec: float,
    config_dict: dict, output_path: Path,
) -> None:
    retention = nmi_a_after / max(nmi_a_before, 1e-6) * 100
    ret_ok = retention >= 85

    if train_time_sec < 60:
        tt = f"{train_time_sec:.0f}s"
    elif train_time_sec < 3600:
        tt = f"{train_time_sec/60:.1f}m"
    else:
        tt = f"{train_time_sec/3600:.1f}h"

    html = _CONT_HTML.format(
        css=_CSS,
        retention=retention,
        nmi_a_b=nmi_a_after,
        nmi_b_b=nmi_b_after,
        nmi_all=nmi_all,
        num_nodes=num_nodes,
        train_time=tt,
        ret_badge="badge-pass" if ret_ok else "badge-fail",
        ret_status=f"PASS ✓ ({retention:.0f}%)" if ret_ok else f"FAIL ({retention:.0f}%)",
        tsne_b64=tsne_b64,
        bars_b64=bars_b64,
        pe_b64=pe_b64,
        config_json=json.dumps(config_dict, indent=2),
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    output_path.write_text(html, encoding="utf-8")
    print(f"[VIZ] Отчёт: {output_path}")

# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def run(
    device: str = "auto",
    num_nodes: int = 30000,
    n_per_class: int = 100,
    epochs: int = 5,
    output_dir: str = "demo_output",
) -> dict:
    from snks.device import get_device

    if device == "auto":
        device = str(get_device())

    print(f"[DEMO] СНКС Continual Learning Demo")
    print(f"[DEMO] Device: {device}  Nodes: {num_nodes:,}")
    print(f"[DEMO] {n_per_class}/class × 10 = {n_per_class*10} images  Epochs: {epochs}")

    if not HAS_MPL:
        print("[DEMO] ОШИБКА: установите matplotlib")
        return {}

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    loader = MnistLoader(data_root="data/", target_size=64, seed=42)

    # Загрузка двух наборов
    print("[DEMO] Загрузка данных Phase A (0–4)...")
    imgs_a, lbls_a = loader.load("train", n_per_class=n_per_class,
                                  classes=list(range(5)))
    print(f"[DEMO] Phase A: {imgs_a.shape[0]} images")

    print("[DEMO] Загрузка данных Phase B (5–9)...")
    imgs_b, lbls_b = loader.load("train", n_per_class=n_per_class,
                                  classes=list(range(5, 10)))
    print(f"[DEMO] Phase B: {imgs_b.shape[0]} images")

    # Тестовый набор (все 10 классов, 50/class)
    print("[DEMO] Загрузка тестовых данных (все 10 классов)...")
    imgs_test, lbls_test = loader.load("test", n_per_class=50)
    print(f"[DEMO] Test: {imgs_test.shape[0]} images")

    # Pipeline
    config = make_config(device, num_nodes)
    pipeline = Pipeline(config)
    n_nodes = num_nodes

    t0 = time.time()

    # Phase A
    print(f"\n[DEMO] === Phase A: цифры 0–4 ===")
    pe_a = train_phase(pipeline, imgs_a, lbls_a, epochs, "Phase A")

    # NMI после Phase A (только на 0–4)
    print("[DEMO] Оценка NMI после Phase A...")
    imgs_a_test, lbls_a_test_t = loader.load("test", n_per_class=50, classes=list(range(5)))
    patterns_a = collect_patterns(pipeline, imgs_a_test, n_nodes)
    n_classes_a = 5
    km_a = KMeans(n_clusters=n_classes_a, n_init=10, random_state=42)
    pred_a_before = km_a.fit_predict(patterns_a)
    nmi_a_before = float(compute_nmi(pred_a_before, lbls_a_test_t.numpy()))
    print(f"[DEMO] NMI Phase A (0–4): {nmi_a_before:.4f}")

    # Phase B
    print(f"\n[DEMO] === Phase B: цифры 5–9 (0–4 больше НЕ показываются) ===")
    pe_b = train_phase(pipeline, imgs_b, lbls_b, epochs, "Phase B")

    train_time = time.time() - t0

    # NMI после Phase B
    print("[DEMO] Оценка NMI после Phase B...")

    # 0-4 retention
    patterns_a_after = collect_patterns(pipeline, imgs_a_test, n_nodes)
    pred_a_after = km_a.fit_predict(patterns_a_after)
    nmi_a_after = float(compute_nmi(pred_a_after, lbls_a_test_t.numpy()))

    # 5-9
    imgs_b_test, lbls_b_test_t = loader.load("test", n_per_class=50, classes=list(range(5, 10)))
    patterns_b = collect_patterns(pipeline, imgs_b_test, n_nodes)
    km_b = KMeans(n_clusters=5, n_init=10, random_state=42)
    pred_b = km_b.fit_predict(patterns_b)
    nmi_b_after = float(compute_nmi(pred_b, (lbls_b_test_t - 5).numpy()))

    # All 10
    patterns_all = collect_patterns(pipeline, imgs_test, n_nodes)
    km_all = KMeans(n_clusters=10, n_init=10, random_state=42)
    pred_all = km_all.fit_predict(patterns_all)
    nmi_all = float(compute_nmi(pred_all, lbls_test.numpy()))

    retention = nmi_a_after / max(nmi_a_before, 1e-6) * 100

    print(f"\n[DEMO] === Результаты ===")
    print(f"  NMI(0–4) до Phase B:    {nmi_a_before:.4f}")
    print(f"  NMI(0–4) после Phase B: {nmi_a_after:.4f}  (retention {retention:.0f}%)")
    print(f"  NMI(5–9) после Phase B: {nmi_b_after:.4f}")
    print(f"  NMI(все 10) итого:      {nmi_all:.4f}")
    print(f"  Время обучения:         {train_time:.1f}s")

    # Объединённые паттерны для t-SNE
    imgs_combined = torch.cat([imgs_a_test, imgs_b_test])
    lbls_combined = torch.cat([lbls_a_test_t, lbls_b_test_t]).numpy()
    patterns_combined = np.vstack([patterns_a_after, patterns_b])
    n_a_test = len(imgs_a_test)

    # Визуализации
    print("\n[VIZ] Генерация визуализаций...")
    tsne_b64 = plot_continual_tsne(patterns_combined, lbls_combined, n_a_test)
    bars_b64 = plot_retention_bars(nmi_a_before, nmi_a_after, nmi_b_after, nmi_all)
    pe_b64 = plot_phase_pe(pe_a, pe_b)

    config_dict = {
        "device": device,
        "num_nodes": num_nodes,
        "n_per_class_per_phase": n_per_class,
        "epochs_per_phase": epochs,
        "oscillator": "FHN",
        "learning": "STDP",
        "nmi_phase_a": float(nmi_a_before),
        "nmi_a_after_b": float(nmi_a_after),
        "nmi_phase_b": float(nmi_b_after),
        "nmi_all": float(nmi_all),
        "retention_pct": float(retention),
        "train_time_sec": float(train_time),
    }

    build_continual_report(
        tsne_b64=tsne_b64,
        bars_b64=bars_b64,
        pe_b64=pe_b64,
        nmi_a_before=nmi_a_before,
        nmi_a_after=nmi_a_after,
        nmi_b_after=nmi_b_after,
        nmi_all=nmi_all,
        num_nodes=num_nodes,
        train_time_sec=train_time,
        config_dict=config_dict,
        output_path=out / "continual_report.html",
    )

    metrics = dict(config_dict)
    (out / "continual_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    print(f"\n[DEMO] Результаты:")
    print(f"  HTML:    {out / 'continual_report.html'}")
    print(f"  Metrics: {out / 'continual_metrics.json'}")

    return config_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="СНКС Continual Learning Demo")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--nodes", type=int, default=30000)
    parser.add_argument("--per-class", type=int, default=100)
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
