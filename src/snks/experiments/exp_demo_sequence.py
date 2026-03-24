"""СНКС Demo: Sequence Prediction.

Система учит последовательность 0→1→2→3→4→5→6→0→... и предсказывает
следующий стимул. Генерирует HTML-отчёт с матрицей переходов и
примерами правильных предсказаний.

Ничего не заходит: ни метки, ни явная структура последовательности.
Система обнаруживает порядок через STDP (усиление связей при повторных
co-firing).

Usage:
    python -m snks.experiments.exp_demo_sequence
    python -m snks.experiments.exp_demo_sequence --nodes 15000 --length 7 --repeats 30
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

from snks.daf.types import DafConfig, EncoderConfig, PipelineConfig, PredictionConfig, SKSConfig
from snks.data.stimuli import GratingGenerator
from snks.data.sequences import SequenceGenerator
from snks.pipeline.runner import Pipeline
from snks.experiments.exp_demo_mnist import COLORS, _fig_to_b64, _CSS

# --------------------------------------------------------------------------- #
# Config                                                                       #
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
            sdr_size=4096,
            sdr_sparsity=0.04,
            sdr_current_strength=1.0,
        ),
        sks=SKSConfig(
            top_k=min(num_nodes // 2, 3000),
            dbscan_eps=0.3,
            dbscan_min_samples=5,
            min_cluster_size=5,
            coherence_mode="rate",
        ),
        prediction=PredictionConfig(causal_min_confidence=0.2),
        steps_per_cycle=200,
        device=device,
    )

# --------------------------------------------------------------------------- #
# Core logic                                                                   #
# --------------------------------------------------------------------------- #

def collect_vectors(pipeline: Pipeline, images: torch.Tensor) -> np.ndarray:
    n_nodes = pipeline.engine.config.num_nodes
    vecs = np.zeros((len(images), n_nodes), dtype=np.float32)
    for i in range(len(images)):
        pipeline.perception_cycle(images[i])
        fh = pipeline.engine.get_fired_history()
        if fh is not None:
            vecs[i] = fh.float().mean(dim=0).cpu().numpy()
    return vecs


def train_sequence(
    pipeline: Pipeline,
    seq_gen: SequenceGenerator,
    order: list[int],
    n_train_repeats: int,
) -> dict:
    """Train on sequence, collect vectors, return all data needed for analysis."""
    print(f"[SEQ] Обучение: {n_train_repeats} повторений последовательности {order}...")
    images_train, labels_train, _ = seq_gen.deterministic(order, n_repeats=n_train_repeats)
    n = len(images_train)
    t0 = time.time()
    vecs = np.zeros((n, pipeline.engine.config.num_nodes), dtype=np.float32)
    pe_list = []
    for i in range(n):
        result = pipeline.perception_cycle(images_train[i])
        pe_list.append(result.mean_prediction_error)
        fh = pipeline.engine.get_fired_history()
        if fh is not None:
            vecs[i] = fh.float().mean(dim=0).cpu().numpy()
        if (i + 1) % 50 == 0 or (i + 1) == n:
            speed = (i + 1) / max(time.time() - t0, 1e-6)
            print(f"  [{i+1}/{n}] PE={np.mean(pe_list[-20:]):.3f}  {speed:.1f} img/s")

    # Cluster vectors
    k = len(order)
    km = KMeans(n_clusters=k, n_init=20, random_state=42)
    cluster_ids = km.fit_predict(vecs)

    # cluster → dominant label mapping
    train_labels = labels_train.numpy()
    cluster_to_label: dict[int, int] = {}
    for c in range(k):
        mask = cluster_ids == c
        if mask.any():
            counts = np.bincount(train_labels[mask], minlength=max(order) + 1)
            cluster_to_label[c] = int(np.argmax(counts))
        else:
            cluster_to_label[c] = order[c]

    # Transition matrix
    trans = np.zeros((k, k), dtype=np.float64)
    for i in range(len(cluster_ids) - 1):
        trans[cluster_ids[i], cluster_ids[i + 1]] += 1
    row_sums = trans.sum(axis=1, keepdims=True)
    trans_prob = np.where(row_sums > 0, trans / row_sums, 0.0)

    return {
        "km": km,
        "cluster_to_label": cluster_to_label,
        "trans_prob": trans_prob,
        "train_vecs": vecs,
        "train_labels": train_labels,
        "cluster_ids": cluster_ids,
        "mean_pe": float(np.mean(pe_list)),
        "k": k,
    }


def test_sequence(
    pipeline: Pipeline,
    seq_gen: SequenceGenerator,
    order: list[int],
    train_data: dict,
    n_test_repeats: int,
) -> dict:
    """Test prediction on unseen sequence repetitions."""
    images_test, labels_test, next_labels_test = seq_gen.deterministic(
        order, n_repeats=n_test_repeats
    )
    print(f"[SEQ] Тестирование: {n_test_repeats} повторений...")
    vecs_test = collect_vectors(pipeline, images_test)

    km = train_data["km"]
    trans_prob = train_data["trans_prob"]
    c2l = train_data["cluster_to_label"]

    test_cluster_ids = km.predict(vecs_test)
    test_labels = labels_test.numpy()
    next_true = next_labels_test.numpy()

    correct = 0
    total = len(test_labels) - 1
    predictions = []
    for i in range(total):
        cur_c = test_cluster_ids[i]
        pred_c = int(np.argmax(trans_prob[cur_c]))
        pred_label = c2l.get(pred_c, pred_c)
        actual_next = next_true[i]
        ok = pred_label == actual_next
        if ok:
            correct += 1
        predictions.append({
            "step": i,
            "input_label": int(test_labels[i]),
            "predicted_next": int(pred_label),
            "actual_next": int(actual_next),
            "correct": bool(ok),
        })

    accuracy = correct / max(total, 1)
    print(f"[SEQ] Accuracy: {accuracy:.1%} ({correct}/{total})")

    return {
        "predictions": predictions,
        "accuracy": accuracy,
        "vecs_test": vecs_test,
        "test_labels": test_labels,
        "test_cluster_ids": test_cluster_ids,
    }

# --------------------------------------------------------------------------- #
# Visualization                                                                #
# --------------------------------------------------------------------------- #

def get_class_image(gen: GratingGenerator, class_idx: int) -> np.ndarray:
    """Get representative image for a class (no noise)."""
    # Use seed=class_idx for stable, clean image
    g = GratingGenerator(image_size=gen.image_size, seed=class_idx * 1000 + 1)
    img, _ = g.generate(class_idx=class_idx, n_variations=1)
    return img[0].numpy()


def plot_transition_matrix(trans_prob: np.ndarray, order: list[int]) -> str:
    k = len(order)
    fig, ax = plt.subplots(figsize=(7, 6), facecolor="#0d0d1a")
    ax.set_facecolor("#0a0a18")

    im = ax.imshow(trans_prob, cmap="Blues", aspect="auto", vmin=0, vmax=1)

    for i in range(k):
        for j in range(k):
            v = trans_prob[i, j]
            ax.text(j, i, f"{v:.0%}", ha="center", va="center",
                    fontsize=9, color="white" if v < 0.5 else "#0d1a2e",
                    fontweight="bold" if v > 0.3 else "normal")

    labels = [str(o) for o in order]
    ax.set_xticks(range(k))
    ax.set_xticklabels(labels, color="#aaa")
    ax.set_yticks(range(k))
    ax.set_yticklabels(labels, color="#aaa")
    ax.set_xlabel("Следующий стимул", color="#aaa")
    ax.set_ylabel("Текущий стимул", color="#aaa")
    ax.set_title("Матрица переходов (выученная СНКС)\nОжидаем: сдвинутую диагональ",
                 color="white", fontsize=11, pad=10)
    for sp in ax.spines.values():
        sp.set_edgecolor("#2a2a4e")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Вероятность", color="#888", fontsize=9)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#888", fontsize=8)
    plt.tight_layout()
    return _fig_to_b64(fig)


def plot_prediction_examples(
    gen: GratingGenerator,
    predictions: list[dict],
    order: list[int],
    n_show: int = 7,
) -> str:
    """For each starting stimulus in the sequence, show input → predicted → actual."""
    # Pick the first prediction for each input label in order
    seen = set()
    selected = []
    for p in predictions:
        lab = p["input_label"]
        if lab not in seen:
            seen.add(lab)
            selected.append(p)
        if len(selected) >= n_show:
            break

    n = len(selected)
    fig, axes = plt.subplots(n, 3, figsize=(9, n * 1.5 + 0.8), facecolor="#0d0d1a")
    if n == 1:
        axes = axes[None, :]  # ensure 2D

    for row, pred in enumerate(selected):
        inp = get_class_image(gen, pred["input_label"])
        pred_img = get_class_image(gen, pred["predicted_next"])
        act_img = get_class_image(gen, pred["actual_next"])
        correct = pred["correct"]
        border = "#4ECDC4" if correct else "#FF6B6B"
        status_sym = "✓" if correct else "✗"

        for col, (img, title) in enumerate([
            (inp, f"Вход: {pred['input_label']}"),
            (pred_img, f"Предсказано: {pred['predicted_next']} {status_sym}"),
            (act_img, f"Факт: {pred['actual_next']}"),
        ]):
            ax = axes[row, col]
            ax.set_facecolor("#0a0a18")
            ax.imshow(img, cmap="gray", interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(title, color="#9999bb", fontsize=9, pad=4)
            if col == 1:  # prediction column — colored border
                for sp in ax.spines.values():
                    sp.set_edgecolor(border)
                    sp.set_linewidth(2.5)
            else:
                for sp in ax.spines.values():
                    sp.set_edgecolor("#2a2a4e")
                    sp.set_linewidth(1)

    fig.suptitle("Примеры предсказаний (зелёный = правильно, красный = ошибка)",
                 color="white", fontsize=10, y=1.01)
    plt.tight_layout(pad=0.4)
    return _fig_to_b64(fig)


def plot_accuracy_by_position(predictions: list[dict], order: list[int]) -> str:
    """Accuracy when predicting each step in the sequence."""
    k = len(order)
    per_pos: dict[int, list[bool]] = {i: [] for i in range(k)}
    for pred in predictions:
        pos = pred["input_label"] % k if pred["input_label"] in order else None
        if pos is None:
            for i, o in enumerate(order):
                if o == pred["input_label"]:
                    pos = i
                    break
        if pos is not None:
            per_pos[pos].append(pred["correct"])

    acc = [np.mean(per_pos[i]) if per_pos[i] else 0.0 for i in range(k)]
    labels = [f"{order[i]}→{order[(i+1) % k]}" for i in range(k)]

    fig, ax = plt.subplots(figsize=(9, 4), facecolor="#0d0d1a")
    ax.set_facecolor("#0a0a18")

    colors_bar = ["#4ECDC4" if a >= 0.7 else "#F7DC6F" if a >= 0.5 else "#FF6B6B" for a in acc]
    bars = ax.bar(labels, acc, color=colors_bar, edgecolor="#2a3a6e", linewidth=1.5, width=0.6)

    for bar, a in zip(bars, acc):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{a:.0%}", ha="center", va="bottom", color="white", fontsize=11, fontweight="bold")

    ax.axhline(0.7, color="#FF6B6B", ls="--", alpha=0.6, lw=1.5, label="Gate 70%")
    ax.set_ylim(0, 1.15)
    ax.set_xlabel("Переход", color="#888")
    ax.set_ylabel("Accuracy", color="#888")
    ax.set_title("Точность предсказания по переходам последовательности",
                 color="white", fontsize=11, pad=10)
    ax.tick_params(colors="#888")
    for sp in ax.spines.values():
        sp.set_edgecolor("#2a2a4e")
    ax.grid(True, alpha=0.1, axis="y", color="white")
    ax.legend(facecolor="#1a1a2e", edgecolor="#444", labelcolor="white", fontsize=9)
    plt.tight_layout()
    return _fig_to_b64(fig)

# --------------------------------------------------------------------------- #
# HTML Report                                                                  #
# --------------------------------------------------------------------------- #

_SEQ_HTML = """<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<title>СНКС Demo — Sequence Prediction</title>
<style>{css}</style>
</head>
<body>
<div class="header">
  <h1>СНКС: Предсказание последовательностей</h1>
  <p>Система учит повторяющуюся последовательность 0→1→2→…→{seqlen}→0→…</p>
  <div class="sub">
    Обучение: только perception_cycle + STDP. Никакой явной разметки порядка.
    Система сама обнаруживает структуру через временны́е корреляции co-firing.
  </div>
</div>
<div class="metrics">
  <div class="mc {acc_cls}"><div class="val">{accuracy:.0%}</div><div class="lbl">Accuracy</div></div>
  <div class="mc"><div class="val">{seqlen}</div><div class="lbl">Длина последов.</div></div>
  <div class="mc"><div class="val">{n_train}</div><div class="lbl">Train шагов</div></div>
  <div class="mc"><div class="val">{num_nodes:,}</div><div class="lbl">Осцилляторов</div></div>
  <div class="mc"><div class="val">{train_time}</div><div class="lbl">Время</div></div>
</div>
<div class="content">

<div class="two">
  <div class="card">
    <h2>Матрица переходов
      <span class="badge {acc_badge}">{acc_status}</span>
    </h2>
    <div class="desc">
      Выученные вероятности переходов между стимулами.
      Для последовательности 0→1→2→3→... ожидаем сдвинутую диагональ:
      высокую вероятность вдоль суперdiagonali.
    </div>
    <img src="data:image/png;base64,{trans_b64}">
  </div>
  <div class="card">
    <h2>Точность по переходам</h2>
    <div class="desc">
      Для каждого перехода A→B: % случаев, когда система правильно
      предсказала следующий стимул. Gate > 70%.
    </div>
    <img src="data:image/png;base64,{acc_b64}">
  </div>
</div>

<div class="card">
  <h2>Примеры предсказаний</h2>
  <div class="desc">
    Для каждого входного стимула: что предсказала система и что было на самом деле.
    Зелёная рамка = правильно. Красная = ошибка.
    Стимулы — ориентированные решётки Gabor (10 углов: 0°, 18°, ..., 162°).
  </div>
  <img src="data:image/png;base64,{pred_b64}">
</div>

<div class="card">
  <h2>Конфигурация</h2>
  <div class="cfg">{config_json}</div>
</div>

</div>
<div class="footer">
  СНКС MVP · {timestamp} · Sequence Prediction · Accuracy={accuracy:.1%} · STDP+FHN
</div>
</body>
</html>"""


def build_seq_report(
    accuracy: float,
    seqlen: int,
    n_train: int,
    num_nodes: int,
    train_time_sec: float,
    trans_b64: str,
    acc_b64: str,
    pred_b64: str,
    config_dict: dict,
    output_path: Path,
) -> None:
    ok = accuracy >= 0.7
    if train_time_sec < 60:
        tt = f"{train_time_sec:.0f}s"
    elif train_time_sec < 3600:
        tt = f"{train_time_sec/60:.1f}m"
    else:
        tt = f"{train_time_sec/3600:.1f}h"

    html = _SEQ_HTML.format(
        css=_CSS,
        accuracy=accuracy,
        acc_cls="pass" if ok else "warn",
        acc_badge="badge-pass" if ok else "badge-fail",
        acc_status="PASS ✓" if ok else "FAIL",
        seqlen=seqlen,
        n_train=n_train,
        num_nodes=num_nodes,
        train_time=tt,
        trans_b64=trans_b64,
        acc_b64=acc_b64,
        pred_b64=pred_b64,
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
    num_nodes: int = 15000,
    seq_length: int = 7,
    n_train_repeats: int = 40,
    n_test_repeats: int = 10,
    output_dir: str = "demo_output",
) -> float:
    from snks.device import get_device

    if device == "auto":
        device = str(get_device())

    print(f"[DEMO] СНКС Sequence Prediction Demo")
    print(f"[DEMO] Device: {device}  Nodes: {num_nodes:,}")
    print(f"[DEMO] Sequence length: {seq_length}  Train repeats: {n_train_repeats}")

    if not HAS_MPL:
        print("[DEMO] ОШИБКА: установите matplotlib")
        return 0.0

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    gen = GratingGenerator(image_size=64, seed=42)
    seq_gen = SequenceGenerator(gen, seed=42)
    order = list(range(seq_length))

    config = make_config(device, num_nodes)
    pipeline = Pipeline(config)

    t0 = time.time()

    # Train
    train_data = train_sequence(pipeline, seq_gen, order, n_train_repeats)

    # Test
    test_data = test_sequence(pipeline, seq_gen, order, train_data, n_test_repeats)

    train_time = time.time() - t0
    accuracy = test_data["accuracy"]

    print(f"\n[DEMO] Accuracy: {accuracy:.1%}  ({'PASS' if accuracy >= 0.7 else 'FAIL'})")
    print(f"[DEMO] Время: {train_time:.1f}s")

    # Visualizations
    print("[VIZ] Матрица переходов...")
    trans_b64 = plot_transition_matrix(train_data["trans_prob"], order)

    print("[VIZ] Accuracy по позициям...")
    acc_b64 = plot_accuracy_by_position(test_data["predictions"], order)

    print("[VIZ] Примеры предсказаний...")
    pred_b64 = plot_prediction_examples(gen, test_data["predictions"], order, n_show=seq_length)

    config_dict = {
        "device": device,
        "num_nodes": num_nodes,
        "seq_length": seq_length,
        "n_train_repeats": n_train_repeats,
        "n_test_repeats": n_test_repeats,
        "n_train_steps": seq_length * n_train_repeats,
        "oscillator": "FHN",
        "learning": "STDP",
        "stimuli": "GratingGenerator (oriented Gabor gratings)",
        "accuracy": float(accuracy),
        "train_time_sec": float(train_time),
    }

    build_seq_report(
        accuracy=accuracy,
        seqlen=seq_length,
        n_train=seq_length * n_train_repeats,
        num_nodes=num_nodes,
        train_time_sec=train_time,
        trans_b64=trans_b64,
        acc_b64=acc_b64,
        pred_b64=pred_b64,
        config_dict=config_dict,
        output_path=out / "sequence_report.html",
    )

    metrics = {
        "accuracy": float(accuracy),
        "seq_length": seq_length,
        "n_train_steps": seq_length * n_train_repeats,
        "train_time_sec": float(train_time),
        "gate_pass": accuracy >= 0.7,
    }
    (out / "sequence_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    print(f"\n[DEMO] Результаты:")
    print(f"  HTML:    {out / 'sequence_report.html'}")
    print(f"  Metrics: {out / 'sequence_metrics.json'}")

    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="СНКС Sequence Prediction Demo")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--nodes", type=int, default=15000)
    parser.add_argument("--length", type=int, default=7)
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("--test-repeats", type=int, default=10)
    parser.add_argument("--output", default="demo_output")
    args = parser.parse_args()

    run(
        device=args.device,
        num_nodes=args.nodes,
        seq_length=args.length,
        n_train_repeats=args.repeats,
        n_test_repeats=args.test_repeats,
        output_dir=args.output,
    )
