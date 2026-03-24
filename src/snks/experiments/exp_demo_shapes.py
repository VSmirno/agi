"""СНКС Demo: Unsupervised Shape Classification.

Быстрый демо-эксперимент на синтетических геометрических фигурах.
10 классов (круг, квадрат, ...) — никаких меток при обучении.

Генерирует HTML-отчёт. Работает быстрее MNIST — хорош для быстрой проверки.

Usage:
    python -m snks.experiments.exp_demo_shapes
    python -m snks.experiments.exp_demo_shapes --nodes 20000 --per-class 80 --epochs 5
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

try:
    import matplotlib
    matplotlib.use("Agg")
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

from sklearn.cluster import KMeans

from snks.daf.types import DafConfig, EncoderConfig, PipelineConfig, PredictionConfig, SKSConfig
from snks.data.shapes import ShapeGenerator
from snks.pipeline.runner import Pipeline
from snks.sks.metrics import compute_nmi

# Импорт визуализаций из MNIST-демо (общий код)
from snks.experiments.exp_demo_mnist import (
    COLORS,
    make_config,
    train_and_collect,
    build_report,
)

# --------------------------------------------------------------------------- #
# Имена классов                                                                #
# --------------------------------------------------------------------------- #

SHAPE_NAMES = [
    "Круг", "Квадрат", "Треуголь.", "Эллипс", "Прямоуг.",
    "Пентагон", "Звезда", "Крест", "Ромб", "Стрелка",
]

# --------------------------------------------------------------------------- #
# Config для фигур (меньше узлов, быстрее)                                    #
# --------------------------------------------------------------------------- #

def make_shapes_config(device: str, num_nodes: int) -> PipelineConfig:
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
            pool_h=4,
            pool_w=8,
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
        prediction=PredictionConfig(),
        steps_per_cycle=200,
        device=device,
    )

# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def run(
    device: str = "auto",
    num_nodes: int = 20000,
    n_per_class: int = 80,
    epochs: int = 5,
    output_dir: str = "demo_output",
) -> float:
    from snks.device import get_device

    if device == "auto":
        device = str(get_device())

    print(f"[DEMO] СНКС Shapes Demo")
    print(f"[DEMO] Device: {device}  Nodes: {num_nodes:,}")
    print(f"[DEMO] {n_per_class}/class × 10 = {n_per_class * 10} images  Epochs: {epochs}")

    if not HAS_MPL:
        print("[DEMO] ОШИБКА: установите matplotlib: pip install matplotlib")
        return 0.0

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Генерация синтетических данных
    print("[DEMO] Генерация геометрических фигур...")
    generator = ShapeGenerator(image_size=64, seed=42)
    images, labels = generator.generate_all(n_variations=n_per_class)
    n_classes = int(labels.max().item()) + 1
    print(f"[DEMO] Сгенерировано {images.shape[0]} изображений, {n_classes} классов")

    # Pipeline
    config = make_shapes_config(device, num_nodes)
    pipeline = Pipeline(config)

    # Обучение
    t0 = time.time()
    results = train_and_collect(pipeline, images, labels, epochs, n_classes)
    train_time = time.time() - t0

    final_nmi = results["nmi_history"][-1]
    print(f"\n[DEMO] Обучение завершено за {train_time:.1f}s")
    print(f"[DEMO] Финальный NMI: {final_nmi:.4f}  {'PASS (>0.7)' if final_nmi > 0.7 else 'NEED MORE TRAINING'}")

    config_dict = {
        "device": device,
        "num_nodes": num_nodes,
        "n_per_class": n_per_class,
        "total_images": images.shape[0],
        "epochs": epochs,
        "oscillator": "FHN",
        "learning": "STDP",
        "data": "Synthetic geometric shapes (10 classes)",
        "final_nmi": float(final_nmi),
        "nmi_history": [float(x) for x in results["nmi_history"]],
        "train_time_sec": float(train_time),
    }

    build_report(
        results=results,
        images=images,
        n_classes=n_classes,
        class_names=SHAPE_NAMES[:n_classes],
        epochs=epochs,
        num_nodes=num_nodes,
        train_time_sec=train_time,
        config_dict=config_dict,
        mode="геометрических фигур",
        output_path=out / "shapes_report.html",
    )

    import json
    metrics = {
        "final_nmi": float(final_nmi),
        "nmi_history": [float(x) for x in results["nmi_history"]],
        "pe_history": [float(x) for x in results["pe_history"]],
        "train_time_sec": float(train_time),
    }
    (out / "shapes_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"\n[DEMO] Результаты:")
    print(f"  HTML:    {out / 'shapes_report.html'}")
    print(f"  Metrics: {out / 'shapes_metrics.json'}")

    return final_nmi


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="СНКС Shapes Visual Demo")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--nodes", type=int, default=20000)
    parser.add_argument("--per-class", type=int, default=80)
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
