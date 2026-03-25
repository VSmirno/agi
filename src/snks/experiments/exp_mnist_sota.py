"""MNIST SOTA Benchmark — СНКС vs Classical Methods.

Цель: доказать, что СНКС не хуже классических неуправляемых методов на MNIST.

Фазы (запускаются последовательно):
  0  — Classical baselines: k-means raw, PCA+k-means, Gabor+k-means
  A  — СНКС 10K nodes, raw, 500/class, 3 epochs  (warmup + timing probe)
  B  — СНКС 10K nodes, contour, 1000/class, 5 epochs
  C  — СНКС 50K nodes, contour, 1000/class, 5 epochs
  D  — СНКС 200K nodes, contour, 2000/class, 5 epochs  (if time allows)

Запуск на minipc:
  cd /opt/agi && source venv/bin/activate
  HSA_OVERRIDE_GFX_VERSION=11.0.0 PYTHONUNBUFFERED=1 \\
    python -m snks.experiments.exp_mnist_sota > mnist_sota.log 2>&1 &

Результаты: results/mnist_sota/
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler

from snks.daf.types import (
    DafConfig,
    EncoderConfig,
    PipelineConfig,
    PredictionConfig,
    SKSConfig,
)
from snks.data.mnist import MnistLoader
from snks.device import get_device
from snks.pipeline.runner import Pipeline

# ---------------------------------------------------------------------------
# SOTA reference numbers (literature, purely unsupervised on MNIST)
# ---------------------------------------------------------------------------
SOTA_REFERENCE = {
    "K-Means (raw pixels)":     {"nmi": 0.500, "train_sec": 2,    "ref": "Xie et al. 2016"},
    "PCA(50) + K-Means":        {"nmi": 0.530, "train_sec": 5,    "ref": "Guo et al. 2017"},
    "Sparse AE + K-Means":      {"nmi": 0.710, "train_sec": 300,  "ref": "Le et al. 2011"},
    "DEC (Xie 2016)":           {"nmi": 0.836, "train_sec": 1800, "ref": "Xie et al. 2016"},
    "IDEC (Guo 2017)":          {"nmi": 0.881, "train_sec": 3600, "ref": "Guo et al. 2017"},
}

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
OUT_DIR = Path("results/mnist_sota")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = OUT_DIR / "run.log"

_t_start_global = time.perf_counter()


def _log(*args, **kwargs) -> None:
    """Print to stdout and log file with timestamp."""
    elapsed = time.perf_counter() - _t_start_global
    h = int(elapsed // 3600)
    m = int((elapsed % 3600) // 60)
    s = int(elapsed % 60)
    prefix = f"[{h:02d}:{m:02d}:{s:02d}]"
    msg = " ".join(str(a) for a in args)
    line = f"{prefix} {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class ExpResult:
    name: str
    nmi: float
    train_sec: float
    n_samples: int
    n_epochs: int
    num_nodes: Optional[int]
    cycles_per_sec: Optional[float]
    device: str
    notes: str = ""


# ---------------------------------------------------------------------------
# Helper: compute NMI via k-means on feature matrix
# ---------------------------------------------------------------------------
def nmi_via_kmeans(X: np.ndarray, true_labels: np.ndarray, n_clusters: int = 10) -> float:
    km = KMeans(n_clusters=n_clusters, n_init=20, random_state=42, max_iter=500)
    pred = km.fit_predict(X)
    return float(normalized_mutual_info_score(true_labels, pred))


# ---------------------------------------------------------------------------
# Phase 0: Classical baselines
# ---------------------------------------------------------------------------
def run_baselines(images_28: np.ndarray, labels: np.ndarray) -> list[ExpResult]:
    """Run classical unsupervised baselines on 28x28 raw images."""
    results = []
    n = len(labels)
    _log(f"Phase 0: Classical baselines on {n} samples")

    # 1) K-Means on raw pixels
    X_raw = images_28.reshape(n, -1).astype(np.float32)
    t0 = time.perf_counter()
    nmi = nmi_via_kmeans(X_raw, labels)
    dt = time.perf_counter() - t0
    _log(f"  K-Means (raw 784-dim):  NMI={nmi:.4f}  time={dt:.1f}s")
    results.append(ExpResult(
        name="K-Means (raw pixels)",
        nmi=nmi, train_sec=dt, n_samples=n, n_epochs=1,
        num_nodes=None, cycles_per_sec=None, device="cpu",
    ))

    # 2) PCA(50) + K-Means
    t0 = time.perf_counter()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    pca = PCA(n_components=50, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    nmi = nmi_via_kmeans(X_pca, labels)
    dt = time.perf_counter() - t0
    _log(f"  PCA(50) + K-Means:      NMI={nmi:.4f}  time={dt:.1f}s")
    results.append(ExpResult(
        name="PCA(50) + K-Means",
        nmi=nmi, train_sec=dt, n_samples=n, n_epochs=1,
        num_nodes=None, cycles_per_sec=None, device="cpu",
    ))

    # 3) PCA(100) + K-Means
    t0 = time.perf_counter()
    pca100 = PCA(n_components=100, random_state=42)
    X_pca100 = pca100.fit_transform(X_scaled)
    nmi = nmi_via_kmeans(X_pca100, labels)
    dt = time.perf_counter() - t0
    _log(f"  PCA(100) + K-Means:     NMI={nmi:.4f}  time={dt:.1f}s")
    results.append(ExpResult(
        name="PCA(100) + K-Means",
        nmi=nmi, train_sec=dt, n_samples=n, n_epochs=1,
        num_nodes=None, cycles_per_sec=None, device="cpu",
    ))

    return results


# ---------------------------------------------------------------------------
# Phase A/B/C/D: СНКС experiments
# ---------------------------------------------------------------------------
def make_snks_config(
    device: str,
    num_nodes: int = 10_000,
    coupling_strength: float = 0.05,
    avg_degree: int = 30,
    sdr_size: int = 8192,
    steps_per_cycle: int = 200,
) -> PipelineConfig:
    return PipelineConfig(
        daf=DafConfig(
            num_nodes=num_nodes,
            avg_degree=avg_degree,
            oscillator_model="fhn",
            coupling_strength=coupling_strength,
            dt=0.01,
            noise_sigma=0.005,
            fhn_I_base=0.0,
            device=device,
        ),
        encoder=EncoderConfig(
            sdr_size=sdr_size,
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
        steps_per_cycle=steps_per_cycle,
        device=device,
    )


def run_snks_experiment(
    name: str,
    images: torch.Tensor,
    labels: torch.Tensor,
    device: str,
    num_nodes: int,
    epochs: int,
    coupling_strength: float = 0.05,
    avg_degree: int = 30,
    sdr_size: int = 8192,
    steps_per_cycle: int = 200,
    time_budget_sec: float = float("inf"),
) -> ExpResult:
    """Run one СНКС MNIST experiment. Returns ExpResult."""
    n_images = images.shape[0]
    n_classes = int(labels.max().item()) + 1
    total_cycles = n_images * epochs

    _log(f"\n{'='*60}")
    _log(f"Phase {name}: СНКС MNIST")
    _log(f"  nodes={num_nodes:,}  degree={avg_degree}  coupling={coupling_strength}")
    _log(f"  images={n_images}  epochs={epochs}  total_cycles={total_cycles:,}")
    _log(f"  sdr_size={sdr_size}  steps_per_cycle={steps_per_cycle}")
    _log(f"  device={device}")
    _log(f"{'='*60}")

    config = make_snks_config(
        device=device,
        num_nodes=num_nodes,
        coupling_strength=coupling_strength,
        avg_degree=avg_degree,
        sdr_size=sdr_size,
        steps_per_cycle=steps_per_cycle,
    )
    pipeline = Pipeline(config)

    pe_history: list[float] = []
    nmi_history: list[float] = []

    t_exp_start = time.perf_counter()
    cycle_times: list[float] = []

    # Prepare timing probe: measure first 20 cycles
    PROBE_CYCLES = 20
    probe_done = False

    # Store firing patterns AND label order from last epoch
    last_epoch_patterns = torch.zeros(n_images, num_nodes)
    last_epoch_labels = torch.zeros(n_images, dtype=torch.long)

    for epoch in range(epochs):
        perm = torch.randperm(n_images)
        epoch_pe: list[float] = []
        is_last = (epoch == epochs - 1)

        _log(f"  Epoch {epoch+1}/{epochs} ...")
        t_epoch = time.perf_counter()

        for step_i, idx in enumerate(perm):
            t_cycle = time.perf_counter()

            result = pipeline.perception_cycle(images[idx])
            pe_history.append(result.mean_prediction_error)
            epoch_pe.append(result.mean_prediction_error)

            dt_cycle = time.perf_counter() - t_cycle
            cycle_times.append(dt_cycle)

            # Timing probe after PROBE_CYCLES
            if not probe_done and len(cycle_times) >= PROBE_CYCLES:
                probe_done = True
                mean_cycle = float(np.mean(cycle_times[-PROBE_CYCLES:]))
                remaining = total_cycles - len(cycle_times)
                eta_sec = remaining * mean_cycle
                eta_h = eta_sec / 3600
                _log(f"  [PROBE] {mean_cycle*1000:.1f} ms/cycle  "
                     f"ETA: {eta_h:.1f}h  budget: {time_budget_sec/3600:.1f}h")
                if eta_sec > time_budget_sec * 1.5:
                    _log("  [WARNING] ETA significantly exceeds budget. Continuing anyway.")

            # Record firing patterns and labels on last epoch (correct label order)
            if is_last:
                fired = pipeline.engine.get_fired_history()
                if fired is not None:
                    last_epoch_patterns[step_i] = fired.float().mean(dim=0).cpu()
                last_epoch_labels[step_i] = labels[idx]

            # Progress print every 500 cycles
            global_step = epoch * n_images + step_i + 1
            if global_step % 500 == 0 or global_step == total_cycles:
                mean_c = float(np.mean(cycle_times[-100:]))
                remaining_cycles = total_cycles - global_step
                eta = remaining_cycles * mean_c
                _log(f"    step {global_step:6d}/{total_cycles}  "
                     f"pe={result.mean_prediction_error:.4f}  "
                     f"{mean_c*1000:.1f}ms/cyc  ETA={eta/60:.1f}min")

        epoch_time = time.perf_counter() - t_epoch
        mean_pe_epoch = float(np.mean(epoch_pe))
        _log(f"  Epoch {epoch+1} done: {epoch_time:.1f}s  mean_pe={mean_pe_epoch:.4f}")

    # Final NMI: k-means on firing rate vectors (labels correctly ordered via last_epoch_labels)
    _log("  Computing final NMI via k-means on firing rate vectors...")
    t_nmi = time.perf_counter()
    X_final = last_epoch_patterns.numpy()
    true_labels_final = last_epoch_labels.numpy()

    nmi = nmi_via_kmeans(X_final, true_labels_final, n_clusters=n_classes)
    dt_nmi = time.perf_counter() - t_nmi
    _log(f"  Final NMI: {nmi:.4f}  (k-means on {num_nodes:,}-dim firing vectors, {dt_nmi:.1f}s)")

    total_time = time.perf_counter() - t_exp_start
    mean_cycle_time = float(np.mean(cycle_times)) if cycle_times else 0.0
    cps = 1.0 / mean_cycle_time if mean_cycle_time > 0 else 0.0

    _log(f"  Total time: {total_time:.1f}s ({total_time/3600:.2f}h)")
    _log(f"  Throughput: {cps:.2f} cycles/sec  ({mean_cycle_time*1000:.1f} ms/cycle)")

    res = ExpResult(
        name=f"СНКС-{name}",
        nmi=nmi,
        train_sec=total_time,
        n_samples=n_images,
        n_epochs=epochs,
        num_nodes=num_nodes,
        cycles_per_sec=cps,
        device=device,
        notes=f"coupling={coupling_strength} avg_degree={avg_degree} sdr={sdr_size} steps={steps_per_cycle}",
    )
    return res


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def print_comparison_table(results: list[ExpResult]) -> None:
    _log("\n" + "=" * 72)
    _log("СРАВНИТЕЛЬНАЯ ТАБЛИЦА: СНКС vs Classical Unsupervised (MNIST)")
    _log("=" * 72)
    _log(f"{'Метод':<35} {'NMI':>6}  {'Время':>10}  {'Источник'}")
    _log("-" * 72)

    # Classical SOTA reference
    for method, info in SOTA_REFERENCE.items():
        time_str = f"{info['train_sec']:.0f}s"
        _log(f"  {method:<33} {info['nmi']:.3f}  {time_str:>10}  [{info['ref']}]")

    _log("-" * 72)
    _log("  [Наши результаты]")
    for r in results:
        time_str = f"{r.train_sec:.0f}s" if r.train_sec < 3600 else f"{r.train_sec/3600:.1f}h"
        better = " ✓" if r.nmi > 0.53 else ""  # better than PCA+kmeans
        better += " ★" if r.nmi > 0.71 else ""  # better than Sparse AE
        better += " ☆" if r.nmi > 0.84 else ""  # better than DEC!
        _log(f"  {r.name:<33} {r.nmi:.3f}  {time_str:>10}{better}")
    _log("=" * 72)


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------
def save_results(all_results: list[ExpResult], phase: str) -> None:
    out = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "phase_completed": phase,
        "sota_reference": SOTA_REFERENCE,
        "snks_results": [asdict(r) for r in all_results],
    }
    path = OUT_DIR / f"results_{phase}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    _log(f"  → Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    _log("=" * 72)
    _log("СНКС MNIST SOTA Benchmark")
    _log(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    _log("=" * 72)

    # Detect device
    dev_obj = get_device("auto")
    device = str(dev_obj)
    _log(f"Device: {device}")
    if device != "cpu":
        _log(f"GPU detected: {torch.cuda.get_device_name(0)}")
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        _log(f"VRAM: {vram_gb:.1f} GB")

    all_results: list[ExpResult] = []

    # -----------------------------------------------------------------------
    # Load MNIST data (various sizes)
    # -----------------------------------------------------------------------
    _log("\nLoading MNIST...")
    loader_raw = MnistLoader(data_root="data/", target_size=64, seed=42, preprocess="raw")
    loader_contour = MnistLoader(data_root="data/", target_size=64, seed=42, preprocess="contour")
    loader_raw28 = MnistLoader(data_root="data/", target_size=28, seed=42, preprocess="raw")

    # Baselines use 100/class = 1000 samples (quick)
    imgs_base28, lbl_base = loader_raw28.load("train", n_per_class=100)
    _log(f"Baseline dataset: {imgs_base28.shape[0]} images (28x28)")

    # СНКС experiments use larger sets (64x64)
    imgs_500_raw, lbl_500 = loader_raw.load("train", n_per_class=500)
    imgs_1000_ctr, lbl_1000 = loader_contour.load("train", n_per_class=1000)
    imgs_2000_ctr, lbl_2000 = loader_contour.load("train", n_per_class=2000)
    imgs_6000_ctr, lbl_6000 = loader_contour.load("train", n_per_class=6000)

    _log(f"Loaded: 500/class={imgs_500_raw.shape[0]}, "
         f"1000/class={imgs_1000_ctr.shape[0]}, "
         f"2000/class={imgs_2000_ctr.shape[0]}, "
         f"6000/class={imgs_6000_ctr.shape[0]}")

    # -----------------------------------------------------------------------
    # Phase 0: Classical baselines
    # -----------------------------------------------------------------------
    baseline_results = run_baselines(imgs_base28.numpy(), lbl_base.numpy())
    all_results.extend(baseline_results)
    save_results(all_results, "0-baselines")
    print_comparison_table(all_results)

    # -----------------------------------------------------------------------
    # Phase A: 10K nodes, raw, 500/class, 3 epochs  — warmup + timing probe
    # -----------------------------------------------------------------------
    _log("\n[Phase A: СНКС 10K raw — warmup + timing probe]")
    res_a = run_snks_experiment(
        name="A-10K-raw",
        images=imgs_500_raw,
        labels=lbl_500,
        device=device,
        num_nodes=10_000,
        epochs=3,
        coupling_strength=0.05,
        avg_degree=30,
        sdr_size=8192,
        steps_per_cycle=200,
    )
    all_results.append(res_a)
    save_results(all_results, "A")
    print_comparison_table(all_results)

    # -----------------------------------------------------------------------
    # Phase B: 10K nodes, contour, 1000/class, 5 epochs
    # -----------------------------------------------------------------------
    _log("\n[Phase B: СНКС 10K contour — 1000/class, 5ep]")
    res_b = run_snks_experiment(
        name="B-10K-contour",
        images=imgs_1000_ctr,
        labels=lbl_1000,
        device=device,
        num_nodes=10_000,
        epochs=5,
        coupling_strength=0.05,
        avg_degree=30,
        sdr_size=8192,
        steps_per_cycle=200,
    )
    all_results.append(res_b)
    save_results(all_results, "B")
    print_comparison_table(all_results)

    # -----------------------------------------------------------------------
    # Phase C: 50K nodes, contour, 1000/class, 5 epochs  (main experiment)
    # -----------------------------------------------------------------------
    _log("\n[Phase C: СНКС 50K contour — 1000/class, 5ep  (MAIN)]")
    res_c = run_snks_experiment(
        name="C-50K-contour",
        images=imgs_1000_ctr,
        labels=lbl_1000,
        device=device,
        num_nodes=50_000,
        epochs=5,
        coupling_strength=0.08,
        avg_degree=50,
        sdr_size=8192,
        steps_per_cycle=200,
    )
    all_results.append(res_c)
    save_results(all_results, "C")
    print_comparison_table(all_results)

    # -----------------------------------------------------------------------
    # Phase D: 200K nodes, contour, 2000/class, 5 epochs  (large)
    # -----------------------------------------------------------------------
    _log("\n[Phase D: СНКС 200K contour — 2000/class, 5ep  (LARGE)]")
    # Estimate available VRAM
    can_run_d = True
    if device != "cpu":
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        # 200K nodes needs ~25GB estimated
        can_run_d = vram_gb > 20
        if not can_run_d:
            _log(f"  [SKIP] Phase D requires ~25GB VRAM, only {vram_gb:.1f}GB available")

    if can_run_d:
        res_d = run_snks_experiment(
            name="D-200K-contour",
            images=imgs_2000_ctr,
            labels=lbl_2000,
            device=device,
            num_nodes=200_000,
            epochs=5,
            coupling_strength=0.08,
            avg_degree=50,
            sdr_size=8192,
            steps_per_cycle=200,
        )
        all_results.append(res_d)
        save_results(all_results, "D")
        print_comparison_table(all_results)

    # -----------------------------------------------------------------------
    # Phase E: 500K nodes, full MNIST, 5 epochs  (if VRAM allows)
    # -----------------------------------------------------------------------
    can_run_e = False
    if device != "cpu":
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        can_run_e = vram_gb > 55  # 500K needs ~60GB
        if not can_run_e:
            _log(f"  [SKIP] Phase E (500K) requires ~60GB VRAM, only {vram_gb:.1f}GB available")

    if can_run_e:
        _log("\n[Phase E: СНКС 500K contour — full MNIST 6000/class, 5ep  (FULL SCALE)]")
        res_e = run_snks_experiment(
            name="E-500K-full",
            images=imgs_6000_ctr,
            labels=lbl_6000,
            device=device,
            num_nodes=500_000,
            epochs=5,
            coupling_strength=0.1,
            avg_degree=50,
            sdr_size=8192,
            steps_per_cycle=200,
        )
        all_results.append(res_e)
        save_results(all_results, "E")
        print_comparison_table(all_results)

    # -----------------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------------
    _log("\n" + "=" * 72)
    _log("ФИНАЛЬНЫЙ ОТЧЁТ")
    _log(f"Завершено: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    _log(f"Полное время: {(time.perf_counter() - _t_start_global)/3600:.2f}h")
    print_comparison_table(all_results)

    # Best СНКС result
    snks_results = [r for r in all_results if "СНКС" in r.name]
    if snks_results:
        best = max(snks_results, key=lambda r: r.nmi)
        _log(f"\nЛучший результат СНКС: {best.name}  NMI={best.nmi:.4f}")
        if best.nmi > 0.836:
            _log("  ★★★ ПРЕВЫСИЛ DEC! (DEC NMI=0.836)")
        elif best.nmi > 0.710:
            _log("  ★★  Лучше Sparse AE (NMI=0.710)")
        elif best.nmi > 0.530:
            _log("  ★   Лучше PCA+K-Means (NMI=0.530)")
        else:
            _log("  ✗   Не превысил PCA+K-Means (NMI=0.530) — нужна доработка")

    save_results(all_results, "FINAL")
    _log(f"\nВсе результаты: {OUT_DIR}/")


if __name__ == "__main__":
    main()
