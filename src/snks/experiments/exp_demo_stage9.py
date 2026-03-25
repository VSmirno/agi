"""SNKS Demo: Stage 9 - SKS-Space Prediction Enhancement.

Визуализирует четыре механизма Stage 9:
  1. SKS Embedding Quality (Exp 19) — 2D-карта embeddings по категориям
  2. HAC Prediction (Exp 20)         — точность предсказания последовательности
  3. Confidence Ratio Gate (Exp 16b) — разделение focused vs noise
  4. Broadcast Policy (Exp 18)       — усиление активации через broadcast

Генерирует HTML-отчёт: demo_output/stage9_report.html

Usage:
    python -m snks.experiments.exp_demo_stage9
"""

from __future__ import annotations

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

from sklearn.decomposition import PCA

from snks.dcam.hac import HACEngine
from snks.daf.engine import DafEngine
from snks.daf.hac_prediction import HACPredictionEngine
from snks.daf.types import DafConfig, HACPredictionConfig, MetacogConfig
from snks.gws.workspace import GWSState
from snks.metacog.monitor import MetacogMonitor
from snks.metacog.policies import BroadcastPolicy
from snks.sks.embedder import SKSEmbedder

# ------------------------------------------------------------------ #
# Style constants                                                      #
# ------------------------------------------------------------------ #
COLORS = [
    "#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6",
    "#1abc9c", "#e67e22", "#34495e", "#e91e63", "#00bcd4",
]

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: #080818;
  color: #d0d0f0;
  font-family: 'Segoe UI', 'Inter', system-ui, monospace;
  line-height: 1.7;
  min-height: 100vh;
}
.header {
  background: linear-gradient(135deg, #0f0f2e 0%, #1a1a3e 50%, #0a1628 100%);
  padding: 2.5rem 2rem 2rem;
  border-bottom: 1px solid #2a2a5a;
  text-align: center;
}
.header h1 { font-size: 2rem; color: #7eb8f7; letter-spacing: 0.05em; }
.header .subtitle { color: #8888bb; margin-top: 0.4rem; font-size: 0.95rem; }
.container { max-width: 1100px; margin: 0 auto; padding: 2rem 1.5rem; }
.section {
  background: #0e0e28;
  border: 1px solid #2a2a5a;
  border-radius: 10px;
  padding: 1.5rem;
  margin-bottom: 2rem;
}
.section h2 {
  color: #7eb8f7;
  font-size: 1.15rem;
  margin-bottom: 0.5rem;
  border-bottom: 1px solid #2a2a5a;
  padding-bottom: 0.4rem;
}
.section .desc { color: #8888bb; font-size: 0.88rem; margin-bottom: 1rem; }
.metrics-row {
  display: flex;
  gap: 1rem;
  flex-wrap: wrap;
  margin-bottom: 1rem;
}
.metric-card {
  background: #13132e;
  border: 1px solid #2a2a5a;
  border-radius: 8px;
  padding: 0.8rem 1.2rem;
  min-width: 160px;
  text-align: center;
}
.metric-card .label { font-size: 0.75rem; color: #8888bb; }
.metric-card .value { font-size: 1.6rem; font-weight: 700; margin-top: 0.2rem; }
.metric-card .threshold { font-size: 0.7rem; color: #666; margin-top: 0.1rem; }
.pass { color: #2ecc71; }
.fail { color: #e74c3c; }
.chart-wrap { text-align: center; margin-top: 0.5rem; }
.chart-wrap img { max-width: 100%; border-radius: 6px; }
.footer {
  text-align: center;
  color: #444477;
  font-size: 0.8rem;
  padding: 1.5rem;
  border-top: 1px solid #1a1a3a;
  margin-top: 1rem;
}
"""


def _fig_to_b64(fig) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor="#080818")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# ------------------------------------------------------------------ #
# Section 1: SKS Embedding Quality                                    #
# ------------------------------------------------------------------ #
N_CATEGORIES = 5
N_VARIATIONS = 8
CORE_SIZE = 80
VARIATION_SIZE = 20
N_NODES_EMB = 2000
HAC_DIM_EMB = 256


def run_embedding_quality() -> tuple[dict, str]:
    """Returns metrics dict and base64 chart."""
    embedder = SKSEmbedder(n_nodes=N_NODES_EMB, hac_dim=HAC_DIM_EMB, device="cpu")
    nodes_per_cat = N_NODES_EMB // N_CATEGORIES
    clusters, labels = [], []
    for cat in range(N_CATEGORIES):
        base = cat * nodes_per_cat
        core = set(range(base, base + CORE_SIZE))
        for var in range(N_VARIATIONS):
            torch.manual_seed(cat * 1000 + var)
            extra = set(
                torch.randint(base + CORE_SIZE,
                              min(base + nodes_per_cat, N_NODES_EMB),
                              (VARIATION_SIZE,)).tolist()
            )
            clusters.append(core | extra)
            labels.append(cat)

    embeddings = []
    for i, cl in enumerate(clusters):
        result = embedder.embed({i: cl})
        embeddings.append(result[i].numpy())

    emb_matrix = np.stack(embeddings)
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True).clip(min=1e-8)
    normed = emb_matrix / norms
    sim_matrix = normed @ normed.T

    # NMI via KNN
    from sklearn.metrics import normalized_mutual_info_score
    pred_labels = []
    for i in range(len(clusters)):
        sims = sim_matrix[i].copy()
        sims[i] = -1
        nn_idx = np.argsort(sims)[-3:]
        nn_lab = [labels[j] for j in nn_idx]
        pred_labels.append(max(set(nn_lab), key=nn_lab.count))
    nmi = normalized_mutual_info_score(labels, pred_labels)
    acc = sum(p == t for p, t in zip(pred_labels, labels)) / len(labels)

    intra, inter = [], []
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            s = float(sim_matrix[i, j])
            if labels[i] == labels[j]:
                intra.append(s)
            else:
                inter.append(s)

    # PCA 2D plot
    chart_b64 = ""
    if HAS_MPL:
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(emb_matrix)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4), facecolor="#080818")

        # Left: PCA scatter
        ax = axes[0]
        ax.set_facecolor("#0e0e28")
        for cat in range(N_CATEGORIES):
            idx = [i for i, l in enumerate(labels) if l == cat]
            ax.scatter(coords[idx, 0], coords[idx, 1],
                       color=COLORS[cat], s=50, alpha=0.85,
                       label=f"Cat {cat}", edgecolors="none")
        ax.set_title("SKS Embeddings — PCA 2D", color="#7eb8f7", fontsize=11)
        ax.tick_params(colors="#666")
        for sp in ax.spines.values():
            sp.set_color("#2a2a5a")
        ax.legend(fontsize=7, facecolor="#13132e", labelcolor="#d0d0f0",
                  edgecolor="#2a2a5a")

        # Right: similarity histogram
        ax2 = axes[1]
        ax2.set_facecolor("#0e0e28")
        ax2.hist(intra, bins=30, color=COLORS[1], alpha=0.7, label="Intra-cat")
        ax2.hist(inter, bins=30, color=COLORS[0], alpha=0.5, label="Inter-cat")
        ax2.set_title("Cosine similarity distribution", color="#7eb8f7", fontsize=11)
        ax2.tick_params(colors="#666")
        for sp in ax2.spines.values():
            sp.set_color("#2a2a5a")
        ax2.legend(fontsize=8, facecolor="#13132e", labelcolor="#d0d0f0",
                   edgecolor="#2a2a5a")

        plt.tight_layout(pad=0.8)
        chart_b64 = _fig_to_b64(fig)
        plt.close(fig)

    return {
        "nmi": float(nmi),
        "accuracy": float(acc),
        "intra_mean": float(np.mean(intra)),
        "inter_mean": float(np.mean(inter)),
    }, chart_b64


# ------------------------------------------------------------------ #
# Section 2: HAC Prediction sequence                                  #
# ------------------------------------------------------------------ #
SEQ_LENGTH = 3
N_NODES_SEQ = 1000
HAC_DIM_SEQ = 512
N_TRAIN = 50
N_TEST = 10
CORE_SEQ = 60


def run_hac_prediction() -> tuple[dict, str]:
    embedder = SKSEmbedder(n_nodes=N_NODES_SEQ, hac_dim=HAC_DIM_SEQ, device="cpu")
    nodes_per = N_NODES_SEQ // SEQ_LENGTH
    seq_embeddings = []
    for i in range(SEQ_LENGTH):
        base = i * nodes_per
        cluster = set(range(base, base + CORE_SEQ))
        result = embedder.embed({i: cluster})
        seq_embeddings.append(result[i])

    hac = HACEngine(dim=HAC_DIM_SEQ)
    cfg = HACPredictionConfig(memory_decay=0.95)
    predictor = HACPredictionEngine(hac, cfg)

    for _ in range(N_TRAIN):
        for i in range(SEQ_LENGTH):
            predictor.observe({0: seq_embeddings[i]})

    # Test
    correct = 0
    total = 0
    per_step = []
    for i in range(SEQ_LENGTH):
        predicted = predictor.predict_next({0: seq_embeddings[i]})
        if predicted is None:
            per_step.append(False)
            continue
        sims = [float(hac.similarity(predicted, e)) for e in seq_embeddings]
        top1 = int(np.argmax(sims))
        expected = (i + 1) % SEQ_LENGTH
        hit = top1 == expected
        per_step.append(hit)
        if hit:
            correct += 1
        total += 1

    # Collect PE across test repeats
    predictor2 = HACPredictionEngine(hac, cfg)
    for _ in range(N_TRAIN):
        for i in range(SEQ_LENGTH):
            predictor2.observe({0: seq_embeddings[i]})

    pe_values = []
    for rep in range(N_TEST):
        for i in range(SEQ_LENGTH):
            pred = predictor2.predict_next({0: seq_embeddings[i]})
            if pred is not None:
                actual_next = seq_embeddings[(i + 1) % SEQ_LENGTH]
                pe = predictor2.compute_winner_pe(pred, actual_next)
                pe_values.append((i, pe))

    accuracy = correct / total if total > 0 else 0.0

    chart_b64 = ""
    if HAS_MPL:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), facecolor="#080818")

        # Left: per-step accuracy bars
        ax = axes[0]
        ax.set_facecolor("#0e0e28")
        step_labels = [f"Step {i}→{(i+1)%SEQ_LENGTH}" for i in range(SEQ_LENGTH)]
        colors_bar = [COLORS[1] if h else COLORS[0] for h in per_step]
        bars = ax.bar(step_labels, [1 if h else 0 for h in per_step],
                      color=colors_bar, edgecolor="#2a2a5a", linewidth=0.5)
        ax.axhline(0.64, color="#f39c12", linestyle="--", alpha=0.7,
                   linewidth=1.2, label="Baseline 64%")
        ax.set_ylim(0, 1.2)
        ax.set_title("HAC Prediction: per-step accuracy", color="#7eb8f7", fontsize=11)
        ax.tick_params(colors="#aaa")
        for sp in ax.spines.values():
            sp.set_color("#2a2a5a")
        ax.legend(fontsize=8, facecolor="#13132e", labelcolor="#d0d0f0",
                  edgecolor="#2a2a5a")
        for bar, hit in zip(bars, per_step):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.04,
                    "✓" if hit else "✗",
                    ha="center", va="bottom", fontsize=14,
                    color=COLORS[1] if hit else COLORS[0])

        # Right: prediction error per step
        ax2 = axes[1]
        ax2.set_facecolor("#0e0e28")
        if pe_values:
            for step_i in range(SEQ_LENGTH):
                pes = [pe for (s, pe) in pe_values if s == step_i]
                if pes:
                    ax2.scatter([step_i] * len(pes), pes,
                                color=COLORS[step_i % len(COLORS)],
                                s=20, alpha=0.6,
                                label=f"Step {step_i}")
                    ax2.plot([step_i - 0.3, step_i + 0.3],
                             [np.mean(pes)] * 2,
                             color=COLORS[step_i % len(COLORS)], linewidth=2)
        ax2.set_xticks(range(SEQ_LENGTH))
        ax2.set_xticklabels([f"Step {i}" for i in range(SEQ_LENGTH)], color="#aaa")
        ax2.set_title("Prediction Error per step", color="#7eb8f7", fontsize=11)
        ax2.set_ylabel("PE", color="#aaa")
        ax2.tick_params(colors="#aaa")
        for sp in ax2.spines.values():
            sp.set_color("#2a2a5a")
        ax2.legend(fontsize=8, facecolor="#13132e", labelcolor="#d0d0f0",
                   edgecolor="#2a2a5a")

        plt.tight_layout(pad=0.8)
        chart_b64 = _fig_to_b64(fig)
        plt.close(fig)

    return {"accuracy": accuracy, "correct": correct, "total": total}, chart_b64


# ------------------------------------------------------------------ #
# Section 3: Confidence Ratio Gate                                    #
# ------------------------------------------------------------------ #
N_NODES_CONF = 500
HAC_DIM_CONF = 512
CORE_CONF = 80
N_TRAIN_CONF = 60
N_TEST_CONF = 20


def run_confidence_gate() -> tuple[dict, str]:
    from snks.daf.hac_prediction import HACPredictionEngine
    from snks.daf.types import HACPredictionConfig

    hac = HACEngine(dim=HAC_DIM_CONF)
    embedder = SKSEmbedder(n_nodes=N_NODES_CONF, hac_dim=HAC_DIM_CONF, device="cpu")

    focused_emb = embedder.embed({0: set(range(CORE_CONF))})[0]

    cfg = HACPredictionConfig(memory_decay=0.99)
    predictor = HACPredictionEngine(hac, cfg)
    for _ in range(N_TRAIN_CONF):
        predictor.observe({0: focused_emb})

    monitor = MetacogMonitor(MetacogConfig(alpha=0.0, beta=0.0, gamma=1.0))

    def get_confidence(winner_pe: float) -> float:
        class _P:
            mean_prediction_error = 0.1
        p = _P()
        p.winner_pe = winner_pe
        gws = GWSState(winner_id=0, winner_nodes=set(range(10)),
                       winner_size=10, winner_score=10.0, dominance=0.9)
        return monitor.update(gws, p).confidence

    focused_pes, noise_pes = [], []
    for i in range(N_TEST_CONF):
        torch.manual_seed(i)
        var_cluster = set(range(CORE_CONF)) | {CORE_CONF + i % 20}
        var_emb = embedder.embed({0: var_cluster})[0]
        pred = predictor.predict_next({0: var_emb})
        if pred is not None:
            focused_pes.append(predictor.compute_winner_pe(pred, var_emb))

    for i in range(N_TEST_CONF):
        torch.manual_seed(i + 10000)
        noise_emb = hac.random_vector()
        pred = predictor.predict_next({0: noise_emb})
        if pred is not None:
            noise_pes.append(predictor.compute_winner_pe(pred, noise_emb))

    focused_conf = [get_confidence(pe) for pe in focused_pes]
    noise_conf = [get_confidence(pe) for pe in noise_pes]
    mean_focused = np.mean(focused_conf) if focused_conf else 0.0
    mean_noise = np.mean(noise_conf) if noise_conf else 1.0
    ratio = mean_focused / mean_noise if mean_noise > 1e-6 else 0.0

    chart_b64 = ""
    if HAS_MPL:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), facecolor="#080818")

        # Left: PE distributions
        ax = axes[0]
        ax.set_facecolor("#0e0e28")
        ax.hist(focused_pes, bins=15, color=COLORS[1], alpha=0.75,
                label=f"Focused (μ={np.mean(focused_pes):.3f})" if focused_pes else "Focused")
        ax.hist(noise_pes, bins=15, color=COLORS[0], alpha=0.55,
                label=f"Noise (μ={np.mean(noise_pes):.3f})" if noise_pes else "Noise")
        ax.set_title("Prediction Error: focused vs noise", color="#7eb8f7", fontsize=11)
        ax.tick_params(colors="#aaa")
        for sp in ax.spines.values():
            sp.set_color("#2a2a5a")
        ax.legend(fontsize=8, facecolor="#13132e", labelcolor="#d0d0f0",
                  edgecolor="#2a2a5a")

        # Right: confidence bars
        ax2 = axes[1]
        ax2.set_facecolor("#0e0e28")
        bars = ax2.bar(["Focused", "Noise"],
                       [mean_focused, mean_noise],
                       color=[COLORS[1], COLORS[0]],
                       edgecolor="#2a2a5a", linewidth=0.5, width=0.5)
        ax2.axhline(1.0, color="#555577", linestyle=":", alpha=0.5)
        ratio_color = COLORS[1] if ratio >= 1.5 else COLORS[0]
        ax2.set_title(f"Confidence ratio: {ratio:.3f}  (≥1.5)", color="#7eb8f7", fontsize=11)
        for bar in bars:
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.01,
                     f"{bar.get_height():.3f}",
                     ha="center", va="bottom", fontsize=10,
                     color="#d0d0f0")
        ax2.set_ylim(0, max(mean_focused, mean_noise) * 1.3 + 0.1)
        ax2.tick_params(colors="#aaa")
        for sp in ax2.spines.values():
            sp.set_color("#2a2a5a")

        plt.tight_layout(pad=0.8)
        chart_b64 = _fig_to_b64(fig)
        plt.close(fig)

    return {
        "ratio": float(ratio),
        "mean_focused": float(mean_focused),
        "mean_noise": float(mean_noise),
    }, chart_b64


# ------------------------------------------------------------------ #
# Section 4: Broadcast Policy activation                             #
# ------------------------------------------------------------------ #
N_NODES_BC = 500
N_STEPS_BC = 50
N_TRIALS_BC = 10
BROADCAST_STRENGTH = 2.0
WINNER_FRACTION = 0.1


def run_broadcast() -> tuple[dict, str]:
    n_winner = int(N_NODES_BC * WINNER_FRACTION)
    winner_nodes = set(range(n_winner))

    policy = BroadcastPolicy(strength=BROADCAST_STRENGTH, threshold=0.0)
    from snks.metacog.monitor import MetacogState
    state = MetacogState(
        confidence=1.0, dominance=0.9, stability=0.8,
        pred_error=0.1, winner_nodes=winner_nodes,
    )

    cfg_daf = DafConfig()
    cfg_daf.num_nodes = N_NODES_BC
    cfg_daf.oscillator_model = "fhn"
    cfg_daf.noise_sigma = 0.05
    cfg_daf.dt = 0.01
    cfg_daf.device = "cpu"

    policy.apply(state, cfg_daf)
    broadcast_currents = policy.get_broadcast_currents(N_NODES_BC)
    assert broadcast_currents is not None

    spikes_base, spikes_bc = [], []
    for trial in range(N_TRIALS_BC):
        torch.manual_seed(trial)
        engine = DafEngine(cfg_daf, enable_learning=False)
        inp = torch.rand(N_NODES_BC) * 0.5
        engine.set_input(inp)
        engine.step(20)
        r_base = engine.step(N_STEPS_BC)
        spikes_base.append(int(r_base.fired_history.sum().item()))

        engine_bc = DafEngine(cfg_daf, enable_learning=False)
        engine_bc.states = engine.states.clone()
        engine_bc.set_input((inp + broadcast_currents).clamp(min=0))
        engine_bc.step(20)
        r_bc = engine_bc.step(N_STEPS_BC)
        spikes_bc.append(int(r_bc.fired_history.sum().item()))

    mean_base = np.mean(spikes_base)
    mean_bc = np.mean(spikes_bc)
    ratio = mean_bc / mean_base if mean_base > 0 else 0.0

    chart_b64 = ""
    if HAS_MPL:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), facecolor="#080818")

        # Left: per-trial spike counts
        ax = axes[0]
        ax.set_facecolor("#0e0e28")
        trials = list(range(N_TRIALS_BC))
        ax.plot(trials, spikes_base, "o-", color=COLORS[3],
                linewidth=1.5, markersize=5, label="Baseline")
        ax.plot(trials, spikes_bc, "s-", color=COLORS[2],
                linewidth=1.5, markersize=5, label="Broadcast")
        ax.set_title("Spikes per trial: baseline vs broadcast", color="#7eb8f7", fontsize=11)
        ax.set_xlabel("Trial", color="#aaa")
        ax.set_ylabel("Spike count", color="#aaa")
        ax.tick_params(colors="#aaa")
        for sp in ax.spines.values():
            sp.set_color("#2a2a5a")
        ax.legend(fontsize=8, facecolor="#13132e", labelcolor="#d0d0f0",
                  edgecolor="#2a2a5a")

        # Right: mean comparison bar
        ax2 = axes[1]
        ax2.set_facecolor("#0e0e28")
        bars = ax2.bar(["Baseline", "Broadcast"],
                       [mean_base, mean_bc],
                       color=[COLORS[3], COLORS[2]],
                       edgecolor="#2a2a5a", linewidth=0.5, width=0.5)
        for bar in bars:
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 100,
                     f"{bar.get_height():.0f}",
                     ha="center", va="bottom", fontsize=10, color="#d0d0f0")
        ratio_color = COLORS[2] if ratio >= 1.2 else COLORS[0]
        ax2.set_title(f"Activation ratio: {ratio:.3f}  (≥1.2)", color="#7eb8f7", fontsize=11)
        ax2.tick_params(colors="#aaa")
        for sp in ax2.spines.values():
            sp.set_color("#2a2a5a")

        plt.tight_layout(pad=0.8)
        chart_b64 = _fig_to_b64(fig)
        plt.close(fig)

    return {
        "ratio": float(ratio),
        "mean_baseline": float(mean_base),
        "mean_broadcast": float(mean_bc),
    }, chart_b64


# ------------------------------------------------------------------ #
# HTML template                                                        #
# ------------------------------------------------------------------ #
_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<title>СНКС Demo — Stage 9: SKS-Space Prediction</title>
<style>
{css}
</style>
</head>
<body>
<div class="header">
  <h1>СНКС — Stage 9: SKS-Space Prediction Enhancement</h1>
  <div class="subtitle">
    HAC-based continuous prediction in embedding space &middot;
    BroadcastPolicy &middot; Confidence Gate &middot; {ts}
  </div>
</div>
<div class="container">

<!-- Section 1 -->
<div class="section">
  <h2>1. SKS Embedding Quality (Exp 19)</h2>
  <div class="desc">
    Фиксированная случайная item memory: embedding = normalize(Σ item_vectors[nodes]).
    5 категорий × 8 вариаций, NMI по KNN-классификации.
  </div>
  <div class="metrics-row">
    <div class="metric-card">
      <div class="label">NMI</div>
      <div class="value {nmi_cls}">{nmi:.3f}</div>
      <div class="threshold">порог ≥ 0.70</div>
    </div>
    <div class="metric-card">
      <div class="label">KNN accuracy</div>
      <div class="value">{emb_acc:.1%}</div>
    </div>
    <div class="metric-card">
      <div class="label">Intra cosine</div>
      <div class="value" style="font-size:1.2rem">{intra:.3f}</div>
    </div>
    <div class="metric-card">
      <div class="label">Inter cosine</div>
      <div class="value" style="font-size:1.2rem">{inter:.3f}</div>
    </div>
  </div>
  {chart_emb}
</div>

<!-- Section 2 -->
<div class="section">
  <h2>2. HAC Prediction — Sequence A→B→C→A (Exp 20)</h2>
  <div class="desc">
    Associative memory = bundle(bind(e_t, e_{{t+1}})).
    predict_next = unbind(current, memory). Top-1 accuracy по nearest neighbor.
  </div>
  <div class="metrics-row">
    <div class="metric-card">
      <div class="label">HAC top-1 accuracy</div>
      <div class="value {hac_cls}">{hac_acc:.1%}</div>
      <div class="threshold">порог ≥ 64% (Exp3 L3)</div>
    </div>
    <div class="metric-card">
      <div class="label">Correct / Total</div>
      <div class="value" style="font-size:1.2rem">{hac_correct}/{hac_total}</div>
    </div>
  </div>
  {chart_hac}
</div>

<!-- Section 3 -->
<div class="section">
  <h2>3. Confidence Ratio Gate (Exp 16b)</h2>
  <div class="desc">
    Обучение self-loop A→A. Focused input (знакомый паттерн) → PE ≈ 0 → confidence ↑.
    Noise input → PE ≈ 0.5 → confidence ↓. Ratio = conf(focused) / conf(noise).
  </div>
  <div class="metrics-row">
    <div class="metric-card">
      <div class="label">Confidence ratio</div>
      <div class="value {conf_cls}">{conf_ratio:.3f}</div>
      <div class="threshold">порог ≥ 1.5</div>
    </div>
    <div class="metric-card">
      <div class="label">Focused confidence</div>
      <div class="value" style="font-size:1.2rem">{conf_focused:.3f}</div>
    </div>
    <div class="metric-card">
      <div class="label">Noise confidence</div>
      <div class="value" style="font-size:1.2rem">{conf_noise:.3f}</div>
    </div>
  </div>
  {chart_conf}
</div>

<!-- Section 4 -->
<div class="section">
  <h2>4. Broadcast Policy — Global Ignition (Exp 18)</h2>
  <div class="desc">
    BroadcastPolicy инжектирует ток в winner_nodes (10% узлов).
    Сравниваем spike count: baseline vs broadcast на FHN DafEngine.
  </div>
  <div class="metrics-row">
    <div class="metric-card">
      <div class="label">Activation ratio</div>
      <div class="value {bc_cls}">{bc_ratio:.3f}</div>
      <div class="threshold">порог ≥ 1.2</div>
    </div>
    <div class="metric-card">
      <div class="label">Baseline spikes</div>
      <div class="value" style="font-size:1.2rem">{bc_base:.0f}</div>
    </div>
    <div class="metric-card">
      <div class="label">Broadcast spikes</div>
      <div class="value" style="font-size:1.2rem">{bc_bc:.0f}</div>
    </div>
  </div>
  {chart_bc}
</div>

</div>
<div class="footer">
  СНКС Stage 9 &mdash; SKS-Space Prediction Enhancement &mdash; {ts}
</div>
</body>
</html>"""


def _chart_html(b64: str) -> str:
    if not b64:
        return '<p style="color:#666">matplotlib not available</p>'
    return f'<div class="chart-wrap"><img src="data:image/png;base64,{b64}" alt="chart"></div>'


def _pass_cls(val: float, threshold: float) -> str:
    return "pass" if val >= threshold else "fail"


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #
def main() -> None:
    out_dir = Path("demo_output")
    out_dir.mkdir(exist_ok=True)

    ts = time.strftime("%Y-%m-%d %H:%M")
    print(f"=== SNKS Demo: Stage 9 ({ts}) ===")

    print("  [1/4] SKS Embedding Quality...")
    m_emb, c_emb = run_embedding_quality()
    print(f"        NMI={m_emb['nmi']:.3f}, acc={m_emb['accuracy']:.1%}")

    print("  [2/4] HAC Prediction...")
    m_hac, c_hac = run_hac_prediction()
    print(f"        accuracy={m_hac['accuracy']:.1%} ({m_hac['correct']}/{m_hac['total']})")

    print("  [3/4] Confidence Ratio Gate...")
    m_conf, c_conf = run_confidence_gate()
    print(f"        ratio={m_conf['ratio']:.3f}")

    print("  [4/4] Broadcast Policy...")
    m_bc, c_bc = run_broadcast()
    print(f"        activation_ratio={m_bc['ratio']:.3f}")

    metrics = {
        "embedding": m_emb,
        "hac_prediction": m_hac,
        "confidence_gate": m_conf,
        "broadcast": m_bc,
        "timestamp": ts,
    }
    (out_dir / "stage9_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    html = _HTML_TEMPLATE.format(
        css=_CSS,
        ts=ts,
        nmi=m_emb["nmi"],
        nmi_cls=_pass_cls(m_emb["nmi"], 0.70),
        emb_acc=m_emb["accuracy"],
        intra=m_emb["intra_mean"],
        inter=m_emb["inter_mean"],
        chart_emb=_chart_html(c_emb),
        hac_acc=m_hac["accuracy"],
        hac_cls=_pass_cls(m_hac["accuracy"], 0.64),
        hac_correct=m_hac["correct"],
        hac_total=m_hac["total"],
        chart_hac=_chart_html(c_hac),
        conf_ratio=m_conf["ratio"],
        conf_cls=_pass_cls(m_conf["ratio"], 1.5),
        conf_focused=m_conf["mean_focused"],
        conf_noise=m_conf["mean_noise"],
        chart_conf=_chart_html(c_conf),
        bc_ratio=m_bc["ratio"],
        bc_cls=_pass_cls(m_bc["ratio"], 1.2),
        bc_base=m_bc["mean_baseline"],
        bc_bc=m_bc["mean_broadcast"],
        chart_bc=_chart_html(c_bc),
    )
    report_path = out_dir / "stage9_report.html"
    report_path.write_text(html, encoding="utf-8")
    print(f"\n  Report: {report_path.resolve()}")
    print("  Done.")


if __name__ == "__main__":
    main()
