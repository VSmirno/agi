"""Stage 78a — Fair DAF spike test.

Answers one empirical question: **can DAF substrate learn conditional dynamics
in a regime Stage 44 audit marked as untested?**

Differences from ``experiments/spike_daf_body_predictor.py`` (prior spike):

1. **Oscillatory FHN** — I_base > 1 (Stage 44 R1.1, never tested).
   Prior spike used default I_base=0.5 which is *excitable* (single neuron
   sits at fixed point unless driven by noise).
2. **Longer integration** — 1000–10000 steps instead of 50.
   Prior spike ran only 50 steps; at dt=1e-4 that's 5 ms of model time —
   not enough for FHN to complete a single spike-recovery cycle even in
   oscillatory regime, let alone coupling propagation.
3. **SKS cluster readout via cofiring coherence** instead of raw output-zone
   voltages or ESN-style linear on all nodes. Reads the substrate at the
   level at which attractor structure lives.
4. **STDP warmup pass** over the training distribution so that clusters can
   form against the input statistics before the readout is learned.
5. **Test matrix across regimes** so the result is not a single lucky hit.

Synthetic task: conjunctive body-delta rule
    sleep + (food=0 OR drink=0) → health -0.067
A 3-category ideology textbook cannot express this without AND/OR grammar,
so it's our canonical "something learning has to discover" target.

**Ideology note — this is a DIAGNOSTIC probe only, not an agent component.**
The supervised MLP readout (Linear → ReLU → Linear trained with MSE on
body_delta labels) is used here strictly to measure whether the substrate
*carries* the conditional signal — it is NOT a proposal for how learning
should happen in a production agent. Strategy v5 rejects supervised backprop
on labeled loss as an agent learning mechanism; Stage 78b residual learner
(if it happens) will use the Dreamer-CDP stop-gradient + neg cosine pattern
on intrinsic predictive signal, not an MSE head on ground-truth body deltas.
``LinearBaseline`` uses the *same* supervised head, so the comparison is a
fair probe of "substrate contribution" not "MLP training efficacy".

Exit criterion (Stage 78a gate): **at least one regime beats the linear
no-DAF baseline on conjunctive health MSE, by at least 10% relative, with
feature discrimination > 0.1**. That answers "yes, substrate can carry
conditional information". If no regime passes, we fall back to MLP residual
(Dreamer-CDP style) in Stage 78b and drop the DAF-as-residual novelty.

Run on minipc GPU via ``scripts/minipc-run.sh``:
    ./scripts/minipc-run.sh stage78a "from stage78a_daf_spike_fair import main; main()"

Smoke test locally (fast, verifies code paths):
    .venv/bin/python experiments/stage78a_daf_spike_fair.py --smoke
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from snks.daf.engine import DafEngine
from snks.daf.types import DafConfig
from snks.sks.detection import cofiring_coherence_matrix, detect_sks
from snks.agent.motor import MotorEncoder


# ---------------------------------------------------------------------------
# Synthetic environment (conjunctive hidden rule)
# ---------------------------------------------------------------------------

BODY_VARS = ["health", "food", "drink", "energy"]
ACTIONS = [
    "sleep", "do_tree", "do_cow", "do_water",
    "move_left", "move_right", "move_up", "move_down",
]
N_ACTIONS = len(ACTIONS)
ALL_CONCEPTS = ["tree", "stone", "cow", "water", "skeleton", "zombie", "iron", "coal", "empty"]


@dataclass
class Sample:
    visible: set[str]
    body: dict[str, float]
    action: str
    delta: dict[str, float]


def true_body_delta(visible: set[str], body: dict[str, float], action: str) -> dict[str, float]:
    """SYNTHETIC ORACLE — not a real-world / Crafter rule.

    Generates ground-truth body delta for the spike test dataset. This is an
    in-experiment oracle used to supervise a diagnostic readout; it does not
    represent the mechanism by which an agent should acquire rules. Real
    agents learn via surprise + verification (Stage 79+), not via access to
    such an oracle.
    """
    delta = {v: 0.0 for v in BODY_VARS}
    delta["food"] -= 0.04
    delta["drink"] -= 0.04
    delta["energy"] -= 0.02

    if action == "sleep":
        if body.get("food", 0) <= 0 or body.get("drink", 0) <= 0:
            delta["health"] -= 0.067  # conjunctive rule — the test target
        else:
            delta["energy"] += 0.2
            delta["health"] += 0.04

    if "skeleton" in visible:
        delta["health"] -= 0.4
    if "zombie" in visible:
        delta["health"] -= 0.5

    if action == "do_cow" and "cow" in visible:
        delta["food"] += 5.0
    if action == "do_water" and "water" in visible:
        delta["drink"] += 5.0

    return delta


def random_state(rng: np.random.RandomState) -> tuple[set[str], dict[str, float]]:
    n_visible = rng.randint(0, 5)
    visible = set(rng.choice(ALL_CONCEPTS, size=n_visible, replace=False))
    body = {
        "health": float(rng.randint(1, 10)),
        "food": float(rng.randint(0, 10)),
        "drink": float(rng.randint(0, 10)),
        "energy": float(rng.randint(0, 10)),
    }
    return visible, body


def generate_dataset(n_samples: int, rng: np.random.RandomState) -> list[Sample]:
    out = []
    for _ in range(n_samples):
        visible, body = random_state(rng)
        action = ACTIONS[rng.randint(0, N_ACTIONS)]
        out.append(Sample(visible, body, action, true_body_delta(visible, body, action)))
    return out


def conjunctive_dataset(n_samples: int, rng: np.random.RandomState) -> list[Sample]:
    out = []
    for _ in range(n_samples):
        visible, body = random_state(rng)
        body[rng.choice(["food", "drink"])] = 0.0
        action = "sleep"
        out.append(Sample(visible, body, action, true_body_delta(visible, body, action)))
    return out


# ---------------------------------------------------------------------------
# Regimes — the matrix being tested
# ---------------------------------------------------------------------------


@dataclass
class Regime:
    name: str
    I_base: float
    tau: float
    sim_steps: int
    noise_sigma: float = 0.02
    coupling: float = 0.3
    enable_stdp_warmup: bool = False
    readout: str = "voltage"   # "voltage" | "spikerate" | "sks_cluster"
    description: str = ""


# Progressive test matrix — ordered from cheapest to most expensive so that
# an early pass on a cheap regime lets us make a ship/no-ship call before
# the expensive regimes finish.
def build_regimes() -> list[Regime]:
    return [
        Regime(
            name="R1_baseline_excitable_short",
            I_base=0.5, tau=12.5, sim_steps=100,
            readout="voltage",
            description="Prior spike reference — excitable, 100 steps, voltage readout.",
        ),
        Regime(
            name="R2_excitable_long",
            I_base=0.5, tau=12.5, sim_steps=2000,
            readout="voltage",
            description="Same excitable regime but 20x more steps — isolates 'was sim too short?'.",
        ),
        Regime(
            name="R3_oscillatory_default_tau",
            I_base=1.1, tau=12.5, sim_steps=10000,
            readout="voltage",
            description="Stage 44 R1.1 untested case — I_base above bifurcation, long sim.",
        ),
        Regime(
            name="R4_oscillatory_fast_tau",
            I_base=1.1, tau=1.0, sim_steps=2000,
            readout="voltage",
            description="Oscillatory with compressed recovery timescale (faster intrinsic rhythm).",
        ),
        Regime(
            name="R5_oscillatory_fast_tau_sks",
            I_base=1.1, tau=1.0, sim_steps=2000,
            readout="sks_cluster",
            description="Same as R4 but SKS cluster readout instead of raw voltage.",
        ),
        Regime(
            name="R6_oscillatory_fast_tau_stdp",
            I_base=1.1, tau=1.0, sim_steps=2000,
            enable_stdp_warmup=True,
            readout="voltage",
            description="R4 + STDP warmup pass: substrate adapts to input statistics first.",
        ),
        Regime(
            name="R7_oscillatory_fast_tau_stdp_sks",
            I_base=1.1, tau=1.0, sim_steps=2000,
            enable_stdp_warmup=True,
            readout="sks_cluster",
            description="R6 + SKS cluster readout. Most-featureful combination.",
        ),
    ]


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------


class DafPredictor:
    """DAF substrate + configurable readout for body-delta regression.

    Zones in a 5000-node network:
      - input_zone   [0, 2000)   — state SDR (visible + body buckets)
      - free hidden  [2000, 3500)
      - output_zone  [3500, 4500) — 1000 nodes, read by readout
      - motor_zone   [4500, 5000) — 500 nodes, 62 per action

    `readout` controls how features are extracted:
      - "voltage"     — raw output-zone v at end of sim
      - "spikerate"   — mean firing rate per output node across sim_steps
      - "sks_cluster" — mean firing rate per discovered cluster
    """

    def __init__(
        self,
        regime: Regime,
        n_nodes: int = 5000,
        device: str = "auto",
    ) -> None:
        self.regime = regime
        self.n_nodes = n_nodes

        # Config for this regime
        config = DafConfig(
            num_nodes=n_nodes,
            avg_degree=20,
            oscillator_model="fhn",
            disable_csr=True,
            device=device,
            noise_sigma=regime.noise_sigma,
            coupling_strength=regime.coupling,
            fhn_I_base=regime.I_base,
            fhn_tau=regime.tau,
            # Disable structural plasticity during test (edge topology stable)
            structural_interval=10**9,
        )
        self.engine = DafEngine(config, enable_learning=regime.enable_stdp_warmup)
        self.device = self.engine.device

        # Zones
        self.input_zone = (0, 2000)
        self.output_zone = (3500, 4500)
        self.motor_size_per_action = 62  # 8 * 62 = 496, fits in last 500
        self.motor_encoder = MotorEncoder(
            n_actions=N_ACTIONS,
            num_nodes=n_nodes,
            sdr_size=self.motor_size_per_action,
            current_strength=5.0,
        )

        self.output_dim = self.output_zone[1] - self.output_zone[0]
        self._concept_seeds: dict[str, list[int]] = {}

        # Cluster bookkeeping (for sks_cluster readout)
        # global-output-zone-index -> cluster-id
        self._cluster_members: list[torch.Tensor] = []
        self._feature_dim: int | None = None
        self._readout: torch.nn.Module | None = None
        self._readout_optim: torch.optim.Optimizer | None = None

    # ---- Input encoding ---------------------------------------------------

    def _concept_indices(self, concept: str, n_active: int = 30) -> torch.Tensor:
        if concept not in self._concept_seeds:
            seed = int(hashlib.md5(concept.encode()).hexdigest()[:8], 16)
            rng = np.random.RandomState(seed)
            idx = rng.choice(np.arange(0, 1000), size=n_active, replace=False).tolist()
            self._concept_seeds[concept] = idx
        return torch.tensor(self._concept_seeds[concept], dtype=torch.long, device=self.device)

    def encode_state(self, visible: set[str], body: dict[str, float]) -> torch.Tensor:
        currents = torch.zeros(self.n_nodes, device=self.device)
        for cid in visible:
            currents[self._concept_indices(cid)] = 5.0
        for i, var in enumerate(BODY_VARS):
            val = body.get(var, 0.0)
            bucket = min(9, max(0, int(val)))
            base = 1000 + i * 250
            currents[base + bucket * 25 : base + (bucket + 1) * 25] = 5.0
        return currents

    # ---- Substrate dynamics -----------------------------------------------

    def _run_sample(self, sample: Sample) -> torch.Tensor:
        """Reset state, inject (state, action), run sim_steps, return fired_history (T, N)."""
        # Fresh state each sample — no temporal leak
        self.engine.states.zero_()
        self.engine.states[:, 0] = torch.randn(self.n_nodes, device=self.device) * 0.05

        state_currents = self.encode_state(sample.visible, sample.body)
        action_idx = ACTIONS.index(sample.action)
        motor_currents = self.motor_encoder.encode(action_idx, device=self.device)
        self.engine.set_input(state_currents + motor_currents)

        result = self.engine.step(self.regime.sim_steps)
        return result.fired_history  # (T, N) bool

    def _extract_features(self, fired_history: torch.Tensor) -> torch.Tensor:
        s, e = self.output_zone
        readout = self.regime.readout
        if readout == "voltage":
            return self.engine.states[s:e, 0].detach().clone()
        elif readout == "spikerate":
            return fired_history[:, s:e].float().mean(dim=0).detach().clone()
        elif readout == "sks_cluster":
            if not self._cluster_members:
                # Fall back to spikerate if clusters not discovered yet
                return fired_history[:, s:e].float().mean(dim=0).detach().clone()
            rates = fired_history[:, s:e].float().mean(dim=0)  # (output_dim,)
            feats = torch.zeros(len(self._cluster_members) + 1, device=self.device)
            for i, members in enumerate(self._cluster_members):
                feats[i] = rates[members].mean()
            feats[-1] = rates.mean()  # overall rate as fallback feature
            return feats
        raise ValueError(f"Unknown readout: {readout}")

    # ---- Cluster discovery (for sks_cluster readout) ----------------------

    def discover_clusters(self, samples: list[Sample], max_samples: int = 64) -> int:
        """Run a subset of samples, accumulate fired_history, discover clusters.

        Called once before readout training if readout == sks_cluster.
        """
        s, e = self.output_zone
        out_n = e - s

        # Accumulate spikes across samples in output zone
        # Use a large (T_total, out_n) bool history by concatenation — on GPU this is OK
        # for ~64 samples * 2000 steps * 1000 nodes = 128M booleans = 128 MB. Safe.
        all_hist: list[torch.Tensor] = []
        n_use = min(max_samples, len(samples))
        for sample in samples[:n_use]:
            fired = self._run_sample(sample)
            all_hist.append(fired[:, s:e].detach().clone())
        combined = torch.cat(all_hist, dim=0)  # (T_total, out_n)

        # Cofiring-based clustering
        # top_k = all output-zone nodes so all of them get a chance to join a cluster
        coherence, active = cofiring_coherence_matrix(combined, top_k=out_n)
        clusters_local = detect_sks(
            coherence, method="dbscan", eps=0.3, min_samples=5, min_size=5,
        )

        # Map local (0..K-1) → global output-zone indices → indices within output_dim
        self._cluster_members = []
        for members in clusters_local:
            members_list = sorted(members)
            global_idx = active[members_list].tolist()  # already in output-zone local space
            # active indices are within [0, out_n) since we passed combined which was already sliced
            self._cluster_members.append(
                torch.tensor(global_idx, dtype=torch.long, device=self.device)
            )
        return len(self._cluster_members)

    # ---- STDP warmup -------------------------------------------------------

    def warmup_stdp(self, samples: list[Sample], n_passes: int = 1, log_every: int = 500) -> None:
        """Run samples with STDP enabled so substrate adapts to input statistics.

        After warmup, learning is disabled so the substrate is frozen for
        feature extraction. This mirrors the "fit reservoir once, then read
        out" pattern but adds STDP shaping.
        """
        assert self.engine.enable_learning, "warmup_stdp requires enable_stdp_warmup=True regime"
        start = time.time()
        for p in range(n_passes):
            for i, sample in enumerate(samples):
                self._run_sample(sample)
                if i % log_every == 0 and (p * len(samples) + i) > 0:
                    elapsed = time.time() - start
                    per_step = elapsed / (p * len(samples) + i + 1)
                    print(f"    stdp pass {p+1}/{n_passes} sample {i}/{len(samples)} "
                          f"elapsed={elapsed:.0f}s per_sample={per_step*1000:.0f}ms",
                          flush=True)
        # Freeze substrate after warmup
        self.engine.enable_learning = False

    # ---- Readout (built lazily once feature_dim is known) -----------------

    def _build_readout(self, feature_dim: int) -> None:
        self._feature_dim = feature_dim
        self._readout = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, len(BODY_VARS)),
        ).to(self.device)
        with torch.no_grad():
            for layer in self._readout:
                if isinstance(layer, torch.nn.Linear):
                    layer.weight.mul_(0.1)
                    layer.bias.zero_()
        self._readout_optim = torch.optim.Adam(self._readout.parameters(), lr=1e-3)

    def predict(self, sample: Sample) -> torch.Tensor:
        fired = self._run_sample(sample)
        feats = self._extract_features(fired)
        if self._readout is None:
            self._build_readout(feats.numel())
        with torch.no_grad():
            return self._readout(feats).detach()

    def update(self, sample: Sample) -> float:
        fired = self._run_sample(sample)
        feats = self._extract_features(fired)
        if self._readout is None:
            self._build_readout(feats.numel())
        target = torch.tensor(
            [sample.delta[v] for v in BODY_VARS], dtype=torch.float32, device=self.device,
        )
        predicted = self._readout(feats)
        loss = torch.nn.functional.mse_loss(predicted, target)
        self._readout_optim.zero_grad()
        loss.backward()
        self._readout_optim.step()
        return float(loss.item())

    # ---- Diagnostics -------------------------------------------------------

    def activity_stats(self, samples: list[Sample]) -> dict:
        """Substrate-level stats — answers 'is the regime alive?'."""
        spikes_total = 0
        feats_list = []
        for s in samples[:16]:
            fired = self._run_sample(s)
            spikes_total += int(fired.sum().item())
            feats_list.append(self._extract_features(fired))
        F = torch.stack(feats_list)  # (16, feat_dim)
        diff = F.unsqueeze(0) - F.unsqueeze(1)
        dists = diff.norm(dim=-1)
        upper = dists[torch.triu(torch.ones_like(dists), diagonal=1) > 0]
        norm_mean = float(F.norm(dim=-1).mean().item())
        return {
            "mean_spikes_per_sample": spikes_total / 16,
            "feature_norm_mean": norm_mean,
            "pairwise_dist_mean": float(upper.mean().item()),
            "pairwise_dist_min": float(upper.min().item()),
            "pairwise_dist_max": float(upper.max().item()),
            "discrimination_ratio": float(upper.mean().item()) / max(norm_mean, 1e-9),
            "feature_dim": int(F.shape[1]),
        }


# ---------------------------------------------------------------------------
# Linear (no-DAF) baseline
# ---------------------------------------------------------------------------


class LinearBaseline:
    """Same input encoding as DafPredictor, no substrate in the middle.

    Answers 'does DAF add value over the static state encoding alone?'.
    """

    def __init__(self, device: str = "auto") -> None:
        # Match DAF predictor device for fair comparison — baseline should run
        # on the same hardware as the substrate being probed.
        self.device = torch.device(device if device != "auto" else (
            "cuda" if torch.cuda.is_available() else "cpu"
        ))
        self._feature_dim = 2000 + N_ACTIONS
        self._readout = torch.nn.Sequential(
            torch.nn.Linear(self._feature_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, len(BODY_VARS)),
        ).to(self.device)
        with torch.no_grad():
            for layer in self._readout:
                if isinstance(layer, torch.nn.Linear):
                    layer.weight.mul_(0.1)
                    layer.bias.zero_()
        self._optim = torch.optim.Adam(self._readout.parameters(), lr=1e-3)
        self._concept_seeds: dict[str, list[int]] = {}

    def _concept_indices(self, concept: str, n_active: int = 30) -> torch.Tensor:
        if concept not in self._concept_seeds:
            seed = int(hashlib.md5(concept.encode()).hexdigest()[:8], 16)
            rng = np.random.RandomState(seed)
            idx = rng.choice(np.arange(0, 1000), size=n_active, replace=False).tolist()
            self._concept_seeds[concept] = idx
        return torch.tensor(self._concept_seeds[concept], dtype=torch.long, device=self.device)

    def _features(self, s: Sample) -> torch.Tensor:
        x = torch.zeros(self._feature_dim, device=self.device)
        for cid in s.visible:
            x[self._concept_indices(cid)] = 1.0
        for i, var in enumerate(BODY_VARS):
            val = s.body.get(var, 0.0)
            bucket = min(9, max(0, int(val)))
            base = 1000 + i * 250
            x[base + bucket * 25 : base + (bucket + 1) * 25] = 1.0
        x[2000 + ACTIONS.index(s.action)] = 1.0
        return x

    def predict(self, s: Sample) -> torch.Tensor:
        with torch.no_grad():
            return self._readout(self._features(s)).detach()

    def update(self, s: Sample) -> float:
        target = torch.tensor(
            [s.delta[v] for v in BODY_VARS], dtype=torch.float32, device=self.device,
        )
        pred = self._readout(self._features(s))
        loss = torch.nn.functional.mse_loss(pred, target)
        self._optim.zero_grad()
        loss.backward()
        self._optim.step()
        return float(loss.item())


# ---------------------------------------------------------------------------
# Eval / train loop
# ---------------------------------------------------------------------------


def evaluate(predictor, samples: list[Sample]) -> dict:
    sq = {v: [] for v in BODY_VARS}
    for s in samples:
        p = predictor.predict(s)
        for i, v in enumerate(BODY_VARS):
            sq[v].append((p[i].item() - s.delta[v]) ** 2)
    return {
        "overall_mse": float(np.mean([e for es in sq.values() for e in es])),
        "per_var_mse": {v: float(np.mean(es)) for v, es in sq.items()},
    }


def train_readout(predictor, samples: list[Sample], n_epochs: int, rng: np.random.RandomState) -> list[float]:
    losses_per_epoch: list[float] = []
    for epoch in range(n_epochs):
        order = rng.permutation(len(samples))
        epoch_losses = []
        t0 = time.time()
        for idx in order:
            epoch_losses.append(predictor.update(samples[int(idx)]))
        losses_per_epoch.append(float(np.mean(epoch_losses)))
        print(f"    epoch {epoch+1}/{n_epochs} loss={losses_per_epoch[-1]:.4f} "
              f"elapsed={time.time() - t0:.0f}s", flush=True)
    return losses_per_epoch


# ---------------------------------------------------------------------------
# Single-regime runner
# ---------------------------------------------------------------------------


def run_regime(
    regime: Regime,
    train: list[Sample],
    test_general: list[Sample],
    test_conj: list[Sample],
    n_epochs: int,
    rng: np.random.RandomState,
    n_nodes: int = 5000,
    device: str = "auto",
) -> dict:
    print("\n" + "=" * 72)
    print(f"REGIME: {regime.name}")
    print(f"  {regime.description}")
    print(f"  I_base={regime.I_base} tau={regime.tau} sim_steps={regime.sim_steps} "
          f"readout={regime.readout} stdp_warmup={regime.enable_stdp_warmup}")
    print("=" * 72, flush=True)

    t_start = time.time()
    predictor = DafPredictor(regime, n_nodes=n_nodes, device=device)
    print(f"  predictor ready (n_edges={predictor.engine.graph.num_edges}) "
          f"device={predictor.device}", flush=True)

    # STDP warmup (substrate shaping) if requested.
    # IDEOLOGY: warmup samples MUST be disjoint from readout training samples
    # so the substrate exposure is a "fresh stream" — matches the "self-induced
    # rules" principle that the substrate shapes to input statistics without
    # any leakage from the supervised readout split.
    if regime.enable_stdp_warmup:
        print("  -- STDP warmup pass (disjoint unlabeled stream) --", flush=True)
        warmup_rng = np.random.RandomState(99)  # different seed from train/test
        warmup_samples = generate_dataset(500, warmup_rng)
        predictor.warmup_stdp(warmup_samples, n_passes=1, log_every=100)

    # Cluster discovery if sks_cluster readout
    n_clusters = 0
    if regime.readout == "sks_cluster":
        print("  -- discovering SKS clusters from warmup batch --", flush=True)
        t0 = time.time()
        n_clusters = predictor.discover_clusters(train, max_samples=64)
        print(f"  discovered {n_clusters} clusters in {time.time()-t0:.0f}s", flush=True)

    # Activity diagnostics BEFORE training
    print("  -- activity stats --", flush=True)
    t0 = time.time()
    stats = predictor.activity_stats(train)
    print(f"    {stats} ({time.time()-t0:.0f}s)", flush=True)

    print("  -- BEFORE training --", flush=True)
    ev_gen_before = evaluate(predictor, test_general)
    ev_con_before = evaluate(predictor, test_conj)
    print(f"    general mse: {ev_gen_before['overall_mse']:.4f}", flush=True)
    print(f"    conj mse:    {ev_con_before['overall_mse']:.4f}", flush=True)

    print(f"  -- TRAINING readout ({n_epochs} epochs, {len(train)} samples) --", flush=True)
    loss_curve = train_readout(predictor, train, n_epochs, rng)

    print("  -- AFTER training --", flush=True)
    ev_gen = evaluate(predictor, test_general)
    ev_con = evaluate(predictor, test_conj)
    print(f"    general mse: {ev_gen['overall_mse']:.4f}", flush=True)
    print(f"    conj    mse: {ev_con['overall_mse']:.4f}", flush=True)
    print(f"    conj health mse: {ev_con['per_var_mse']['health']:.4f}", flush=True)

    elapsed_total = time.time() - t_start
    print(f"  REGIME DONE in {elapsed_total:.0f}s", flush=True)

    return {
        "regime": asdict(regime),
        "n_clusters": n_clusters,
        "activity": stats,
        "eval_general_before": ev_gen_before,
        "eval_conj_before": ev_con_before,
        "eval_general": ev_gen,
        "eval_conj": ev_con,
        "loss_curve": loss_curve,
        "elapsed_s": elapsed_total,
    }


def run_baseline(train: list[Sample], test_general: list[Sample], test_conj: list[Sample],
                 n_epochs: int, rng: np.random.RandomState) -> dict:
    print("\n" + "=" * 72)
    print("BASELINE: linear (no DAF) on same encoding")
    print("=" * 72, flush=True)
    b = LinearBaseline(device="auto")
    ev_gen_before = evaluate(b, test_general)
    ev_con_before = evaluate(b, test_conj)
    print(f"  before: general={ev_gen_before['overall_mse']:.4f} "
          f"conj={ev_con_before['overall_mse']:.4f}", flush=True)
    train_readout(b, train, n_epochs, rng)
    ev_gen = evaluate(b, test_general)
    ev_con = evaluate(b, test_conj)
    print(f"  after:  general={ev_gen['overall_mse']:.4f} "
          f"conj={ev_con['overall_mse']:.4f}", flush=True)
    print(f"  conj health mse: {ev_con['per_var_mse']['health']:.4f}", flush=True)
    return {
        "eval_general_before": ev_gen_before,
        "eval_conj_before": ev_con_before,
        "eval_general": ev_gen,
        "eval_conj": ev_con,
    }


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------


def verdict(results: dict) -> dict:
    """Decide PASS/FAIL for Stage 78a gate.

    Three-level result:
      STRONG_PASS — at least one regime strictly beats baseline on
                    conjunctive health MSE, with discrimination_ratio ≥ 0.10.
                    → DAF substrate adds representational capacity.
      PASS        — at least one regime is within 20% of baseline
                    (conj_health ≤ 1.2 × baseline) with discrimination_ratio ≥ 0.10.
                    → substrate carries the information (DAF-as-residual viable).
      FAIL        — no regime meets the above.
                    → MLP residual (Dreamer-CDP fallback) for Stage 78b.
    """
    if "error" in results.get("baseline", {}):
        return {"status": "FAIL", "reason": "baseline failed"}

    baseline_health = results["baseline"]["eval_conj"]["per_var_mse"]["health"]

    ranked: list[tuple[str, float, float]] = []
    passing: list[str] = []
    strong_passing: list[str] = []

    for regime_name, r in results["regimes"].items():
        if "error" in r:
            continue
        health = r["eval_conj"]["per_var_mse"]["health"]
        disc = r["activity"]["discrimination_ratio"]
        ranked.append((regime_name, health, disc))
        if disc < 0.10:
            continue
        # STRONG_PASS needs ≥10% margin to avoid noise-triggered wins.
        if health <= baseline_health * 0.90:
            strong_passing.append(regime_name)
            passing.append(regime_name)
        elif health <= baseline_health * 1.20:
            passing.append(regime_name)

    ranked.sort(key=lambda x: x[1])

    if strong_passing:
        status = "STRONG_PASS"
    elif passing:
        status = "PASS"
    else:
        status = "FAIL"

    return {
        "status": status,
        "strong_passing_regimes": strong_passing,
        "passing_regimes": passing,
        "baseline_conj_health_mse": baseline_health,
        "ranked_regimes_by_conj_health_mse": ranked,
        "recommendation": {
            "STRONG_PASS": "DAF substrate carries more signal than static encoding. "
                          "Stage 78b: DAF-as-residual in MPC loop.",
            "PASS": "DAF substrate transparent (neither helps nor destroys info). "
                    "Stage 78b: DAF-as-residual viable but test vs MLP residual head-to-head.",
            "FAIL": "DAF substrate destroys conditional info. "
                   "Stage 78b: use MLP residual (Dreamer-CDP style). "
                   "DAF-as-residual novelty dropped; retain rule induction novelty.",
        }[status],
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="tiny run to verify code on CPU — 50 train, 1 epoch, small sims")
    parser.add_argument("--n-train", type=int, default=1200)
    parser.add_argument("--n-test-general", type=int, default=300)
    parser.add_argument("--n-test-conj", type=int, default=200)
    parser.add_argument("--n-epochs", type=int, default=2)
    parser.add_argument("--n-nodes", type=int, default=5000)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--regimes", type=str, default="all",
                        help="comma-separated regime names, or 'all'")
    parser.add_argument("--output", type=str,
                        default="_docs/stage78a_results.json")
    args = parser.parse_args()

    if args.smoke:
        args.n_train = 50
        args.n_test_general = 20
        args.n_test_conj = 20
        args.n_epochs = 1
        args.n_nodes = 1000

    rng = np.random.RandomState(42)
    print("=" * 72)
    print("Stage 78a — FAIR DAF SPIKE TEST")
    print(f"  {args.n_train} train, {args.n_test_general} test_general, "
          f"{args.n_test_conj} test_conj, epochs={args.n_epochs}, "
          f"n_nodes={args.n_nodes}, device={args.device}")
    print("=" * 72, flush=True)

    print("\nGenerating datasets...", flush=True)
    train = generate_dataset(args.n_train, rng)
    test_general = generate_dataset(args.n_test_general, rng)
    test_conj = conjunctive_dataset(args.n_test_conj, rng)
    conj_in_train = sum(
        1 for s in train
        if s.action == "sleep" and (s.body.get("food", 0) <= 0 or s.body.get("drink", 0) <= 0)
    )
    print(f"  conjunctive cases in train: {conj_in_train} "
          f"({100*conj_in_train/len(train):.1f}%)", flush=True)

    # Smoke regime subset — R1 (cheap reference) + R3 (Stage 44 main untested case)
    regimes = build_regimes()
    if args.smoke:
        by_name = {r.name: r for r in regimes}
        regimes = [
            by_name["R1_baseline_excitable_short"],
            by_name["R3_oscillatory_default_tau"],
        ]
    elif args.regimes != "all":
        want = set(args.regimes.split(","))
        regimes = [r for r in regimes if r.name in want]
        if not regimes:
            print(f"ERROR: no regimes match {args.regimes}", file=sys.stderr)
            return 1

    results: dict = {"regimes": {}, "meta": {
        "n_train": args.n_train,
        "n_test_general": args.n_test_general,
        "n_test_conj": args.n_test_conj,
        "n_epochs": args.n_epochs,
        "n_nodes": args.n_nodes,
        "device_requested": args.device,
        "smoke": args.smoke,
    }}

    # Baseline first so we have a reference for each regime
    baseline_rng = np.random.RandomState(43)
    results["baseline"] = run_baseline(train, test_general, test_conj, args.n_epochs, baseline_rng)

    for regime in regimes:
        regime_rng = np.random.RandomState(44)
        try:
            r = run_regime(regime, train, test_general, test_conj,
                           args.n_epochs, regime_rng,
                           n_nodes=args.n_nodes, device=args.device)
            results["regimes"][regime.name] = r
        except Exception as e:
            import traceback
            print(f"  REGIME {regime.name} FAILED: {e}", flush=True)
            traceback.print_exc()
            results["regimes"][regime.name] = {"error": str(e), "traceback": traceback.format_exc()}

        # Save incrementally so crash doesn't lose prior runs
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"  results saved → {out_path}", flush=True)

    # Final verdict
    results["verdict"] = verdict(results)
    with Path(args.output).open("w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 72)
    print("FINAL VERDICT")
    print("=" * 72)
    v = results["verdict"]
    print(f"  status: {v['status']}")
    print(f"  baseline conj health mse: {v['baseline_conj_health_mse']:.4f}")
    print(f"  passing regimes: {v['passing_regimes']}")
    print("  regimes ranked by conj health mse (lower = better):")
    for name, health, disc in v["ranked_regimes_by_conj_health_mse"]:
        print(f"    {name:45s}  health_mse={health:.4f}  disc={disc:.3f}")
    print(f"\n  full results → {args.output}")
    # Exit code: 0 for PASS or STRONG_PASS, 2 for FAIL.
    return 0 if v["status"] in ("PASS", "STRONG_PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
