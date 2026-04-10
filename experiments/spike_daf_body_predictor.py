"""DAF spike test — BodyDeltaPredictor on synthetic Crafter-like rules.

Goal: prove (or refute) that DAF substrate + linear readout can learn
context-dependent body delta predictions from (state, action) tuples.

V2 (Echo State Network pattern):
- Reservoir (DAF coupling matrix) is FROZEN — no STDP learning during training.
- Only the linear readout is trained via supervised gradient descent.
- This is the standard reservoir computing approach: random recurrent
  network produces high-dimensional input-dependent transient features,
  and a simple linear classifier reads them out.

Why ESN instead of STDP-driven learning:
- v1 used DafCausalModel.after_action(reward=-loss). Loss exploded early
  in training, modulating STDP into degenerate fixed-point dynamics where
  the reservoir produced near-constant output regardless of input
  (input discrimination collapsed). See spike v1 results.
- ESN avoids this trap entirely by keeping the substrate fixed.

Crucial test case: CONJUNCTIVE rule that current ConceptStore textbook
cannot express:
    sleep + (food=0 OR drink=0) → health -0.067

Run locally — CPU, ~5K nodes:
    .venv/bin/python experiments/spike_daf_body_predictor.py

Reports MSE before training, after training, per-rule breakdown including
the conjunctive case, and an INPUT DISCRIMINATION check (does the reservoir
produce different output rates for different inputs?).
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

import numpy as np
import torch

from snks.daf.engine import DafEngine
from snks.daf.types import DafConfig
from snks.agent.motor import MotorEncoder


# ---------------------------------------------------------------------------
# Synthetic environment with hidden conjunctive rule
# ---------------------------------------------------------------------------

# Body variables (4) — health is what we care most about
BODY_VARS = ["health", "food", "drink", "energy"]
ACTIONS = ["sleep", "do_tree", "do_cow", "do_water", "move_left", "move_right", "move_up", "move_down"]
N_ACTIONS = len(ACTIONS)
ALL_CONCEPTS = ["tree", "stone", "cow", "water", "skeleton", "zombie", "iron", "coal", "empty"]


@dataclass
class Sample:
    visible: set[str]
    body: dict[str, float]   # current body values
    action: str               # action name
    delta: dict[str, float]   # ground-truth body delta after this action


def true_body_delta(visible: set[str], body: dict[str, float], action: str) -> dict[str, float]:
    """Ground-truth rules — what the synthetic env produces.

    Includes the CONJUNCTIVE rule that ConceptStore textbook can't express.
    """
    delta = {var: 0.0 for var in BODY_VARS}

    # Background decay (all actions, all states)
    delta["food"] -= 0.04
    delta["drink"] -= 0.04
    delta["energy"] -= 0.02

    # Sleep effects (CONJUNCTIVE — the test case)
    if action == "sleep":
        # ANY missing necessity → sleep harmful (HP degen)
        if body.get("food", 0) <= 0 or body.get("drink", 0) <= 0:
            delta["health"] -= 0.067
        else:
            # All necessities OK → energy recovery, no harm
            delta["energy"] += 0.2  # smaller than the +5 hardcoded textbook
            delta["health"] += 0.04  # tiny passive heal

    # Spatial: skeleton visible → -0.4 health
    if "skeleton" in visible:
        delta["health"] -= 0.4

    # Spatial: zombie adjacent (we proxy as "zombie" in visible) → -0.5
    if "zombie" in visible:
        delta["health"] -= 0.5

    # Eat / drink interactions
    if action == "do_cow" and "cow" in visible:
        delta["food"] += 5.0
    if action == "do_water" and "water" in visible:
        delta["drink"] += 5.0

    return delta


def random_state(rng: np.random.RandomState) -> tuple[set[str], dict[str, float]]:
    """Generate a random (visible, body) state."""
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
    """Generate N random (state, action, delta) tuples."""
    samples = []
    for _ in range(n_samples):
        visible, body = random_state(rng)
        action = ACTIONS[rng.randint(0, N_ACTIONS)]
        delta = true_body_delta(visible, body, action)
        samples.append(Sample(visible=visible, body=body, action=action, delta=delta))
    return samples


def conjunctive_dataset(n_samples: int, rng: np.random.RandomState) -> list[Sample]:
    """Force the CONJUNCTIVE case: sleep + drink=0 (or food=0).

    This is what we want the model to learn that the textbook can't express.
    """
    samples = []
    for _ in range(n_samples):
        visible, body = random_state(rng)
        # Force one necessity to be 0
        which = rng.choice(["food", "drink"])
        body[which] = 0.0
        action = "sleep"
        delta = true_body_delta(visible, body, action)
        samples.append(Sample(visible=visible, body=body, action=action, delta=delta))
    return samples


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------


class BodyDeltaPredictor:
    """DAF reservoir + linear readout for body delta prediction (ESN pattern).

    Layout (5000 nodes):
    - Input zone (0..1999):       state SDR (visible + body buckets)
    - Hidden (2000..3499):        free dynamics
    - Output zone (3500..4499):   1000 nodes, 250 per body var
    - Motor zone (4500..4999):    500 nodes (8 actions × 62 nodes)

    Echo State Network pattern: the reservoir (DAF coupling matrix) is
    initialized randomly and FROZEN. Only the linear readout is trained
    via gradient descent. This avoids the reservoir-collapse failure mode
    seen in v1 where reward-modulated STDP destroyed input discrimination.
    """

    def __init__(self, n_nodes: int = 5000, sim_steps: int = 100) -> None:
        self.n_nodes = n_nodes
        self.sim_steps = sim_steps

        config = DafConfig(
            num_nodes=n_nodes,
            avg_degree=20,
            oscillator_model="fhn",
            disable_csr=True,
            device="cpu",
            noise_sigma=0.02,  # 4× more noise → richer state diversity
            coupling_strength=0.3,  # 3× stronger coupling for input propagation
        )
        # FROZEN reservoir — no STDP, no homeostasis updates
        self.engine = DafEngine(config, enable_learning=False)

        # Zones — input/output/motor are non-overlapping
        self.input_zone = (0, 2000)
        self.output_zone = (3500, 4500)  # 1000 nodes, 250 per body var
        self.motor_size_per_action = 62
        self.motor_encoder = MotorEncoder(
            n_actions=N_ACTIONS,
            num_nodes=n_nodes,
            sdr_size=self.motor_size_per_action,
            current_strength=5.0,
        )

        # MLP readout: output zone (1000 dim) → 64 hidden ReLU → 4 body deltas
        # v4 change: nonlinear readout. If conjunctive rules need
        # nonlinear separation in feature space, linear can't see them.
        self.output_dim = self.output_zone[1] - self.output_zone[0]
        self.readout = torch.nn.Sequential(
            torch.nn.Linear(self.output_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, len(BODY_VARS)),
        )
        with torch.no_grad():
            for layer in self.readout:
                if isinstance(layer, torch.nn.Linear):
                    layer.weight.mul_(0.1)
                    layer.bias.zero_()
        self.readout_optim = torch.optim.Adam(self.readout.parameters(), lr=1e-3)

        self._concept_seeds: dict[str, list[int]] = {}

    def _concept_indices(self, concept: str, n_active: int = 30) -> torch.Tensor:
        """Hash concept name → set of input zone indices to activate."""
        if concept not in self._concept_seeds:
            seed = int(hashlib.md5(concept.encode()).hexdigest()[:8], 16)
            rng = np.random.RandomState(seed)
            indices = rng.choice(
                np.arange(self.input_zone[0], self.input_zone[0] + 1000),
                size=n_active, replace=False,
            ).tolist()
            self._concept_seeds[concept] = indices
        return torch.tensor(self._concept_seeds[concept], dtype=torch.long)

    def encode_state(self, visible: set[str], body: dict[str, float]) -> torch.Tensor:
        """state → input zone currents (1D, n_nodes)."""
        currents = torch.zeros(self.n_nodes)
        # Visible: each concept activates 30 hashed nodes in [0..999]
        for cid in visible:
            idx = self._concept_indices(cid)
            currents[idx] = 5.0  # strong drive — push v above threshold
        # Body: each var gets 250 nodes in [1000..1999], bucket position
        for i, var in enumerate(BODY_VARS):
            val = body.get(var, 0.0)
            bucket = min(9, max(0, int(val)))
            base = 1000 + i * 250
            currents[base + bucket * 25 : base + (bucket + 1) * 25] = 5.0
        return currents

    def _read_output(self, fired_history: torch.Tensor) -> torch.Tensor:
        """Read output zone — continuous voltage states (NOT spike rates).

        v1 used fired_history.mean(), but with our small sim_steps budget the
        FHN voltage never crosses threshold (=0.5) so all spike counts are 0.
        Reading raw v gives input-dependent features immediately, no spikes
        needed. This is closer to how reservoir computing literature treats
        analog reservoirs.
        """
        s, e = self.output_zone
        # Use the engine's CURRENT v state (post-step), not the time-averaged
        # spike history. This is what an analog readout would see.
        v = self.engine.states[s:e, 0].detach().clone()
        return v

    def _features(self, visible: set[str], body: dict[str, float], action: str) -> torch.Tensor:
        """Inject state + action, run frozen reservoir, return output rates.

        Resets reservoir state before each sample so trajectories don't leak
        between unrelated (state, action) pairs (we want a function not a
        sequence model in this spike).
        """
        # Reset states to fresh small noise for each sample (no temporal leak)
        self.engine.states.zero_()
        self.engine.states[:, 0] = torch.randn(self.n_nodes) * 0.05

        state_currents = self.encode_state(visible, body)
        action_idx = ACTIONS.index(action)
        motor_currents = self.motor_encoder.encode(action_idx)
        combined = state_currents + motor_currents
        self.engine.set_input(combined)
        result = self.engine.step(self.sim_steps)
        return self._read_output(result.fired_history)

    def predict(self, visible: set[str], body: dict[str, float], action: str) -> torch.Tensor:
        """No-learning forward pass — for measuring accuracy."""
        rates = self._features(visible, body, action)
        with torch.no_grad():
            return self.readout(rates).detach()

    def update(self, sample: Sample) -> float:
        """Forward + supervised readout update. Reservoir is frozen."""
        rates = self._features(sample.visible, sample.body, sample.action)
        target = torch.tensor([sample.delta[v] for v in BODY_VARS], dtype=torch.float32)
        predicted = self.readout(rates)
        loss = torch.nn.functional.mse_loss(predicted, target)
        self.readout_optim.zero_grad()
        loss.backward()
        self.readout_optim.step()
        return float(loss.item())

    def feature_distance_check(self, samples: list[Sample]) -> dict:
        """Sanity: do different inputs produce different reservoir features?

        Returns mean / max pairwise L2 distance between feature vectors.
        If this is ~0, the reservoir is degenerate and ESN won't work.
        """
        feats = [self._features(s.visible, s.body, s.action) for s in samples[:20]]
        F = torch.stack(feats)  # (20, output_dim)
        # Pairwise distances
        diff = F.unsqueeze(0) - F.unsqueeze(1)
        dists = diff.norm(dim=-1)  # (20, 20)
        upper = dists[torch.triu(torch.ones_like(dists), diagonal=1) > 0]
        mean_norm = float(F.norm(dim=-1).mean())
        return {
            "n_features": len(feats),
            "feature_norm_mean": mean_norm,
            "pairwise_dist_mean": float(upper.mean()),
            "pairwise_dist_max": float(upper.max()),
            "pairwise_dist_min": float(upper.min()),
            "discrimination_ratio": float(upper.mean()) / max(mean_norm, 1e-9),
        }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate(predictor, samples: list[Sample]) -> dict:
    """Per-var MSE + overall MSE."""
    sq_errors = {var: [] for var in BODY_VARS}
    for s in samples:
        predicted = predictor.predict(s.visible, s.body, s.action)
        for i, var in enumerate(BODY_VARS):
            err = (predicted[i].item() - s.delta[var]) ** 2
            sq_errors[var].append(err)
    return {
        "overall_mse": float(np.mean([e for errs in sq_errors.values() for e in errs])),
        "per_var_mse": {var: float(np.mean(errs)) for var, errs in sq_errors.items()},
    }


# ---------------------------------------------------------------------------
# Baseline: same encoding, NO reservoir — pure linear regression
# ---------------------------------------------------------------------------


class LinearBaseline:
    """Same input encoding as BodyDeltaPredictor, but no DAF in the middle.

    Reads the input zone currents directly as features, regresses to body
    delta with a linear layer trained on the same supervised loop. This
    answers: does the DAF substrate add value over a static encoding?
    """

    def __init__(self) -> None:
        # Same input zone (2000) + one-hot action (N_ACTIONS)
        # v4: same MLP architecture as DAF predictor — for fair comparison
        self._feature_dim = 2000 + N_ACTIONS
        self.readout = torch.nn.Sequential(
            torch.nn.Linear(self._feature_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, len(BODY_VARS)),
        )
        with torch.no_grad():
            for layer in self.readout:
                if isinstance(layer, torch.nn.Linear):
                    layer.weight.mul_(0.1)
                    layer.bias.zero_()
        self.readout_optim = torch.optim.Adam(self.readout.parameters(), lr=1e-3)
        self._concept_seeds: dict[str, list[int]] = {}

    def _concept_indices(self, concept: str, n_active: int = 30) -> torch.Tensor:
        if concept not in self._concept_seeds:
            seed = int(hashlib.md5(concept.encode()).hexdigest()[:8], 16)
            rng = np.random.RandomState(seed)
            indices = rng.choice(np.arange(0, 1000), size=n_active, replace=False).tolist()
            self._concept_seeds[concept] = indices
        return torch.tensor(self._concept_seeds[concept], dtype=torch.long)

    def _features(self, visible: set[str], body: dict[str, float], action: str) -> torch.Tensor:
        # Same encoding scheme as DAF input zone, plus one-hot action.
        x = torch.zeros(self._feature_dim)
        for cid in visible:
            idx = self._concept_indices(cid)
            x[idx] = 1.0
        for i, var in enumerate(BODY_VARS):
            val = body.get(var, 0.0)
            bucket = min(9, max(0, int(val)))
            base = 1000 + i * 250
            x[base + bucket * 25 : base + (bucket + 1) * 25] = 1.0
        # One-hot action at the tail
        action_idx = ACTIONS.index(action)
        x[2000 + action_idx] = 1.0
        return x

    def predict(self, visible: set[str], body: dict[str, float], action: str) -> torch.Tensor:
        rates = self._features(visible, body, action)
        with torch.no_grad():
            return self.readout(rates).detach()

    def update(self, sample: Sample) -> float:
        rates = self._features(sample.visible, sample.body, sample.action)
        target = torch.tensor([sample.delta[v] for v in BODY_VARS], dtype=torch.float32)
        predicted = self.readout(rates)
        loss = torch.nn.functional.mse_loss(predicted, target)
        self.readout_optim.zero_grad()
        loss.backward()
        self.readout_optim.step()
        return float(loss.item())


def main() -> int:
    print("=" * 70)
    print("DAF spike — BodyDeltaPredictor on synthetic Crafter-like rules")
    print("=" * 70)

    rng = np.random.RandomState(42)
    print("\nGenerating datasets...")
    train = generate_dataset(2000, rng)
    test_general = generate_dataset(500, rng)
    test_conjunctive = conjunctive_dataset(200, rng)
    print(f"  train: {len(train)}, test_general: {len(test_general)}, "
          f"test_conjunctive: {len(test_conjunctive)}")

    # Show what fraction of training data has conjunctive case
    conj_in_train = sum(
        1 for s in train
        if s.action == "sleep" and (s.body.get("food", 0) <= 0 or s.body.get("drink", 0) <= 0)
    )
    print(f"  conjunctive cases in train: {conj_in_train} ({100*conj_in_train/len(train):.1f}%)")

    print("\nBuilding predictor (DafEngine + linear readout, ESN pattern)...")
    t0 = time.time()
    predictor = BodyDeltaPredictor(n_nodes=5000, sim_steps=50)
    print(f"  ready in {time.time() - t0:.1f}s. n_edges={predictor.engine.graph.num_edges}")

    print("\n--- INPUT DISCRIMINATION CHECK ---")
    disc = predictor.feature_distance_check(test_general)
    print(f"  n_samples_compared: {disc['n_features']}")
    print(f"  feature_norm_mean:  {disc['feature_norm_mean']:.4f}")
    print(f"  pairwise dist mean: {disc['pairwise_dist_mean']:.4f}")
    print(f"  pairwise dist range: [{disc['pairwise_dist_min']:.4f}, "
          f"{disc['pairwise_dist_max']:.4f}]")
    print(f"  discrimination ratio (dist/norm): {disc['discrimination_ratio']:.3f}")
    if disc['discrimination_ratio'] < 0.05:
        print("  WARNING: reservoir features are nearly identical across inputs.")
        print("  ESN will fail. Need richer state encoding or different reservoir dynamics.")

    print("\n--- BEFORE training ---")
    eval_general = evaluate(predictor, test_general)
    eval_conj = evaluate(predictor, test_conjunctive)
    print(f"  general:     overall_mse={eval_general['overall_mse']:.4f}")
    print(f"               per_var={eval_general['per_var_mse']}")
    print(f"  conjunctive: overall_mse={eval_conj['overall_mse']:.4f}")
    print(f"               per_var={eval_conj['per_var_mse']}")

    n_epochs = 3
    print(f"\n--- TRAINING ({n_epochs} epochs over {len(train)} samples) ---")
    t0 = time.time()
    for epoch in range(n_epochs):
        epoch_losses = []
        # shuffle each epoch
        order = rng.permutation(len(train))
        for step, idx in enumerate(order):
            loss = predictor.update(train[int(idx)])
            epoch_losses.append(loss)
        print(f"  epoch {epoch+1}/{n_epochs} | mean_loss={np.mean(epoch_losses):.4f} "
              f"| elapsed={time.time()-t0:.0f}s")
    print(f"  training done in {time.time() - t0:.0f}s")

    print("\n--- AFTER training ---")
    eval_general_after = evaluate(predictor, test_general)
    eval_conj_after = evaluate(predictor, test_conjunctive)
    print(f"  general:     overall_mse={eval_general_after['overall_mse']:.4f}")
    print(f"               per_var={eval_general_after['per_var_mse']}")
    print(f"  conjunctive: overall_mse={eval_conj_after['overall_mse']:.4f}")
    print(f"               per_var={eval_conj_after['per_var_mse']}")

    print("\n--- VERDICT ---")
    g_before = eval_general["overall_mse"]
    g_after = eval_general_after["overall_mse"]
    c_before = eval_conj["overall_mse"]
    c_after = eval_conj_after["overall_mse"]
    print(f"  general MSE:     {g_before:.4f} → {g_after:.4f} "
          f"(reduction {100*(1-g_after/max(g_before,1e-9)):.0f}%)")
    print(f"  conjunctive MSE: {c_before:.4f} → {c_after:.4f} "
          f"(reduction {100*(1-c_after/max(c_before,1e-9)):.0f}%)")
    print(f"  health on conj:  before={eval_conj['per_var_mse']['health']:.4f}, "
          f"after={eval_conj_after['per_var_mse']['health']:.4f}")

    # Sanity: predict the conjunctive case explicitly
    print("\n--- SAMPLE PREDICTIONS DAF (conjunctive case) ---")
    for s in test_conjunctive[:3]:
        p = predictor.predict(s.visible, s.body, s.action)
        print(f"  visible={sorted(s.visible)} body=H{int(s.body['health'])}"
              f"F{int(s.body['food'])}D{int(s.body['drink'])}E{int(s.body['energy'])} "
              f"sleep")
        print(f"    truth     = {[f'{s.delta[v]:+.3f}' for v in BODY_VARS]}")
        print(f"    predicted = {[f'{p[i].item():+.3f}' for i in range(len(BODY_VARS))]}")

    # ==================================================================
    # BASELINE — same encoding, no DAF reservoir
    # ==================================================================
    print("\n" + "=" * 70)
    print("BASELINE — Linear regression on same encoding (no DAF)")
    print("=" * 70)
    baseline = LinearBaseline()

    eval_b_general_before = evaluate(baseline, test_general)
    eval_b_conj_before = evaluate(baseline, test_conjunctive)
    print(f"\n  BEFORE: general_mse={eval_b_general_before['overall_mse']:.4f}, "
          f"conj_mse={eval_b_conj_before['overall_mse']:.4f}")

    print(f"\n  Training {n_epochs} epochs over {len(train)} samples...")
    t0 = time.time()
    for epoch in range(n_epochs):
        epoch_losses = []
        order = rng.permutation(len(train))
        for idx in order:
            loss = baseline.update(train[int(idx)])
            epoch_losses.append(loss)
        print(f"    epoch {epoch+1}/{n_epochs} | mean_loss={np.mean(epoch_losses):.4f} "
              f"| elapsed={time.time()-t0:.0f}s")

    eval_b_general = evaluate(baseline, test_general)
    eval_b_conj = evaluate(baseline, test_conjunctive)
    print(f"\n  AFTER:  general_mse={eval_b_general['overall_mse']:.4f}, "
          f"conj_mse={eval_b_conj['overall_mse']:.4f}")
    print(f"          per_var_general={eval_b_general['per_var_mse']}")
    print(f"          per_var_conj={eval_b_conj['per_var_mse']}")

    print("\n--- SAMPLE PREDICTIONS BASELINE (conjunctive case) ---")
    for s in test_conjunctive[:3]:
        p = baseline.predict(s.visible, s.body, s.action)
        print(f"  visible={sorted(s.visible)} body=H{int(s.body['health'])}"
              f"F{int(s.body['food'])}D{int(s.body['drink'])}E{int(s.body['energy'])} "
              f"sleep")
        print(f"    truth     = {[f'{s.delta[v]:+.3f}' for v in BODY_VARS]}")
        print(f"    predicted = {[f'{p[i].item():+.3f}' for i in range(len(BODY_VARS))]}")

    print("\n" + "=" * 70)
    print("HEAD-TO-HEAD")
    print("=" * 70)
    print(f"  general MSE  | DAF: {eval_general_after['overall_mse']:.4f}  "
          f"| Baseline: {eval_b_general['overall_mse']:.4f}")
    print(f"  conj MSE     | DAF: {eval_conj_after['overall_mse']:.4f}  "
          f"| Baseline: {eval_b_conj['overall_mse']:.4f}")
    print(f"  health@conj  | DAF: {eval_conj_after['per_var_mse']['health']:.4f}  "
          f"| Baseline: {eval_b_conj['per_var_mse']['health']:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
