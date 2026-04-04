# Stage 65: Calibrated Uncertainty

## Summary

Replace binary/uncalibrated confidence with calibrated probabilities [0, 1].
Track prediction accuracy per confidence bucket. Brier score as quality metric.

## Gate

- Brier score < 0.15 on held-out transitions
- Confidence ~ accuracy correlation ρ > 0.7 (Spearman)
- Crafter QA ≥80% (regression)
- MiniGrid QA ≥90% (regression)

## Confidence Calibration

### Per-source mapping

**Neocortex** (verified rules):
```python
conf = min(n_confirmations / (n_confirmations + 2), 0.95)
# 1 confirmation → 0.33
# 3 confirmations → 0.60
# 10 confirmations → 0.83
# Cap at 0.95 — never 100% certain
```

**SDM (hippocampus)** (fuzzy generalization):
```python
conf = sigmoid((magnitude - mu) / sigma)
# mu, sigma learned from calibration data
# Raw magnitude varies by SDM fill level
# Sigmoid maps to [0, 1]
```

**AbstractionEngine** (category-based):
```python
# Known member: 0.85 (lower than neocortex — category match, not exact)
# Unknown member via SDM: sdm_conf * 0.7
```

### CalibrationTracker

Collects (predicted_confidence, was_correct) pairs during exploration,
then computes calibration metrics.

```python
class CalibrationTracker:
    def __init__(self, n_buckets: int = 5):
        self.predictions: list[tuple[float, bool]] = []
        self.n_buckets = n_buckets

    def record(self, confidence: float, predicted: str, actual: str):
        correct = predicted == actual
        self.predictions.append((confidence, correct))

    def brier_score(self) -> float:
        """Mean squared error between confidence and correctness."""
        if not self.predictions:
            return 1.0
        return sum((c - int(ok)) ** 2
                   for c, ok in self.predictions) / len(self.predictions)

    def calibration_curve(self) -> list[tuple[float, float, int]]:
        """Returns (mean_conf, accuracy, count) per bucket."""
        buckets = [[] for _ in range(self.n_buckets)]
        for conf, correct in self.predictions:
            idx = min(int(conf * self.n_buckets), self.n_buckets - 1)
            buckets[idx].append((conf, correct))

        curve = []
        for bucket in buckets:
            if bucket:
                mean_conf = sum(c for c, _ in bucket) / len(bucket)
                accuracy = sum(int(ok) for _, ok in bucket) / len(bucket)
                curve.append((mean_conf, accuracy, len(bucket)))
        return curve

    def correlation(self) -> float:
        """Spearman correlation between confidence and correctness."""
        if len(self.predictions) < 5:
            return 0.0
        from scipy.stats import spearmanr
        confs = [c for c, _ in self.predictions]
        correct = [int(ok) for _, ok in self.predictions]
        rho, _ = spearmanr(confs, correct)
        return rho if rho == rho else 0.0  # NaN guard
```

## Changes to CLSWorldModel

### query() — calibrated confidence

```python
def query(self, situation, action):
    key = make_situation_key(situation, action)

    # Neocortex (exact match)
    if key in self.neocortex:
        rule = self.neocortex[key]
        conf = min(rule.confidence / (rule.confidence + 2), 0.95)
        return rule.outcome, conf, "neocortex"

    # Abstract generalization
    outcome_str, abs_conf = self.abstraction.query_abstract(...)
    if outcome_str != "unknown" and abs_conf > 0.01:
        # Scale: known member → 0.85, SDM → sdm_conf * 0.7
        return {"result": outcome_str}, abs_conf, "abstract"

    # Hippocampus (SDM)
    predicted, raw_conf = self.hippocampus.read_next(sit_vec, zeros)
    if raw_conf > 0.01:
        calibrated = self._calibrate_sdm(raw_conf)
        outcome = self._decode_outcome(predicted)
        return outcome, calibrated, "hippocampus"

    return {"result": "unknown"}, 0.0, "none"
```

### _calibrate_sdm() — sigmoid mapping

```python
def _calibrate_sdm(self, raw_magnitude: float) -> float:
    """Map raw SDM magnitude to calibrated probability."""
    import math
    # Default params, can be tuned via calibration
    mu = self._sdm_cal_mu      # default 0.3
    sigma = self._sdm_cal_sigma  # default 0.2
    x = (raw_magnitude - mu) / max(sigma, 0.01)
    return 1.0 / (1.0 + math.exp(-x))
```

### SDM calibration fitting

After exploration, fit mu/sigma from actual data:
```python
def fit_sdm_calibration(self, tracker: CalibrationTracker):
    """Adjust SDM sigmoid params to minimize Brier score."""
    # Simple grid search over mu, sigma
    # Pick params that minimize Brier score on SDM-sourced predictions
```

## Changes to AbstractionEngine

```python
def query_abstract(self, obj_type, action, obj_state, carrying):
    ...
    # Known member: return 0.85 (not 1.0)
    if obj_type in cat.members:
        return cat.outcome, 0.85

    # Unknown via SDM: scale down
    if conf > 0.01:
        return outcome, conf * 0.7
```

## Exploration with CalibrationTracker

CuriosityExplorer records predictions during exploration:
```python
def explore_episode(self, env, tracker, max_steps):
    for step in range(max_steps):
        situation = env.observe()
        action = self.select_action(situation, env.available_actions())

        # Record prediction BEFORE seeing outcome
        predicted, conf, source = self.wm.query(situation, action)

        outcome, reward = env.step(action)

        # Record for calibration
        tracker.record(conf, predicted.get("result"), outcome.get("result"))

        # Train
        self.wm.train([Transition(situation, action, outcome, reward)])
```

## Files

| File | Change |
|------|--------|
| `src/snks/agent/cls_world_model.py` | Calibrated confidence in query(), _calibrate_sdm() |
| `src/snks/agent/abstraction_engine.py` | 0.85 for known, sdm*0.7 for unknown |
| `src/snks/agent/calibration.py` | NEW: CalibrationTracker |
| `tests/test_stage65_uncertainty.py` | Brier score, calibration curve, correlation |
| `experiments/exp121_uncertainty.py` | Gate experiment |

## What does NOT change

- CuriosityExplorer thresholds — already uses [0, 1] confidence
- Neocortex/SDM storage — same data, different read interpretation
- Training pipeline — unchanged
