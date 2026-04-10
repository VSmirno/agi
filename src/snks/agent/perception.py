"""Stage 77a: Perception primitives — HomeostaticTracker, VisualField,
tile-based perception, verification.

Post-cleanup (Commit 8): this file holds only the live path's perception
primitives. All Stage 72-74 cosine-matching, grounding, and strategy
functions were removed as dead code.

Live consumers:
- mpc_agent.run_mpc_episode — uses HomeostaticTracker, VisualField,
  perceive_tile_field, verify_outcome
- concept_store.simulate_forward — uses HomeostaticTracker via TYPE_CHECKING
- tests/test_stage75.py — uses perceive_tile_field, VisualField
- tests/test_stage77_*.py — uses HomeostaticTracker, VisualField
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from snks.agent.concept_store import ConceptStore


# ---------------------------------------------------------------------------
# HomeostaticTracker — observe the body
# ---------------------------------------------------------------------------


@dataclass
class HomeostaticTracker:
    """Tracks body variable rates from observation.

    Stage 77a architecture (ideology-first):
    - `innate_rates`: rough directional prior loaded from textbook body block
    - `observed_rates`: running mean updated from real env observations
    - `get_rate(var)`: Bayesian combination, prior-weighted early then
      observation-weighted as experience accumulates

    No hardcoded HOMEOSTATIC_VARS — the tracker learns which variables
    matter from what the env actually exposes in inventory dicts and what
    the textbook declares innate for.
    """

    # === Stage 76: observed_max rolling max per variable ===
    observed_max: dict[str, int] = field(default_factory=dict)

    # === Stage 77a: innate + observed split with Bayesian combination ===
    innate_rates: dict[str, float] = field(default_factory=dict)
    observed_rates: dict[str, float] = field(default_factory=dict)
    observation_counts: dict[str, int] = field(default_factory=dict)
    prior_strength: int = 20

    reference_min: dict[str, float] = field(default_factory=dict)
    reference_max: dict[str, float] = field(default_factory=dict)
    vital_mins: dict[str, float] = field(default_factory=dict)

    _initialized: bool = False

    def init_from_textbook(
        self,
        body_block: dict,
        passive_rules: list[Any] | None = None,
    ) -> None:
        """Stage 77a: initialize from structured textbook body block.

        Reads `prior_strength`, `variables[*].reference_min/max/initial/vital`,
        and any `passive_body_rate` rules from `passive_rules` for innate rates.
        """
        if self._initialized:
            return

        self.prior_strength = int(body_block.get("prior_strength", 20))
        for var_def in body_block.get("variables", []):
            name = var_def["name"]
            self.reference_min[name] = float(var_def.get("reference_min", 0))
            self.reference_max[name] = float(var_def.get("reference_max", 9))
            initial = int(var_def.get("initial", 9))
            if name not in self.observed_max:
                self.observed_max[name] = initial
            if var_def.get("vital", False):
                self.vital_mins[name] = float(var_def.get("reference_min", 0))

        if passive_rules:
            for rule in passive_rules:
                if getattr(rule, "kind", None) == "passive_body_rate":
                    effect = getattr(rule, "effect", None)
                    if effect and effect.body_rate_variable:
                        self.innate_rates[effect.body_rate_variable] = effect.body_rate

        self._initialized = True

    def update(
        self,
        inv_before: dict[str, int],
        inv_after: dict[str, int],
        visible_concepts: set[str],
    ) -> None:
        """Observe real env transitions. Running mean on observed_rates
        and rolling max on observed_max. `visible_concepts` is accepted
        for API compatibility but not used in current implementation
        (conditional rates from visible concepts are a Stage 77b feature).
        """
        all_vars = set(inv_before) | set(inv_after)
        for var in all_vars:
            delta = float(inv_after.get(var, 0) - inv_before.get(var, 0))
            old_rate = self.observed_rates.get(var, 0.0)
            old_count = self.observation_counts.get(var, 0)
            new_count = old_count + 1
            self.observed_rates[var] = (old_rate * old_count + delta) / new_count
            self.observation_counts[var] = new_count

        for inv in (inv_before, inv_after):
            for var, value in inv.items():
                current_max = self.observed_max.get(var, 0)
                if value > current_max:
                    self.observed_max[var] = value

    def observed_variables(self) -> set[str]:
        """Set of body variables the tracker knows about (from observation
        or from innate textbook prior). Replaces hardcoded drive list."""
        return set(self.observed_max.keys()) | set(self.innate_rates.keys())

    def get_rate(self, variable: str) -> float:
        """Effective rate via Bayesian combination:
            w = prior_strength / (prior_strength + n)
            effective = w * innate + (1 - w) * observed
        """
        n = self.observation_counts.get(variable, 0)
        innate = self.innate_rates.get(variable, 0.0)
        observed = self.observed_rates.get(variable, 0.0)

        if n == 0 and variable not in self.innate_rates:
            return 0.0

        w = self.prior_strength / (self.prior_strength + n)
        return w * innate + (1 - w) * observed


# ---------------------------------------------------------------------------
# Visual Field
# ---------------------------------------------------------------------------


@dataclass
class VisualField:
    """What the agent sees — concepts at each grid position."""

    detections: list[tuple[str, float, int, int]] = field(default_factory=list)
    near_concept: str = "empty"
    near_similarity: float = 0.0
    center_feature: torch.Tensor | None = None
    raw_center_feature: torch.Tensor | None = None

    def visible_concepts(self) -> set[str]:
        return {cid for cid, _, _, _ in self.detections}

    def find(self, concept_id: str) -> list[tuple[float, int, int]]:
        return [
            (sim, gy, gx) for cid, sim, gy, gx in self.detections
            if cid == concept_id
        ]


# ---------------------------------------------------------------------------
# Tile-based perception (Stage 75)
# ---------------------------------------------------------------------------


def _center_positions(grid_size: int) -> set[tuple[int, int]]:
    c0 = grid_size // 2 - 1
    return {(c0, c0), (c0, c0 + 1), (c0 + 1, c0), (c0 + 1, c0 + 1)}


def perceive_tile_field(
    pixels: torch.Tensor,
    encoder: Any,
    min_confidence: float = 0.3,
) -> VisualField:
    """Perceive visual field via encoder.classify_tiles.

    Per-tile classification (no cosine matching). Returns a VisualField
    populated with detections above `min_confidence`. The center of the
    viewport determines `near_concept`.

    Args:
        pixels: (3, H, W) float tensor or compatible.
        encoder: object with `.classify_tiles(pixels)` method returning
            (class_ids, confidences) tensors. Both CNNEncoder (Stage 75)
            and TileSegmenter (Stage 77a) satisfy this interface.
        min_confidence: minimum softmax confidence to report a detection.
    """
    from snks.agent.decode_head import NEAR_CLASSES

    class_ids, confidences = encoder.classify_tiles(pixels)
    H, W = class_ids.shape
    center_pos = _center_positions(H)

    vf = VisualField()

    # Capture raw center feature when available (for grounding tests)
    try:
        with torch.no_grad():
            out = encoder(pixels.unsqueeze(0) if pixels.dim() == 3 else pixels)
            fmap = out.feature_map
            if fmap.dim() == 4:
                fmap = fmap.squeeze(0)
        c0 = H // 2 - 1
        vf.raw_center_feature = fmap[:, c0:c0 + 2, c0:c0 + 2].mean(dim=(1, 2))
        vf.center_feature = vf.raw_center_feature
    except Exception:
        # Some encoders (e.g., bare TileSegmenter) may not return feature_map
        pass

    for gy in range(H):
        for gx in range(W):
            cls_idx = int(class_ids[gy, gx].item())
            conf = float(confidences[gy, gx].item())

            if conf < min_confidence:
                continue

            if cls_idx < len(NEAR_CLASSES):
                cls_name = NEAR_CLASSES[cls_idx]
            else:
                cls_name = f"class_{cls_idx}"

            if cls_name == "empty" and (gy, gx) not in center_pos:
                continue

            vf.detections.append((cls_name, conf, gy, gx))

            if (gy, gx) in center_pos and conf > vf.near_similarity:
                vf.near_concept = cls_name
                vf.near_similarity = conf

    return vf


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def verify_outcome(
    near_label: str | None,
    action: str,
    actual_outcome: str | None,
    concept_store: "ConceptStore",
) -> None:
    """Update rule confidence based on the observed outcome of an action."""
    if near_label is None or actual_outcome is None:
        return
    concept_store.verify(near_label, action, actual_outcome)


def outcome_to_verify(
    action: str,
    inv_before: dict[str, int],
    inv_after: dict[str, int],
) -> str | None:
    """Compute a label for the verification step. Used by mpc_agent to
    update ConceptStore rule confidence after each action."""
    gains, losses = {}, {}
    for k in set(inv_before) | set(inv_after):
        d = inv_after.get(k, 0) - inv_before.get(k, 0)
        if d > 0:
            gains[k] = d
        elif d < 0:
            losses[k] = -d

    body_vars = {"health", "food", "drink", "energy"}

    if action == "do":
        for k in gains:
            if k not in body_vars:
                return k
        for stat in ("food", "drink"):
            if gains.get(stat, 0) > 0:
                return f"restore_{stat}"
        return None
    if action.startswith("place_"):
        return action.replace("place_", "") if losses else None
    if action.startswith("make_"):
        crafted = action.replace("make_", "")
        return crafted if gains.get(crafted, 0) > 0 else None
    return None
