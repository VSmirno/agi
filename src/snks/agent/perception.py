"""Stage 74: Homeostatic Agent — goal-free self-organizing behavior.

The agent has NO explicit goal. Its "goal" is its body — homeostatic
variables that must stay in range. All behavior emerges from:
  body (DNA) + world model (textbook + experience) + curiosity (drive to know)

Drives are NOT hardcoded. They are OBSERVED:
  - Rate of change of homeostatic variables → urgency
  - Conditional rates (what's visible when variable drops) → cause identification
  - Curiosity (model incompleteness) → exploration when body is fine

Strategy emerges from urgency + world model planning:
  health dropping fast + zombie visible → world model: "zombie causes this" →
  "sword kills zombie" → plan: craft sword. No programmer told the agent this.

Design: docs/superpowers/specs/2026-04-07-stage74-homeostatic-agent-design.md
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from snks.agent.concept_store import Concept, ConceptStore, PlannedStep
    from snks.agent.outcome_labeler import OutcomeLabeler


MIN_SIMILARITY = 0.4
EMA_ALPHA = 0.2       # prototype refinement (faster convergence for 256-dim)
RATE_EMA_ALPHA = 0.05  # homeostatic rate tracking (slow — body changes gradually)

# The body — the ONLY hardcoded thing (the "DNA")
HOMEOSTATIC_VARS = {"health", "food", "drink", "energy"}

# Grid positions: center 2×2 = "nearby"
CENTER_POSITIONS = [(1, 1), (1, 2), (2, 1), (2, 2)]


# ---------------------------------------------------------------------------
# HomeostaticTracker — observe the body
# ---------------------------------------------------------------------------


@dataclass
class HomeostaticTracker:
    """Tracks rate of change of body variables and what causes changes.

    This is the agent's interoception — sensing its own body.
    No formulas, just observation: "when zombie visible, health drops fast."

    Persists across episodes (agent remembers what hurts).
    """

    # Background rates: average delta per step for each variable
    rates: dict[str, float] = field(default_factory=lambda: {
        v: 0.0 for v in HOMEOSTATIC_VARS
    })

    # Conditional rates: (concept_id, variable) → average delta
    # "when concept X is visible, variable Y changes by this much"
    conditional_rates: dict[tuple[str, str], float] = field(default_factory=dict)

    # Initial rates from body rules (innate knowledge, set once from textbook)
    _initialized: bool = False

    def init_from_body_rules(self, body_rules: list[dict]) -> None:
        """Set initial rates from textbook body rules (innate knowledge).

        Like a baby knowing pain = bad. Not learned, pre-wired.
        _background rules set unconditional baseline rates.
        """
        if self._initialized:
            return
        for rule in body_rules:
            concept = rule.get("concept")
            var = rule.get("variable")
            rate = rule.get("rate", 0.0)
            if concept and var:
                if concept == "_background":
                    # Background rate = unconditional baseline
                    self.rates[var] = rate
                else:
                    self.conditional_rates[(concept, var)] = rate
        self._initialized = True

    def update(
        self,
        inv_before: dict[str, int],
        inv_after: dict[str, int],
        visible_concepts: set[str],
    ) -> None:
        """Called every step: observe what changed and what was visible."""
        for var in HOMEOSTATIC_VARS:
            delta = inv_after.get(var, 0) - inv_before.get(var, 0)

            # Update background rate (unconditional)
            old = self.rates.get(var, 0.0)
            self.rates[var] = old * (1 - RATE_EMA_ALPHA) + delta * RATE_EMA_ALPHA

            # Update conditional rates (what's nearby affects what?)
            for concept in visible_concepts:
                key = (concept, var)
                old_c = self.conditional_rates.get(key, 0.0)
                self.conditional_rates[key] = (
                    old_c * (1 - RATE_EMA_ALPHA) + delta * RATE_EMA_ALPHA
                )

    def get_rate(self, variable: str, visible_concepts: set[str] | None = None) -> float:
        """Get effective rate for a variable given what's visible.

        Uses conditional rates if available, falls back to background rate.
        """
        if visible_concepts:
            # Use the WORST (most negative) conditional rate among visible concepts
            worst = self.rates.get(variable, 0.0)
            for concept in visible_concepts:
                key = (concept, variable)
                cr = self.conditional_rates.get(key, 0.0)
                if cr < worst:
                    worst = cr
            return worst
        return self.rates.get(variable, 0.0)


# ---------------------------------------------------------------------------
# Visual Field (from Stage 73)
# ---------------------------------------------------------------------------


@dataclass
class VisualField:
    """What the agent sees — concepts at each grid position."""

    detections: list[tuple[str, float, int, int]] = field(default_factory=list)
    near_concept: str = "empty"
    near_similarity: float = 0.0
    center_feature: torch.Tensor | None = None

    def visible_concepts(self) -> set[str]:
        return {cid for cid, _, _, _ in self.detections}

    def find(self, concept_id: str) -> list[tuple[float, int, int]]:
        return [
            (sim, gy, gx) for cid, sim, gy, gx in self.detections
            if cid == concept_id
        ]


# ---------------------------------------------------------------------------
# Perception
# ---------------------------------------------------------------------------


def perceive_field(
    pixels: torch.Tensor,
    encoder: Any,
    concept_store: ConceptStore,
    min_similarity: float = MIN_SIMILARITY,
) -> VisualField:
    """Perceive visual field: match 256-dim prototypes at each grid position."""
    with torch.no_grad():
        out = encoder(pixels.unsqueeze(0) if pixels.dim() == 3 else pixels)
        fmap = out.feature_map
        if fmap.dim() == 4:
            fmap = fmap.squeeze(0)

    vf = VisualField()
    vf.center_feature = fmap[:, 1:3, 1:3].mean(dim=(1, 2))

    for gy in range(4):
        for gx in range(4):
            feat = fmap[:, gy, gx]
            feat_norm = F.normalize(feat.unsqueeze(0), dim=1)

            best_concept = None
            best_sim = -1.0
            for concept in concept_store.concepts.values():
                if concept.visual is None:
                    continue
                c_norm = F.normalize(concept.visual.unsqueeze(0), dim=1)
                sim = (feat_norm @ c_norm.T).item()
                if sim > best_sim:
                    best_sim = sim
                    best_concept = concept

            if best_concept is not None and best_sim >= min_similarity:
                vf.detections.append((best_concept.id, best_sim, gy, gx))
                if (gy, gx) in CENTER_POSITIONS and best_sim > vf.near_similarity:
                    vf.near_concept = best_concept.id
                    vf.near_similarity = best_sim

    return vf


def perceive(
    pixels: torch.Tensor,
    encoder: Any,
    concept_store: ConceptStore,
    min_similarity: float = MIN_SIMILARITY,
) -> tuple[Any, torch.Tensor]:
    """Legacy perceive — returns (concept, center_feature)."""
    vf = perceive_field(pixels, encoder, concept_store, min_similarity)
    concept = concept_store.query_text(vf.near_concept) if vf.near_similarity > 0 else None
    cf = vf.center_feature if vf.center_feature is not None else torch.zeros(256)
    return concept, cf


# ---------------------------------------------------------------------------
# Bootstrap grounding
# ---------------------------------------------------------------------------


def ground_empty_on_start(
    pixels: torch.Tensor, encoder: Any, concept_store: ConceptStore,
) -> bool:
    concept = concept_store.query_text("empty")
    if concept is None or concept.visual is not None:
        return False
    with torch.no_grad():
        out = encoder(pixels.unsqueeze(0) if pixels.dim() == 3 else pixels)
        fmap = out.feature_map
        if fmap.dim() == 4:
            fmap = fmap.squeeze(0)
        center_feat = fmap[:, 1:3, 1:3].mean(dim=(1, 2))
        z_norm = F.normalize(center_feat.unsqueeze(0), dim=1).squeeze(0)
        concept_store.ground_visual("empty", z_norm)
    return True


def ground_zombie_on_damage(
    inv_before: dict[str, int], inv_after: dict[str, int],
    visual_field: VisualField, concept_store: ConceptStore,
) -> bool:
    health_before = inv_before.get("health", 9)
    health_after = inv_after.get("health", 9)
    if health_after >= health_before:
        return False
    if inv_after.get("food", 0) == 0 or inv_after.get("drink", 0) == 0:
        return False
    concept = concept_store.query_text("zombie")
    if concept is None or visual_field.center_feature is None:
        return False
    z_norm = F.normalize(visual_field.center_feature.unsqueeze(0), dim=1).squeeze(0)
    if concept.visual is None:
        concept_store.ground_visual("zombie", z_norm)
    else:
        concept.visual = F.normalize(
            ((1 - EMA_ALPHA) * concept.visual + EMA_ALPHA * z_norm).unsqueeze(0),
            dim=1,
        ).squeeze(0)
    return True


# ---------------------------------------------------------------------------
# Experiential grounding
# ---------------------------------------------------------------------------

_STAT_GAIN_TO_NEAR: dict[str, str] = {"food": "cow", "drink": "water"}


def on_action_outcome(
    action: str, inv_before: dict[str, int], inv_after: dict[str, int],
    center_feature: torch.Tensor, concept_store: ConceptStore,
    labeler: Any,
) -> str | None:
    label = labeler.label(action, inv_before, inv_after)
    if label is None and action == "do":
        for stat, near in _STAT_GAIN_TO_NEAR.items():
            if inv_after.get(stat, 0) > inv_before.get(stat, 0):
                label = near
                break
    if label is None:
        return None
    concept = concept_store.query_text(label)
    if concept is None:
        return None
    z_norm = F.normalize(center_feature.unsqueeze(0), dim=1).squeeze(0)
    if concept.visual is None:
        concept_store.ground_visual(label, z_norm)
    else:
        concept.visual = F.normalize(
            ((1 - EMA_ALPHA) * concept.visual + EMA_ALPHA * z_norm).unsqueeze(0),
            dim=1,
        ).squeeze(0)
    return label


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def verify_outcome(
    near_label: str | None, action: str, actual_outcome: str | None,
    concept_store: ConceptStore,
) -> None:
    if near_label is None or actual_outcome is None:
        return
    concept_store.verify(near_label, action, actual_outcome)


def outcome_to_verify(
    action: str, inv_before: dict[str, int], inv_after: dict[str, int],
) -> str | None:
    survival = {"health", "food", "drink", "energy"}
    gains, losses = {}, {}
    for k in set(inv_before) | set(inv_after):
        d = inv_after.get(k, 0) - inv_before.get(k, 0)
        if d > 0:
            gains[k] = d
        elif d < 0:
            losses[k] = -d
    if action == "do":
        for k in gains:
            if k not in survival:
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


# ---------------------------------------------------------------------------
# Curiosity / Motor Babbling
# ---------------------------------------------------------------------------

BABBLE_BASE_PROB = 0.15
BABBLE_MIN_PROB = 0.03
_DIRECTIONS = ["move_up", "move_down", "move_left", "move_right"]


def babble_probability(concept_store: ConceptStore) -> float:
    n_grounded = sum(1 for c in concept_store.concepts.values() if c.visual is not None)
    prob = BABBLE_BASE_PROB / (1 + n_grounded * 0.3)
    return max(BABBLE_MIN_PROB, prob)


def compute_curiosity(concept_store: ConceptStore, spatial_map: Any) -> float:
    """How incomplete is the agent's world model? Biological curiosity.

    Curiosity is a BACKGROUND drive — it only dominates when body is fine.
    Its raw value is moderate (0.0-0.3 range) so that even mild body
    urgency overrides it. A hungry animal doesn't explore.
    """
    total = len(concept_store.concepts)
    grounded = sum(1 for c in concept_store.concepts.values() if c.visual is not None)
    visual_gap = 1.0 - (grounded / max(1, total))

    confidences = [
        link.confidence
        for c in concept_store.concepts.values()
        for link in c.causal_links
    ]
    mean_conf = sum(confidences) / max(1, len(confidences))
    knowledge_gap = 1.0 - mean_conf

    visited = spatial_map.n_visited if spatial_map else 0
    map_gap = 1.0 / max(1, visited / 50)

    # Scale down to background level — body drives should override easily
    # Curiosity maxes at ~0.03, body drives start at ~0.04 for mildly low stats
    return (visual_gap + knowledge_gap + map_gap) / 3.0 * 0.05


def explore_action(
    rng: np.random.RandomState,
    concept_store: ConceptStore,
    inventory: dict[str, int] | None = None,
) -> str:
    p = babble_probability(concept_store)
    if rng.random() < p:
        if inventory is not None and inventory.get("wood", 0) >= 2:
            if rng.random() < 0.3:
                return "babble_place_table"
        if inventory is not None and inventory.get("wood", 0) >= 1:
            if rng.random() < 0.15:
                return "babble_make_wood_pickaxe"
        return "babble_do"
    return str(rng.choice(_DIRECTIONS))


# ---------------------------------------------------------------------------
# Drive computation — NO hardcoded weights
# ---------------------------------------------------------------------------


def compute_drive(variable: str, current_value: float, rate: float) -> float:
    """Urgency = inverse of time until variable reaches zero.

    No magic numbers. Pure physics of the body.
    """
    if rate >= 0:
        return 0.0  # stable or rising
    steps_until_zero = current_value / abs(rate)
    return 1.0 / max(1.0, steps_until_zero)


def select_goal(
    inventory: dict[str, int],
    concept_store: ConceptStore,
    tracker: HomeostaticTracker | None = None,
    visual_field: VisualField | None = None,
    spatial_map: Any = None,
) -> tuple[str, list]:
    """Goal selection from body urgency + curiosity. Zero hardcoded strategy.

    1. Compute urgency for each homeostatic variable from observed rates
    2. Add curiosity drive (model incompleteness)
    3. Most urgent wins
    4. Plan from world model (Strategy 1: restore, Strategy 2: remove cause)
    """
    visible = visual_field.visible_concepts() if visual_field else set()

    # Compute urgency for each body variable
    urgencies: dict[str, float] = {}
    for var in HOMEOSTATIC_VARS:
        value = inventory.get(var, 9)
        if tracker:
            rate = tracker.get_rate(var, visible)
        else:
            rate = 0.0
        urgencies[var] = compute_drive(var, float(value), rate)

    # Preparation drive: known threats the agent can't yet handle
    # "I know zombie hurts (from body rules/experience) but I have no sword"
    # This creates urgency to PREPARE before the threat materializes
    if tracker:
        for (concept_id, var), rate in tracker.conditional_rates.items():
            if var == "health" and rate < -0.5 and concept_id != "_background":
                # Severe health threat known. Can we handle it?
                cause_concept = concept_store.query_text(concept_id)
                if cause_concept:
                    for link in cause_concept.causal_links:
                        if link.action == "do" and link.requires:
                            # Need weapon/tool to handle this threat
                            has_req = all(
                                inventory.get(r, 0) >= n
                                for r, n in link.requires.items()
                            )
                            if not has_req:
                                # Can't handle threat → preparation urgency
                                # Scale: proportional to threat severity
                                urgencies["preparation"] = min(0.1, abs(rate) * 0.05)

    # Curiosity drive
    if spatial_map is not None:
        urgencies["curiosity"] = compute_curiosity(concept_store, spatial_map)
    else:
        urgencies["curiosity"] = 0.5

    # Most urgent need
    critical = max(urgencies, key=urgencies.get)  # type: ignore[arg-type]

    if critical == "curiosity":
        return "explore", []

    if critical == "preparation":
        # Find what we need to prepare for the threat
        if tracker:
            for (concept_id, var), rate in tracker.conditional_rates.items():
                if var == "health" and rate < -0.5 and concept_id != "_background":
                    cause_concept = concept_store.query_text(concept_id)
                    if cause_concept:
                        for link in cause_concept.causal_links:
                            if link.action == "do":
                                plan = concept_store.plan(link.result)
                                if plan:
                                    return link.result, plan
        return "explore", []

    # Strategy 1: direct restore ("do cow restores food")
    plan = concept_store.plan(f"restore_{critical}")
    if plan:
        return f"restore_{critical}", plan

    # Strategy 2: find cause → remove it
    # "health dropping because zombie" → "kill zombie" → "craft sword"
    if tracker:
        worst_cause = None
        worst_rate = 0.0
        for (concept_id, var), rate in tracker.conditional_rates.items():
            if var == critical and rate < worst_rate:
                worst_rate = rate
                worst_cause = concept_id

        if worst_cause:
            cause_concept = concept_store.query_text(worst_cause)
            if cause_concept:
                for link in cause_concept.causal_links:
                    if link.action == "do":
                        cause_plan = concept_store.plan(link.result)
                        if cause_plan:
                            return link.result, cause_plan

    # Nothing helps → curiosity (explore, maybe discover solution)
    return "explore", []


def get_drive_strengths(
    inventory: dict[str, int],
    tracker: HomeostaticTracker | None = None,
    visual_field: VisualField | None = None,
    spatial_map: Any = None,
) -> dict[str, float]:
    """Drive strengths for UI display."""
    visible = visual_field.visible_concepts() if visual_field else set()
    drives: dict[str, float] = {}
    for var in HOMEOSTATIC_VARS:
        value = inventory.get(var, 9)
        rate = tracker.get_rate(var, visible) if tracker else 0.0
        drives[var] = compute_drive(var, float(value), rate)
    if spatial_map is not None:
        from snks.agent.concept_store import ConceptStore
        # Can't compute curiosity without store — show 0
        drives["curiosity"] = 0.0
    return drives
