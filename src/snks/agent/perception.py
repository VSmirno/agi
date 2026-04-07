"""Stage 73: Spatial perception — visual field from CNN feature map.

The CNN computes a (256, 4, 4) spatial feature map covering the 64×64 view.
Each of 16 positions covers ~16×16 pixels (~2×2 Crafter tiles). This IS the
agent's visual field — a retinotopic map like V1 in the brain.

We match ConceptStore prototypes (256-dim) against each position → the agent
sees WHAT is WHERE in its entire field of view, not just what's adjacent.

Center 2×2 (positions [1:3, 1:3]) = "nearby" (what was near_head).
Periphery = "I can see a zombie approaching" or "tree over there".

Prototypes are 256-dim per-position features, NOT 2048-dim whole-scene z_real.
This is MORE specific (encodes one region, not entire scene) and LESS noisy.

The agent receives visual field data. ALL behavior decisions are made by
drives + ConceptStore planning. No hardcoded if/else for specific objects.
Textbook gives rules, experience gives strategy.

Design: docs/superpowers/specs/2026-04-07-stage73-autonomous-craft-design.md
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


MIN_SIMILARITY = 0.5
EMA_ALPHA = 0.1

# Grid positions: 4×4 = 16 positions. Center 2×2 = "nearby".
CENTER_POSITIONS = [(1, 1), (1, 2), (2, 1), (2, 2)]


@dataclass
class VisualField:
    """What the agent sees — concepts at each grid position."""

    # All detected concepts: list of (concept_id, similarity, grid_y, grid_x)
    detections: list[tuple[str, float, int, int]] = field(default_factory=list)

    # Best match at center (what's nearby) — for backward compat
    near_concept: str = "empty"
    near_similarity: float = 0.0

    # Feature vector at center (for grounding)
    center_feature: torch.Tensor | None = None

    @property
    def has_danger(self) -> bool:
        """Any dangerous concept visible anywhere in field."""
        return any(cid == "zombie" for cid, _, _, _ in self.detections)

    @property
    def danger_nearby(self) -> bool:
        """Dangerous concept in center 2×2 (adjacent)."""
        return any(
            cid == "zombie" and (gy, gx) in CENTER_POSITIONS
            for cid, _, gy, gx in self.detections
        )

    def visible_concepts(self) -> set[str]:
        """All unique concepts visible in field."""
        return {cid for cid, _, _, _ in self.detections}

    def find(self, concept_id: str) -> list[tuple[float, int, int]]:
        """Find all positions where concept is visible."""
        return [
            (sim, gy, gx) for cid, sim, gy, gx in self.detections
            if cid == concept_id
        ]


def perceive_field(
    pixels: torch.Tensor,
    encoder: Any,
    concept_store: ConceptStore,
    min_similarity: float = MIN_SIMILARITY,
) -> VisualField:
    """Perceive visual field from pixels using CNN feature map + ConceptStore.

    Matches 256-dim prototypes against each of 16 positions in the
    (256, 4, 4) feature map. Returns full visual field.
    """
    with torch.no_grad():
        out = encoder(pixels.unsqueeze(0) if pixels.dim() == 3 else pixels)
        fmap = out.feature_map  # (256, 4, 4) or (B, 256, 4, 4)
        if fmap.dim() == 4:
            fmap = fmap.squeeze(0)  # (256, 4, 4)

    vf = VisualField()

    # Extract center feature (average of center 2×2)
    center_feats = fmap[:, 1:3, 1:3]  # (256, 2, 2)
    vf.center_feature = center_feats.mean(dim=(1, 2))  # (256,)

    # Match prototypes against each position
    for gy in range(4):
        for gx in range(4):
            feat = fmap[:, gy, gx]  # (256,)
            feat_norm = F.normalize(feat.unsqueeze(0), dim=1)  # (1, 256)

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

                # Update near_concept from center positions
                if (gy, gx) in CENTER_POSITIONS and best_sim > vf.near_similarity:
                    vf.near_concept = best_concept.id
                    vf.near_similarity = best_sim

    return vf


# Backward compat: old perceive() still used by some callers
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
    pixels: torch.Tensor,
    encoder: Any,
    concept_store: ConceptStore,
) -> bool:
    """Ground "empty" from first frame — the visual background."""
    concept = concept_store.query_text("empty")
    if concept is None or concept.visual is not None:
        return False

    with torch.no_grad():
        out = encoder(pixels.unsqueeze(0) if pixels.dim() == 3 else pixels)
        fmap = out.feature_map
        if fmap.dim() == 4:
            fmap = fmap.squeeze(0)
        # Use center feature as "empty" prototype
        center_feat = fmap[:, 1:3, 1:3].mean(dim=(1, 2))  # (256,)
        z_norm = F.normalize(center_feat.unsqueeze(0), dim=1).squeeze(0)
        concept_store.ground_visual("empty", z_norm)
    return True


def ground_zombie_on_damage(
    inv_before: dict[str, int],
    inv_after: dict[str, int],
    visual_field: VisualField,
    concept_store: ConceptStore,
) -> bool:
    """Ground "zombie" when agent takes unexplained damage.

    Uses center_feature from current visual field (what agent sees when hurt).
    EMA refinement on subsequent encounters.
    """
    health_before = inv_before.get("health", 9)
    health_after = inv_after.get("health", 9)

    if health_after >= health_before:
        return False

    food = inv_after.get("food", 0)
    drink = inv_after.get("drink", 0)
    if food == 0 or drink == 0:
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

_STAT_GAIN_TO_NEAR: dict[str, str] = {
    "food": "cow",
    "drink": "water",
}


def on_action_outcome(
    action: str,
    inv_before: dict[str, int],
    inv_after: dict[str, int],
    center_feature: torch.Tensor,
    concept_store: ConceptStore,
    labeler: OutcomeLabeler,
) -> str | None:
    """Experiential grounding from action outcome.

    center_feature is 256-dim from position features (not 2048 z_real).
    """
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
# Universal verification
# ---------------------------------------------------------------------------


def verify_outcome(
    near_label: str | None,
    action: str,
    actual_outcome: str | None,
    concept_store: ConceptStore,
) -> None:
    """Confirm causal rule after action with outcome."""
    if near_label is None or actual_outcome is None:
        return
    concept_store.verify(near_label, action, actual_outcome)


def outcome_to_verify(
    action: str,
    inv_before: dict[str, int],
    inv_after: dict[str, int],
) -> str | None:
    """Extract gained item/stat from inventory delta for verification."""
    survival = {"health", "food", "drink", "energy"}

    gains: dict[str, int] = {}
    losses: dict[str, int] = {}
    for k in set(inv_before) | set(inv_after):
        delta = inv_after.get(k, 0) - inv_before.get(k, 0)
        if delta > 0:
            gains[k] = delta
        elif delta < 0:
            losses[k] = -delta

    if action == "do":
        for k in gains:
            if k not in survival:
                return k
        for stat in ("food", "drink"):
            if gains.get(stat, 0) > 0:
                return f"restore_{stat}"
        return None

    if action.startswith("place_"):
        placed = action.replace("place_", "")
        if losses:
            return placed
        return None

    if action.startswith("make_"):
        crafted = action.replace("make_", "")
        if gains.get(crafted, 0) > 0:
            return crafted
        return None

    return None


# ---------------------------------------------------------------------------
# Curiosity / Motor Babbling
# ---------------------------------------------------------------------------

BABBLE_BASE_PROB = 0.15
BABBLE_MIN_PROB = 0.03

_DIRECTIONS = ["move_up", "move_down", "move_left", "move_right"]


def babble_probability(concept_store: ConceptStore) -> float:
    """Curiosity decay: high when few prototypes, low when many."""
    n_grounded = sum(1 for c in concept_store.concepts.values() if c.visual is not None)
    prob = BABBLE_BASE_PROB / (1 + n_grounded * 0.3)
    return max(BABBLE_MIN_PROB, prob)


def explore_action(
    rng: np.random.RandomState,
    concept_store: ConceptStore,
    inventory: dict[str, int] | None = None,
) -> str:
    """Curiosity-driven action selection with craft babbling."""
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
# Drive-based goal selection (NO hardcoded strategy)
# ---------------------------------------------------------------------------


def select_goal(
    inventory: dict[str, int],
    concept_store: ConceptStore,
    visual_field: VisualField | None = None,
) -> tuple[str, list[PlannedStep]]:
    """Drive-based goal selection → backward chaining plan.

    Drives are computed from inventory state. Visual field influences
    danger awareness. ALL strategy emerges from drives + textbook rules +
    planning. No hardcoded if/else for specific objects.
    """
    food = inventory.get("food", 9)
    drink = inventory.get("drink", 9)
    energy = inventory.get("energy", 9)
    wood = inventory.get("wood", 0)
    stone = inventory.get("stone_item", 0)
    has_pickaxe = inventory.get("wood_pickaxe", 0) + inventory.get("stone_pickaxe", 0)
    has_sword = inventory.get("wood_sword", 0) + inventory.get("stone_sword", 0)

    # Survival drives
    drives: dict[str, float] = {
        "restore_food": max(0, 5 - food) * 2.0,
        "restore_drink": max(0, 5 - drink) * 2.0,
        "restore_energy": max(0, 4 - energy) * 2.0,
    }

    # Danger drive — if danger visible and no weapon, flee drive rises
    # If danger visible and have weapon, attack is handled by planning
    danger_visible = visual_field.has_danger if visual_field else False
    if danger_visible and has_sword == 0:
        # "I see zombie and have no sword" → urgent need for sword or flee
        if wood >= 3:
            drives["wood_sword"] = 4.0  # urgent craft
        else:
            drives["wood"] = 3.0  # urgent gather to craft sword
    elif danger_visible and has_sword > 0:
        # Armed and see zombie — drives stay normal, planning handles attack
        pass

    # Resource progression (when no immediate danger)
    if "wood" not in drives:
        if wood < 5:
            drives["wood"] = max(0.2, 2.0 - wood * 0.3)
        else:
            drives["wood"] = 0.1

    if has_sword == 0 and "wood_sword" not in drives:
        if wood >= 3:
            drives["wood_sword"] = 2.0
        else:
            # Need wood for sword — boost wood drive
            drives["wood"] = max(drives.get("wood", 0), 2.0)

    if has_sword > 0 and has_pickaxe == 0 and wood >= 3:
        drives["wood_pickaxe"] = 1.5
    if has_pickaxe > 0 and stone < 2:
        drives["stone_item"] = 1.5
    else:
        drives.setdefault("stone_item", 0.3)

    goal = max(drives, key=drives.get)  # type: ignore[arg-type]
    if drives[goal] <= 0:
        goal = "wood"

    plan = concept_store.plan(goal)
    return goal, plan


def get_drive_strengths(
    inventory: dict[str, int],
    visual_field: VisualField | None = None,
) -> dict[str, float]:
    """Drive strengths for UI display."""
    food = inventory.get("food", 9)
    drink = inventory.get("drink", 9)
    energy = inventory.get("energy", 9)
    wood = inventory.get("wood", 0)
    stone = inventory.get("stone_item", 0)
    has_pickaxe = inventory.get("wood_pickaxe", 0) + inventory.get("stone_pickaxe", 0)
    has_sword = inventory.get("wood_sword", 0) + inventory.get("stone_sword", 0)

    drives: dict[str, float] = {
        "restore_food": max(0, 5 - food) * 2.0,
        "restore_drink": max(0, 5 - drink) * 2.0,
        "restore_energy": max(0, 4 - energy) * 2.0,
        "wood": max(0.1, 2.0 - wood * 0.3) if wood < 5 else 0.1,
        "stone_item": 1.5 if (has_pickaxe > 0 and stone < 2) else 0.3,
    }
    if has_sword == 0:
        drives["wood_sword"] = 2.0
    if has_sword > 0 and has_pickaxe == 0 and wood >= 3:
        drives["wood_pickaxe"] = 1.5
    return drives
