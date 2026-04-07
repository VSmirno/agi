"""Stage 72: Self-organized perception module.

Replaces NearDetector (supervised near_head) with ConceptStore.query_visual()
(cosine similarity matching against prototypes learned from experience).

Components:
- perceive(): pixels → concept + z_real via frozen CNN + ConceptStore
- on_action_outcome(): experiential grounding (one-shot + EMA refinement)
- select_goal(): drive-based goal selection → ConceptStore.plan()

Design: docs/superpowers/specs/2026-04-07-stage72-perception-pivot-design.md
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from snks.agent.concept_store import Concept, ConceptStore, PlannedStep
    from snks.agent.outcome_labeler import OutcomeLabeler


MIN_SIMILARITY = 0.5
EMA_ALPHA = 0.1  # new observation weight in prototype update


def perceive(
    pixels: torch.Tensor,
    encoder: Any,
    concept_store: ConceptStore,
    min_similarity: float = MIN_SIMILARITY,
) -> tuple[Concept | None, torch.Tensor]:
    """Perceive nearby object from pixels using frozen CNN + ConceptStore.

    Returns (concept, z_real). concept is None if best match
    similarity is below min_similarity (unknown object).
    """
    with torch.no_grad():
        out = encoder(pixels.unsqueeze(0) if pixels.dim() == 3 else pixels)
        z_real = out.z_real.squeeze(0)  # (2048,)

    concept, similarity = concept_store.query_visual_scored(z_real)
    if similarity < min_similarity:
        return None, z_real
    return concept, z_real


def on_action_outcome(
    action: str,
    inv_before: dict[str, int],
    inv_after: dict[str, int],
    z_real: torch.Tensor,
    concept_store: ConceptStore,
    labeler: OutcomeLabeler,
) -> str | None:
    """Experiential grounding: learn visual prototype from action outcome.

    One-shot grounding on first encounter, EMA refinement on subsequent.

    Returns label string if grounding happened, None otherwise.
    """
    label = labeler.label(action, inv_before, inv_after)
    if label is None:
        return None

    concept = concept_store.query_text(label)
    if concept is None:
        return None

    z_norm = F.normalize(z_real.unsqueeze(0), dim=1).squeeze(0)

    if concept.visual is None:
        # First encounter — one-shot grounding
        concept_store.ground_visual(label, z_norm)
    else:
        # Seen before — EMA update
        concept.visual = F.normalize(
            ((1 - EMA_ALPHA) * concept.visual + EMA_ALPHA * z_norm).unsqueeze(0),
            dim=1,
        ).squeeze(0)

    return label


def select_goal(
    inventory: dict[str, int],
    concept_store: ConceptStore,
) -> tuple[str, list[PlannedStep]]:
    """Drive-based goal selection → backward chaining plan.

    Returns (goal_name, planned_steps). Drive competition picks
    the most urgent need; ConceptStore.plan() generates the chain.
    """
    food = inventory.get("food", 9)
    drink = inventory.get("drink", 9)
    energy = inventory.get("energy", 9)

    drives: dict[str, float] = {
        "restore_food": max(0, 5 - food) * 2.0,
        "restore_drink": max(0, 5 - drink) * 2.0,
        "restore_energy": max(0, 4 - energy) * 2.0,
        "wood": 1.0,
        "stone_item": 0.5,
    }

    goal = max(drives, key=drives.get)  # type: ignore[arg-type]
    if drives[goal] <= 0:
        goal = "wood"

    # Map survival drives to plannable concepts
    _DRIVE_TO_GOAL: dict[str, str] = {
        "restore_food": "restore_food",
        "restore_drink": "restore_drink",
        "restore_energy": "restore_energy",
    }
    plan_goal = _DRIVE_TO_GOAL.get(goal, goal)

    plan = concept_store.plan(plan_goal)
    if not plan and goal.startswith("restore_"):
        # Fallback: survival goals may not have backward chains yet.
        # Use direct seek behavior instead.
        pass

    return goal, plan


def get_drive_strengths(inventory: dict[str, int]) -> dict[str, float]:
    """Return current drive strengths for UI display."""
    food = inventory.get("food", 9)
    drink = inventory.get("drink", 9)
    energy = inventory.get("energy", 9)
    return {
        "restore_food": max(0, 5 - food) * 2.0,
        "restore_drink": max(0, 5 - drink) * 2.0,
        "restore_energy": max(0, 4 - energy) * 2.0,
        "wood": 1.0,
        "stone_item": 0.5,
    }
