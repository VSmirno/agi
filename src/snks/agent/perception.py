"""Stage 73: Self-organized perception + autonomous craft.

Extends Stage 72 motor babbling with:
- Empty grounding from first frame (figure/ground separation)
- Zombie grounding through damage (teacher gave name, experience gives face)
- Craft babbling (place/make actions, not just "do")
- Universal verification (every outcome confirms a causal rule)

Design: docs/superpowers/specs/2026-04-07-stage73-autonomous-craft-design.md
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from snks.agent.concept_store import Concept, ConceptStore, PlannedStep
    from snks.agent.outcome_labeler import OutcomeLabeler


MIN_SIMILARITY = 0.5  # needs to be selective with few grounded prototypes
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


# ---------------------------------------------------------------------------
# Bootstrap grounding
# ---------------------------------------------------------------------------


def ground_empty_on_start(
    pixels: torch.Tensor,
    encoder: Any,
    concept_store: ConceptStore,
) -> bool:
    """Ground "empty" from first frame of first episode.

    The agent's first visual experience IS empty terrain — the background
    (ground) against which all objects (figure) are recognized.

    Called once per ConceptStore lifetime, after env.reset().
    Returns True if grounding happened.
    """
    concept = concept_store.query_text("empty")
    if concept is None or concept.visual is not None:
        return False  # already grounded or concept doesn't exist

    _, z_real = perceive(pixels, encoder, concept_store, min_similarity=999.0)
    z_norm = F.normalize(z_real.unsqueeze(0), dim=1).squeeze(0)
    concept_store.ground_visual("empty", z_norm)
    return True


def ground_zombie_on_damage(
    inv_before: dict[str, int],
    inv_after: dict[str, int],
    z_real: torch.Tensor,
    concept_store: ConceptStore,
) -> bool:
    """Ground "zombie" when agent takes unexplained damage.

    Textbook gave the name and dangerous=True. Experience gives the face.
    z_real is from current perception — may be noisy (zombie could attack
    from behind). EMA refinement improves prototype over multiple encounters.

    Returns True if grounding/update happened.
    """
    health_before = inv_before.get("health", 9)
    health_after = inv_after.get("health", 9)

    if health_after >= health_before:
        return False  # no damage

    # Starvation check: food=0 or drink=0 causes gradual health loss
    food = inv_after.get("food", 0)
    drink = inv_after.get("drink", 0)
    if food == 0 or drink == 0:
        return False  # starvation, not entity damage

    concept = concept_store.query_text("zombie")
    if concept is None:
        return False

    z_norm = F.normalize(z_real.unsqueeze(0), dim=1).squeeze(0)

    if concept.visual is None:
        # First encounter — one-shot grounding
        concept_store.ground_visual("zombie", z_norm)
    else:
        # EMA refinement
        concept.visual = F.normalize(
            ((1 - EMA_ALPHA) * concept.visual + EMA_ALPHA * z_norm).unsqueeze(0),
            dim=1,
        ).squeeze(0)

    return True


# ---------------------------------------------------------------------------
# Experiential grounding (from Stage 72)
# ---------------------------------------------------------------------------

# Survival stat changes → what was nearby
_STAT_GAIN_TO_NEAR: dict[str, str] = {
    "food": "cow",
    "drink": "water",
}


def on_action_outcome(
    action: str,
    inv_before: dict[str, int],
    inv_after: dict[str, int],
    z_real: torch.Tensor,
    concept_store: ConceptStore,
    labeler: OutcomeLabeler,
) -> str | None:
    """Experiential grounding: learn visual prototype from action outcome.

    Also detects survival stat changes (food/drink gained → cow/water nearby).

    Returns label string if grounding happened, None otherwise.
    """
    label = labeler.label(action, inv_before, inv_after)

    # Extend: detect survival stat restoration (food/drink from cow/water)
    if label is None and action == "do":
        for stat, near in _STAT_GAIN_TO_NEAR.items():
            before_val = inv_before.get(stat, 0)
            after_val = inv_after.get(stat, 0)
            if after_val > before_val:
                label = near
                break

    if label is None:
        return None

    concept = concept_store.query_text(label)
    if concept is None:
        return None

    z_norm = F.normalize(z_real.unsqueeze(0), dim=1).squeeze(0)

    if concept.visual is None:
        concept_store.ground_visual(label, z_norm)
    else:
        concept.visual = F.normalize(
            ((1 - EMA_ALPHA) * concept.visual + EMA_ALPHA * z_norm).unsqueeze(0),
            dim=1,
        ).squeeze(0)

    return label


# ---------------------------------------------------------------------------
# Universal verification (Stage 73)
# ---------------------------------------------------------------------------


def verify_outcome(
    near_label: str | None,
    action: str,
    actual_outcome: str | None,
    concept_store: ConceptStore,
) -> None:
    """After any action with outcome, confirm the causal rule.

    Args:
        near_label: concept_id of what was NEARBY (e.g. "tree")
        action: what was done (e.g. "do")
        actual_outcome: what was gained/produced (e.g. "wood")
        concept_store: ConceptStore to update confidence
    """
    if near_label is None or actual_outcome is None:
        return
    concept_store.verify(near_label, action, actual_outcome)


# ---------------------------------------------------------------------------
# Curiosity / Motor Babbling
# ---------------------------------------------------------------------------

BABBLE_BASE_PROB = 0.15  # probability of trying action during exploration
BABBLE_MIN_PROB = 0.03   # floor after many concepts grounded

# Craft actions the agent can try during babbling (when inventory allows)
_CRAFT_ACTIONS = ["place_table", "make_wood_pickaxe", "make_wood_sword"]
_DIRECTIONS = ["move_up", "move_down", "move_left", "move_right"]


def babble_probability(concept_store: ConceptStore) -> float:
    """Curiosity-driven action probability.

    High when few concepts grounded (lots to discover).
    Low when many grounded (mostly exploit, rare babble).
    """
    n_grounded = sum(1 for c in concept_store.concepts.values() if c.visual is not None)
    prob = BABBLE_BASE_PROB / (1 + n_grounded * 0.3)
    return max(BABBLE_MIN_PROB, prob)


def explore_action(
    rng: np.random.RandomState,
    concept_store: ConceptStore,
    inventory: dict[str, int] | None = None,
) -> str:
    """Select an exploration action with curiosity-driven diversity.

    Stage 73: also tries craft actions (place/make) when inventory allows.
    """
    p = babble_probability(concept_store)
    if rng.random() < p:
        # Motor babble — try an action to discover things
        if inventory is not None and inventory.get("wood", 0) >= 2:
            # Can try placing table — craft babble
            if rng.random() < 0.3:
                return "babble_place_table"
        if inventory is not None and inventory.get("wood", 0) >= 1:
            # Can try making pickaxe (if near table — will fail gracefully)
            if rng.random() < 0.15:
                return "babble_make_wood_pickaxe"
        return "babble_do"
    # Spatial exploration
    return str(rng.choice(_DIRECTIONS))


# ---------------------------------------------------------------------------
# Drive-based goal selection
# ---------------------------------------------------------------------------

def select_goal(
    inventory: dict[str, int],
    concept_store: ConceptStore,
) -> tuple[str, list[PlannedStep]]:
    """Drive-based goal selection → backward chaining plan.

    Resource drives are inventory-dependent: wood → pickaxe → stone.
    """
    food = inventory.get("food", 9)
    drink = inventory.get("drink", 9)
    energy = inventory.get("energy", 9)
    wood = inventory.get("wood", 0)
    stone = inventory.get("stone_item", 0)
    has_pickaxe = inventory.get("wood_pickaxe", 0) + inventory.get("stone_pickaxe", 0)

    drives: dict[str, float] = {
        "restore_food": max(0, 5 - food) * 2.0,
        "restore_drink": max(0, 5 - drink) * 2.0,
        "restore_energy": max(0, 4 - energy) * 2.0,
    }

    if wood < 5:
        drives["wood"] = max(0.2, 2.0 - wood * 0.3)
    else:
        drives["wood"] = 0.1

    if wood >= 2 and has_pickaxe == 0:
        drives["wood_pickaxe"] = 1.5
    elif has_pickaxe > 0 and stone < 2:
        drives["stone_item"] = 1.5
    else:
        drives["stone_item"] = 0.3

    goal = max(drives, key=drives.get)  # type: ignore[arg-type]
    if drives[goal] <= 0:
        goal = "wood"

    plan = concept_store.plan(goal)
    return goal, plan


def get_drive_strengths(inventory: dict[str, int]) -> dict[str, float]:
    """Return current drive strengths for UI display."""
    food = inventory.get("food", 9)
    drink = inventory.get("drink", 9)
    energy = inventory.get("energy", 9)
    wood = inventory.get("wood", 0)
    stone = inventory.get("stone_item", 0)
    has_pickaxe = inventory.get("wood_pickaxe", 0) + inventory.get("stone_pickaxe", 0)

    drives = {
        "restore_food": max(0, 5 - food) * 2.0,
        "restore_drink": max(0, 5 - drink) * 2.0,
        "restore_energy": max(0, 4 - energy) * 2.0,
        "wood": max(0.1, 2.0 - wood * 0.3) if wood < 5 else 0.1,
        "stone_item": 1.5 if (has_pickaxe > 0 and stone < 2) else 0.3,
    }
    if wood >= 2 and has_pickaxe == 0:
        drives["wood_pickaxe"] = 1.5
    return drives
