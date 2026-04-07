"""Stage 72: Self-organized perception module.

Core principle: concepts form from interaction, not injection.
The agent starts with NO visual prototypes. Through motor babbling
(curiosity-driven action diversity) and prediction error, it discovers
what objects look like by observing the consequences of its actions.

Pipeline:
  pixels → frozen CNN → z_real → ConceptStore.query_visual_scored()
  action outcome → OutcomeLabeler → on_action_outcome() → grounding
  drives → select_goal() → ConceptStore.plan()
  curiosity → explore_action() → motor babbling

The bootstrap mechanism:
  1. Agent walks randomly + occasionally tries "do" (motor babbling)
  2. "do" near tree → inventory gains wood → OutcomeLabeler says "tree"
  3. on_action_outcome() grounds z_real from BEFORE the action as "tree" prototype
  4. Now perceive() can recognize trees → spatial map fills → navigation works
  5. Agent transitions from babbling to goal-directed behavior

This is developmental learning: action → consequence → concept formation.
Not supervised: no labels, no teacher, no controlled environment.

Design: docs/superpowers/specs/2026-04-07-stage72-perception-pivot-design.md
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


def on_action_outcome(
    action: str,
    inv_before: dict[str, int],
    inv_after: dict[str, int],
    z_real: torch.Tensor,
    concept_store: ConceptStore,
    labeler: OutcomeLabeler,
) -> str | None:
    """Experiential grounding: learn visual prototype from action outcome.

    This is the core self-organization mechanism. When an action produces
    an inventory change, the OutcomeLabeler infers what was nearby
    (e.g. wood gained → tree was nearby). The visual embedding z_real
    captured BEFORE the action becomes the prototype for that concept.

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
        # Seen before — EMA update (consolidation)
        concept.visual = F.normalize(
            ((1 - EMA_ALPHA) * concept.visual + EMA_ALPHA * z_norm).unsqueeze(0),
            dim=1,
        ).squeeze(0)

    return label


# ---------------------------------------------------------------------------
# Curiosity / Motor Babbling
# ---------------------------------------------------------------------------

# Action diversity during exploration. Higher = more random actions tried.
# Decays as more concepts get visually grounded (less need to babble).
BABBLE_BASE_PROB = 0.15  # probability of trying "do" during exploration
BABBLE_MIN_PROB = 0.03   # floor after many concepts grounded

_ALL_ACTIONS = ["do", "sleep", "place_table", "make_wood_pickaxe"]
_DIRECTIONS = ["move_up", "move_down", "move_left", "move_right"]


def babble_probability(concept_store: ConceptStore) -> float:
    """Curiosity-driven action probability.

    High when few concepts grounded (lots to discover).
    Low when many grounded (mostly exploit, rare babble).

    Analogous to intrinsic motivation: r_int = 1/(1+count).
    """
    n_grounded = sum(1 for c in concept_store.concepts.values() if c.visual is not None)
    # Decay: 0.15 → 0.02 as grounded concepts increase
    prob = BABBLE_BASE_PROB / (1 + n_grounded * 0.3)
    return max(BABBLE_MIN_PROB, prob)


def explore_action(
    rng: np.random.RandomState,
    concept_store: ConceptStore,
) -> str:
    """Select an exploration action with curiosity-driven diversity.

    Most of the time: random move (spatial exploration).
    With babble_probability: try "do" (motor babbling for discovery).

    This is the bootstrap mechanism — without visual prototypes,
    motor babbling is the only way to discover what's in the world.
    """
    p = babble_probability(concept_store)
    if rng.random() < p:
        # Motor babble — try a random "do" to discover objects
        # First face a random direction, then do
        return "babble_do"
    # Spatial exploration — move to unvisited area
    return str(rng.choice(_DIRECTIONS))


# ---------------------------------------------------------------------------
# Drive-based goal selection
# ---------------------------------------------------------------------------

def select_goal(
    inventory: dict[str, int],
    concept_store: ConceptStore,
) -> tuple[str, list[PlannedStep]]:
    """Drive-based goal selection → backward chaining plan.

    Returns (goal_name, planned_steps). Drive competition picks
    the most urgent need; ConceptStore.plan() generates the chain.

    Resource drives are inventory-dependent: once the agent has
    enough wood, the drive shifts to stone, then coal, then iron.
    This creates natural progression without hardcoded curriculum.
    """
    food = inventory.get("food", 9)
    drink = inventory.get("drink", 9)
    energy = inventory.get("energy", 9)
    wood = inventory.get("wood", 0)
    stone = inventory.get("stone_item", 0)
    has_pickaxe = inventory.get("wood_pickaxe", 0) + inventory.get("stone_pickaxe", 0)
    has_table = inventory.get("table", 0)  # placed tables not tracked, use wood threshold

    # Survival drives (highest priority when low)
    drives: dict[str, float] = {
        "restore_food": max(0, 5 - food) * 2.0,
        "restore_drink": max(0, 5 - drink) * 2.0,
        "restore_energy": max(0, 4 - energy) * 2.0,
    }

    # Resource drives: inversely proportional to how much we have
    # Agent naturally progresses: wood → table → pickaxe → stone → ...
    if wood < 5:
        drives["wood"] = max(0.2, 2.0 - wood * 0.3)
    else:
        drives["wood"] = 0.1  # enough wood, low priority

    if wood >= 2 and has_pickaxe == 0:
        # Have wood, need pickaxe (requires table + craft)
        drives["wood_pickaxe"] = 1.5
    elif has_pickaxe > 0 and stone < 2:
        # Have pickaxe, need stone
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
