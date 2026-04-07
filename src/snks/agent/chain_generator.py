"""Stage 71: ChainGenerator — generate ScenarioChains from ConceptStore rules.

Replaces hardcoded TREE_CHAIN, COAL_CHAIN, IRON_CHAIN with auto-generated
chains derived from textbook rules via backward chaining.

Design: docs/superpowers/specs/2026-04-07-stage71-text-visual-integration-design.md
"""

from __future__ import annotations

from snks.agent.concept_store import ConceptStore, PlannedStep
from snks.agent.scenario_runner import ScenarioStep

# Action name mapping: textbook action → Crafter env action
_ACTION_MAP: dict[tuple[str, str], str] = {
    # (textbook_action, result) → crafter_action
    # "do" stays "do"
    # "place" + "table" → "place_table"
    # "make" + "wood_pickaxe" → "make_wood_pickaxe"
}

# Repeat counts for data collection
_DEFAULT_GATHER_REPEAT = 5
_DEFAULT_CRAFT_REPEAT = 1


def _to_crafter_action(step: PlannedStep) -> str:
    """Convert PlannedStep to Crafter action name."""
    if step.action == "do":
        return "do"
    if step.action == "place":
        return f"place_{step.expected_gain}"
    if step.action == "make":
        return f"make_{step.expected_gain}"
    return step.action


def _near_label(step: PlannedStep) -> str:
    """Determine near_label for NearDetector training data."""
    if step.action == "do":
        return step.target  # "tree", "stone", "coal", "iron"
    if step.action == "place":
        return "empty"  # placing requires empty space
    if step.action == "make":
        return "table"  # crafting requires table nearby
    return step.target


class ChainGenerator:
    """Generate ScenarioChains from ConceptStore backward chaining.

    Usage:
        gen = ChainGenerator(store)
        chain = gen.generate("iron_item")
        # chain: list[ScenarioStep] ready for ScenarioRunner
    """

    def __init__(
        self,
        store: ConceptStore,
        *,
        gather_repeat: int = _DEFAULT_GATHER_REPEAT,
        craft_repeat: int = _DEFAULT_CRAFT_REPEAT,
        max_nav_steps: int = 300,
        use_semantic_nav: bool = True,
    ) -> None:
        self.store = store
        self.gather_repeat = gather_repeat
        self.craft_repeat = craft_repeat
        self.max_nav_steps = max_nav_steps
        self.use_semantic_nav = use_semantic_nav

    def generate(self, goal: str) -> list[ScenarioStep]:
        """Generate a ScenarioChain for a given goal item.

        Args:
            goal: concept_id of the desired outcome (e.g. "iron_item")

        Returns:
            List of ScenarioStep ready for ScenarioRunner.run_chain()
        """
        planned = self.store.plan(goal)
        return [self._convert(step) for step in planned]

    def _convert(self, step: PlannedStep) -> ScenarioStep:
        """Convert PlannedStep to ScenarioStep."""
        action = _to_crafter_action(step)
        label = _near_label(step)

        is_gather = step.action == "do"
        repeat = self.gather_repeat if is_gather else self.craft_repeat

        # For gather: navigate to target object
        # For craft/place: navigate to near target (table/empty) or act in place
        navigate_to: str | None
        if is_gather:
            navigate_to = step.target
        elif step.action == "place":
            navigate_to = None  # place at current location
        elif step.action == "make":
            navigate_to = step.near  # navigate to table
        else:
            navigate_to = step.target

        return ScenarioStep(
            navigate_to=navigate_to,
            action=action,
            near_label=label,
            prerequisite_inv=step.requires,
            repeat=repeat,
            max_nav_steps=self.max_nav_steps,
            use_semantic_nav=self.use_semantic_nav,
            continue_on_probe_fail=is_gather,
        )

    def available_goals(self) -> list[str]:
        """List all items that can be produced via backward chaining."""
        goals = []
        for concept in self.store.concepts.values():
            for link in concept.causal_links:
                if link.result not in ("flee", ) and link.result.startswith(("kill_",)) is False:
                    goals.append(link.result)
        return sorted(set(goals))
