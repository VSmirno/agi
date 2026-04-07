"""Stage 71: CrafterTextbook — YAML loader for atomic causal rules.

Loads a textbook of atomic rules and registers them into a ConceptStore.
Rule parsing is regex-based (fixed format, not SVO chunker).

Design: docs/superpowers/specs/2026-04-07-stage71-text-visual-integration-design.md
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from snks.agent.concept_store import CausalLink, ConceptStore


def _parse_rule(text: str) -> dict[str, Any] | None:
    """Parse a textbook rule into structured form.

    Supported formats:
        "do OBJECT gives RESULT"
        "do OBJECT gives RESULT requires ITEM"
        "do OBJECT gives RESULT requires ITEM1 and ITEM2"
        "place RESULT on OBJECT requires ITEM"
        "make RESULT near OBJECT requires ITEM"
        "make RESULT near OBJECT requires ITEM1 and ITEM2"
        "do OBJECT with ITEM kills OBJECT"  (combat)
        "OBJECT nearby without ITEM means flee" (survival)

    Returns dict with keys: concept, action, result, requires, type
    """
    text = text.strip().lower()

    # Combat: "do zombie with wood_sword kills zombie"
    m = re.match(
        r"do (\w+) with (\w+) kills (\w+)", text
    )
    if m:
        return {
            "concept": m.group(1),
            "action": "do",
            "result": f"kill_{m.group(3)}",
            "requires": {m.group(2): 1},
            "type": "combat",
        }

    # Survival: "zombie nearby without wood_sword means flee"
    m = re.match(
        r"(\w+) nearby without (\w+) means flee", text
    )
    if m:
        return {
            "concept": m.group(1),
            "action": "nearby",
            "result": "flee",
            "requires": {},
            "without": m.group(2),
            "type": "survival",
        }

    # Gather: "do OBJECT gives RESULT [requires ...]"
    m = re.match(
        r"do (\w+) gives (\w+)(?:\s+requires\s+(.+))?$", text
    )
    if m:
        requires = _parse_requires(m.group(3)) if m.group(3) else {}
        return {
            "concept": m.group(1),
            "action": "do",
            "result": m.group(2),
            "requires": requires,
            "type": "gather",
        }

    # Place: "place RESULT on OBJECT requires ..."
    m = re.match(
        r"place (\w+) on (\w+)(?:\s+requires\s+(.+))?$", text
    )
    if m:
        requires = _parse_requires(m.group(3)) if m.group(3) else {}
        return {
            "concept": m.group(2),  # "empty"
            "action": "place",
            "result": m.group(1),  # "table"
            "requires": requires,
            "type": "craft",
        }

    # Make: "make RESULT near OBJECT requires ..."
    m = re.match(
        r"make (\w+) near (\w+)(?:\s+requires\s+(.+))?$", text
    )
    if m:
        requires = _parse_requires(m.group(3)) if m.group(3) else {}
        return {
            "concept": m.group(2),  # "table"
            "action": "make",
            "result": m.group(1),  # "wood_pickaxe"
            "requires": requires,
            "type": "craft",
        }

    return None


def _parse_requires(text: str) -> dict[str, int]:
    """Parse 'wood and stone_item' → {'wood': 1, 'stone_item': 1}"""
    items = re.split(r"\s+and\s+", text.strip())
    return {item.strip(): 1 for item in items if item.strip()}


class CrafterTextbook:
    """Loads a YAML textbook and registers concepts + rules into ConceptStore."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        with open(self.path) as f:
            self.data = yaml.safe_load(f)

    @property
    def vocabulary(self) -> list[dict[str, Any]]:
        return self.data.get("vocabulary", [])

    @property
    def rules(self) -> list[str]:
        return self.data.get("rules", [])

    @property
    def domain(self) -> str:
        return self.data.get("domain", "unknown")

    def load_into(self, store: ConceptStore) -> int:
        """Parse vocabulary and rules, register into ConceptStore.

        Returns number of causal links added.
        """
        # Register vocabulary
        for entry in self.vocabulary:
            attrs = {k: v for k, v in entry.items() if k != "id"}
            store.register(entry["id"], attrs)

        # Parse and add rules
        links_added = 0
        for rule_text in self.rules:
            parsed = _parse_rule(rule_text)
            if parsed is None:
                continue

            concept_id = parsed["concept"]
            # Ensure concept exists
            store.register(concept_id)
            # Ensure result concept exists
            if parsed["result"] not in ("flee",):
                store.register(parsed["result"])

            link = CausalLink(
                action=parsed["action"],
                result=parsed["result"],
                requires=parsed.get("requires", {}),
                confidence=0.5,
            )

            # For survival rules, store "without" as negative condition
            if parsed["type"] == "survival":
                link.condition = f"without_{parsed.get('without', '')}"

            store.add_causal(concept_id, link)
            links_added += 1

        return links_added
