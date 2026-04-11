"""Stage 79 — Learned rule data class for runtime rule induction.

A `LearnedRule` is what `RuleNursery.promote()` writes to
`ConceptStore.learned_rules`. The textbook (`crafter_textbook.yaml`)
holds *facts* (the parental knowledge in the three-category taxonomy);
learned rules hold *experience* discovered at runtime via surprise
accumulation. The two are kept in separate lists so the textbook is
never rewritten by online learning — see
`feedback_textbook_taxonomy.md` and `feedback_self_induced_rules.md`.

Phase 7 in `ConceptStore._apply_tick` iterates `learned_rules` after
the textbook rules and additively applies any whose `precondition`
matches the current sim state.

Design: docs/superpowers/specs/2026-04-11-stage79-rule-nursery-design.md
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from snks.learning.surprise_accumulator import ContextKey, quartile_for


@dataclass
class LearnedRule:
    """A symbolic rule learned at runtime by the nursery.

    `precondition` is a `ContextKey` — the visible-concept set, the
    body-quartile tuple, and the action that the rule fires on.
    `effect` is a body-delta dict (only the load-bearing vars). The
    rule is *additive* — its effect is added to whatever the textbook
    rules produced for the same tick, in Phase 7 of `_apply_tick`.

    `confidence` starts at 0.5 and grows with verified firings via
    the existing `verify_outcome` machinery.

    `n_observations` is the cumulative count of (emit + verify)
    records the rule was promoted on, for diagnostics.
    """

    precondition: ContextKey
    effect: dict[str, float]
    confidence: float = 0.5
    n_observations: int = 0
    source: str = "runtime_nursery"

    def matches(
        self,
        visible: set[str] | frozenset[str],
        body: dict[str, float],
        primitive: str,
    ) -> bool:
        """Predicate evaluation against a current sim state.

        Match conditions:
          1. The rule's `visible` set must be a SUBSET of the agent's
             current visible concepts (rule fires whenever its
             required concepts are all present, even if more are also
             visible).
          2. The action / primitive must match exactly.
          3. If the rule's `body_quartiles` is non-trivial (i.e. not
             all zeros), the current body's quartiles must match
             exactly. If the rule was emitted from an L1 (coarse)
             bucket — body_quartiles == (0, 0, 0, 0) — then body
             state is not load-bearing and the body check is skipped.

        Subset (not equality) on `visible` lets a rule like
        `{skeleton} → health -0.5` fire even when other concepts
        (tree, water, ...) are also visible.
        """
        if not self.precondition.visible.issubset(set(visible)):
            return False
        if self.precondition.action != primitive:
            return False
        if not self.precondition.is_l1():
            # L2 rule — body state is part of the precondition.
            current_quartiles = tuple(
                quartile_for(float(body.get(var, 0.0)))
                for var in ("health", "food", "drink", "energy")
            )
            if current_quartiles != self.precondition.body_quartiles:
                return False
        return True

    # ---- Stage 82: persistence (knowledge flow) ---------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dict for cross-episode persistence.

        Stage 82: learned rules are the core of the knowledge-flow
        principle — experience promotes to facts that the next episode
        (or the next agent, or the teacher) can consume. This format
        is stable, inspectable, and mergeable with textbook YAML.
        """
        return {
            "precondition": {
                "visible": sorted(self.precondition.visible),
                "body_quartiles": list(self.precondition.body_quartiles),
                "action": self.precondition.action,
            },
            "effect": {k: float(v) for k, v in self.effect.items()},
            "confidence": float(self.confidence),
            "n_observations": int(self.n_observations),
            "source": str(self.source),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LearnedRule":
        """Deserialize from the format produced by to_dict()."""
        pre = data["precondition"]
        context = ContextKey(
            visible=frozenset(pre["visible"]),
            body_quartiles=tuple(pre["body_quartiles"]),  # type: ignore[arg-type]
            action=pre["action"],
        )
        return cls(
            precondition=context,
            effect=dict(data.get("effect", {})),
            confidence=float(data.get("confidence", 0.5)),
            n_observations=int(data.get("n_observations", 0)),
            source=str(data.get("source", "runtime_nursery")),
        )
