"""Stage 71: ConceptStore — unified omnimodal concept storage.

One concept = one entity with all modalities (visual, text, attributes,
causal links). Replaces GroundingMap + CausalWorldModel for the
text-visual integration pipeline.

Design: docs/superpowers/specs/2026-04-07-stage71-text-visual-integration-design.md
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from snks.agent.forward_sim_types import RuleEffect


@dataclass
class CausalLink:
    """One causal rule: action on this concept → result.

    Stage 77a adds `kind` discriminator and `effect` structured dispatch.
    The legacy `result: str` field is kept temporarily for backward compat
    with Stage 71-76 code during staged refactor (removed in Commit 8).

    Valid `kind` values:
      - "action_triggered" — fires on agent `do`/`make`/`place`/`sleep`
      - "passive_body_rate" — per-tick unconditional body delta
      - "passive_movement" — entity moves per tick per behavior
      - "passive_spatial" — adjacent damage / effect
      - "passive_stateful" — body delta while condition holds

    For action_triggered rules: `action` + `effect` describe what happens
    when the agent performs the action on `concept` (inherited from Concept).

    For passive rules: `effect` describes the per-tick update; `action` is
    typically None or "_passive".
    """

    action: str  # "do", "make", "place", "sleep", or "_passive" for passive rules
    result: str = ""  # DEPRECATED: string outcome id, kept for backward compat (removed in Commit 8)
    requires: dict[str, int] = field(default_factory=dict)  # {wood_pickaxe: 1}
    condition: str | None = None  # "nearby", None
    confidence: float = 0.5  # starts at 0.5 (from textbook), grows with verification

    # Stage 77a additions — structured dispatch
    kind: str = "action_triggered"  # discriminator for new effect-based dispatch
    effect: "RuleEffect | None" = None  # structured effect (replaces `result`)


MAX_OBSERVATIONS = 20  # raw feature samples per concept for metric learning


@dataclass
class Concept:
    """One concept with all modalities."""

    id: str
    visual: torch.Tensor | None = None  # center_feature prototype (EMA)
    text_sdr: torch.Tensor | None = None  # SDR from GroundedTokenizer
    attributes: dict[str, Any] = field(default_factory=dict)
    causal_links: list[CausalLink] = field(default_factory=list)
    confidence: float = 0.5
    observations: list[torch.Tensor] = field(default_factory=list)  # raw features

    def add_observation(self, feature: torch.Tensor) -> None:
        """Store raw feature from verified interaction (for metric learning)."""
        self.observations.append(feature.detach().clone())
        if len(self.observations) > MAX_OBSERVATIONS:
            self.observations.pop(0)

    def find_causal(
        self,
        action: str,
        check_requires: dict[str, int] | None = None,
    ) -> CausalLink | None:
        """Find a causal link by action, optionally checking inventory requirements.

        When multiple links match, prefers the most specific one (most requires
        items matching the inventory). This resolves ambiguity when e.g.
        make_wood_pickaxe (requires wood) and make_stone_pickaxe (requires
        wood + stone_item) both match an inventory with wood + stone_item.

        Args:
            action: action name to match ("do", "make", "place")
            check_requires: if provided, only return link if inventory
                           has all required items (>=).
        """
        candidates: list[tuple[int, CausalLink]] = []
        for link in self.causal_links:
            if link.action != action:
                continue
            if check_requires is not None and link.requires:
                if not all(
                    check_requires.get(item, 0) >= count
                    for item, count in link.requires.items()
                ):
                    continue
            # Score = number of requires items (more specific = higher)
            score = len(link.requires)
            candidates.append((score, link))
        if not candidates:
            return None
        # Return most specific match
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]


@dataclass
class PlannedStep:
    """One step in a backward-chained plan."""

    action: str  # "do", "make", "place"
    target: str  # concept_id to navigate to ("tree", "iron")
    near: str | None  # "table" for craft, None for gather
    expected_gain: str  # what we get ("wood", "iron_item")
    requires: dict[str, int] = field(default_factory=dict)


@dataclass
class SurpriseEvent:
    """An unexpected outcome not predicted by any rule."""

    outcome: str
    action: str
    near: str | None = None


@dataclass
class Prediction:
    """A prediction made before an action."""

    concept_id: str
    action: str
    expected: str
    confidence: float
    link: CausalLink | None = None


CONFIRM_DELTA = 0.15
REFUTE_DELTA = 0.15


class ConceptStore:
    """Unified omnimodal concept storage.

    Each concept stores visual embedding, text SDR, attributes, and
    causal links. Supports query by any modality, causal prediction,
    backward chaining planning, and experience-based verification.
    """

    def __init__(self) -> None:
        self.concepts: dict[str, Concept] = {}
        self.surprises: list[SurpriseEvent] = []

    # --- Registration ---

    def register(self, id: str, attributes: dict[str, Any] | None = None) -> Concept:
        """Register a new concept (or return existing)."""
        if id in self.concepts:
            if attributes:
                self.concepts[id].attributes.update(attributes)
            return self.concepts[id]
        concept = Concept(id=id, attributes=attributes or {})
        self.concepts[id] = concept
        return concept

    def add_causal(self, concept_id: str, link: CausalLink) -> None:
        """Add a causal link to a concept."""
        if concept_id not in self.concepts:
            self.register(concept_id)
        self.concepts[concept_id].causal_links.append(link)

    # --- Grounding ---

    def ground_visual(self, id: str, z_real: torch.Tensor) -> None:
        """Bind a visual embedding to a concept."""
        if id not in self.concepts:
            self.register(id)
        self.concepts[id].visual = z_real.detach().clone()

    def ground_text(self, id: str, text_sdr: torch.Tensor) -> None:
        """Bind a text SDR to a concept."""
        if id not in self.concepts:
            self.register(id)
        self.concepts[id].text_sdr = text_sdr.detach().clone()

    # --- Query ---

    def query_text(self, word: str) -> Concept | None:
        """Find concept by text label (exact match)."""
        return self.concepts.get(word)

    def query_visual(self, z_real: torch.Tensor) -> Concept | None:
        """Find closest concept by visual embedding (cosine similarity)."""
        concept, _ = self.query_visual_scored(z_real)
        return concept

    def query_visual_scored(
        self, z_real: torch.Tensor
    ) -> tuple[Concept | None, float]:
        """Find closest concept by visual embedding, returning similarity score.

        Returns (best_concept, best_similarity). If no concepts have visual
        embeddings, returns (None, -1.0).
        """
        best_concept = None
        best_sim = -1.0
        z_norm = torch.nn.functional.normalize(z_real.unsqueeze(0), dim=1)
        for concept in self.concepts.values():
            if concept.visual is None:
                continue
            c_norm = torch.nn.functional.normalize(
                concept.visual.unsqueeze(0), dim=1
            )
            sim = (z_norm @ c_norm.T).item()
            if sim > best_sim:
                best_sim = sim
                best_concept = concept
        return best_concept, best_sim

    # --- Causal reasoning ---

    def predict(
        self, concept_id: str, action: str, inventory: dict[str, int]
    ) -> CausalLink | None:
        """Predict outcome of action on concept given inventory."""
        concept = self.query_text(concept_id)
        if concept is None:
            return None
        return concept.find_causal(action=action, check_requires=inventory)

    def plan(
        self,
        goal_id: str,
        inventory: dict[str, int] | None = None,
    ) -> list[PlannedStep]:
        """Backward chaining: find sequence of steps to achieve goal.

        Returns forward-ordered list of PlannedStep (ready to execute).

        Args:
            goal_id: target concept id to produce.
            inventory: current inventory. If provided, prerequisites already
                       present are skipped — avoids redundant steps when the
                       agent already holds required items (e.g. don't re-craft
                       sword if already wielding one).
        """
        steps: list[PlannedStep] = []
        visited: set[str] = set()
        inv = dict(inventory) if inventory else {}
        self._plan_recursive(goal_id, steps, visited, inv)
        return steps

    def _plan_recursive(
        self,
        goal_id: str,
        steps: list[PlannedStep],
        visited: set[str],
        inventory: dict[str, int] | None = None,
    ) -> None:
        """Backward chaining helper. Builds plan in forward order.

        Skips subgoals whose result is already in inventory (quantity ≥ 1).
        """
        if goal_id in visited:
            return
        visited.add(goal_id)

        # Skip if the goal itself is already satisfied in inventory
        # (e.g., don't plan 'make wood_sword' if sword already held)
        if inventory is not None and inventory.get(goal_id, 0) >= 1:
            return

        # Find which concept produces this goal
        for concept in self.concepts.values():
            for link in concept.causal_links:
                if link.result != goal_id:
                    continue

                # For make/place: ensure the near-target concept exists
                # e.g. "make wood_pickaxe near table" → need table first
                if link.action in ("make", "place"):
                    self._plan_recursive(concept.id, steps, visited, inventory)

                # Recurse into prerequisites (inventory items)
                for req_item in link.requires:
                    self._plan_recursive(req_item, steps, visited, inventory)

                steps.append(
                    PlannedStep(
                        action=link.action,
                        target=concept.id,
                        near=concept.id if link.action in ("make", "place") else None,
                        expected_gain=link.result,
                        requires=dict(link.requires),
                    )
                )
                return  # found a way to produce goal_id

    # --- Verification ---

    def verify(
        self, concept_id: str, action: str, actual_outcome: str | None
    ) -> None:
        """Update confidence based on prediction vs actual outcome."""
        concept = self.query_text(concept_id)
        if concept is None:
            return

        for link in concept.causal_links:
            if link.action != action:
                continue
            if actual_outcome == link.result:
                link.confidence = min(1.0, link.confidence + CONFIRM_DELTA)
            else:
                link.confidence = max(0.0, link.confidence - REFUTE_DELTA)
            return

    def record_surprise(self, outcome: str, action: str, near: str | None = None) -> None:
        """Log an unexpected event (no matching rule)."""
        self.surprises.append(SurpriseEvent(outcome=outcome, action=action, near=near))

    # --- Prediction helper ---

    def predict_before_action(
        self, near: str, action: str, inventory: dict[str, int]
    ) -> Prediction | None:
        """Create a prediction for upcoming action."""
        concept = self.query_text(near)
        if concept is None:
            return None
        link = concept.find_causal(action=action, check_requires=inventory)
        if link is None:
            return None
        return Prediction(
            concept_id=near,
            action=action,
            expected=link.result,
            confidence=link.confidence,
            link=link,
        )

    def verify_after_action(
        self,
        prediction: Prediction | None,
        action: str,
        actual_outcome: str | None,
        near: str | None = None,
    ) -> None:
        """Verify a prediction against actual outcome, update confidence."""
        if prediction is None:
            if actual_outcome is not None:
                self.record_surprise(actual_outcome, action, near=near)
            return

        if prediction.link is not None:
            if actual_outcome == prediction.expected:
                prediction.link.confidence = min(
                    1.0, prediction.link.confidence + CONFIRM_DELTA
                )
            else:
                prediction.link.confidence = max(
                    0.0, prediction.link.confidence - REFUTE_DELTA
                )

    # --- Persistence ---

    def save(self, path: str) -> None:
        """Save concept store to directory (JSON metadata + tensors)."""
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)

        meta = {}
        tensors = {}
        for cid, concept in self.concepts.items():
            meta[cid] = {
                "attributes": concept.attributes,
                "confidence": concept.confidence,
                "causal_links": [
                    {
                        "action": l.action,
                        "result": l.result,
                        "requires": l.requires,
                        "condition": l.condition,
                        "confidence": l.confidence,
                    }
                    for l in concept.causal_links
                ],
            }
            if concept.visual is not None:
                tensors[f"{cid}_visual"] = concept.visual
            if concept.text_sdr is not None:
                tensors[f"{cid}_text_sdr"] = concept.text_sdr
            for i, obs in enumerate(concept.observations):
                tensors[f"{cid}_obs_{i}"] = obs

        (p / "concepts.json").write_text(json.dumps(meta, indent=2))
        if tensors:
            torch.save(tensors, p / "tensors.pt")

    def load(self, path: str) -> None:
        """Load concept store from directory."""
        p = Path(path)
        meta = json.loads((p / "concepts.json").read_text())
        tensors_path = p / "tensors.pt"
        tensors = torch.load(tensors_path, weights_only=True) if tensors_path.exists() else {}

        for cid, info in meta.items():
            concept = self.register(cid, info["attributes"])
            concept.confidence = info["confidence"]
            for ldata in info["causal_links"]:
                concept.causal_links.append(
                    CausalLink(
                        action=ldata["action"],
                        result=ldata["result"],
                        requires=ldata.get("requires", {}),
                        condition=ldata.get("condition"),
                        confidence=ldata.get("confidence", 0.5),
                    )
                )
            vis_key = f"{cid}_visual"
            if vis_key in tensors:
                concept.visual = tensors[vis_key]
            sdr_key = f"{cid}_text_sdr"
            if sdr_key in tensors:
                concept.text_sdr = tensors[sdr_key]
            # Load observations
            for i in range(MAX_OBSERVATIONS):
                obs_key = f"{cid}_obs_{i}"
                if obs_key in tensors:
                    concept.observations.append(tensors[obs_key])
                else:
                    break
