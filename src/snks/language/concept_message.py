"""ConceptMessage: structured concept exchange between agents (Stage 33).

Agents communicate through concept-level messages — HAC embeddings,
causal links, and skills — NOT through natural language text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import torch
from torch import Tensor

from snks.agent.causal_model import CausalLink
from snks.language.skill import Skill


class MessageType(Enum):
    """Types of concept messages."""
    CAUSAL_LINKS = "causal_links"   # share discovered causal knowledge
    SKILL = "skill"                  # share a learned skill
    WARNING = "warning"              # signal danger / negative outcome
    REQUEST = "request"              # ask for knowledge about a concept


@dataclass
class ConceptMessage:
    """A concept-level message between agents.

    All content is expressed in concept space (SKS IDs, HAC vectors,
    CausalLinks, Skills) — never in natural language text.
    """
    sender_id: str
    receiver_id: str | None                     # None = broadcast
    content_type: MessageType
    sks_context: frozenset[int] = field(default_factory=frozenset)
    hac_embedding: Tensor | None = None         # 2048-dim HAC vector
    causal_links: list[CausalLink] = field(default_factory=list)
    skill: Skill | None = None
    urgency: float = 0.0                        # 0.0 (info) .. 1.0 (critical)
    timestamp: int = 0                          # logical tick

    def __post_init__(self) -> None:
        if self.urgency < 0.0 or self.urgency > 1.0:
            raise ValueError(f"urgency must be in [0, 1], got {self.urgency}")

    @property
    def is_broadcast(self) -> bool:
        return self.receiver_id is None

    def content_summary(self) -> str:
        """Machine-readable summary (for logging, not for agent consumption)."""
        parts = [f"type={self.content_type.value}"]
        if self.causal_links:
            parts.append(f"links={len(self.causal_links)}")
        if self.skill is not None:
            parts.append(f"skill={self.skill.name}")
        if self.sks_context:
            parts.append(f"ctx_sks={len(self.sks_context)}")
        return ", ".join(parts)
