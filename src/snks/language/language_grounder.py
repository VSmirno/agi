"""Stage 50: LanguageGrounder — maps natural language instructions to VSA vectors and subgoals.

Bridges the language pipeline (Stages 7-24) with the VSA planning pipeline (Stages 45-46).
Uses RuleBasedChunker for parsing and VSACodebook for encoding in a unified vector space.
"""

from __future__ import annotations

import torch

from snks.agent.vsa_world_model import VSACodebook
from snks.language.chunker import Chunk, RuleBasedChunker

# --- Word → VSA filler mappings ---

ACTION_TO_VSA: dict[str, str] = {
    "pick up": "action_pickup",
    "go to": "action_goto",
    "open": "action_open",
    "toggle": "action_toggle",
    "put": "action_put",
    "drop": "action_drop",
}

OBJECT_TO_VSA: dict[str, str] = {
    "key": "object_key",
    "door": "object_door",
    "goal": "object_goal",
    "ball": "object_ball",
    "box": "object_box",
}

ATTR_TO_VSA: dict[str, str] = {
    "red": "color_red",
    "green": "color_green",
    "blue": "color_blue",
    "purple": "color_purple",
    "yellow": "color_yellow",
    "grey": "color_grey",
    "gray": "color_grey",
}

# (action_vsa_name, object_vsa_name) → subgoal name
SUBGOAL_MAP: dict[tuple[str, str], str] = {
    ("action_pickup", "object_key"): "pickup_key",
    ("action_pickup", "object_ball"): "pickup_ball",
    ("action_pickup", "object_box"): "pickup_box",
    ("action_open", "object_door"): "open_door",
    ("action_toggle", "object_door"): "open_door",
    ("action_goto", "object_goal"): "reach_goal",
    ("action_goto", "object_door"): "goto_door",
    ("action_goto", "object_key"): "goto_key",
    ("action_drop", "object_key"): "drop_key",
    ("action_drop", "object_ball"): "drop_ball",
    ("action_put", "object_box"): "put_box",
    ("action_put", "object_ball"): "put_ball",
}


class LanguageGrounder:
    """Maps natural language instructions to VSA vectors and subgoal sequences.

    Encodes instructions into VSACodebook's binary vector space (512-dim),
    same space used by VSAEncoder for world model states. This enables
    direct comparison between language intent and world state.
    """

    # VSA roles for instruction encoding
    ROLE_ACTION = "instr_action"
    ROLE_OBJECT = "instr_object"
    ROLE_ATTR = "instr_attr"

    def __init__(self, codebook: VSACodebook) -> None:
        self.cb = codebook
        self.chunker = RuleBasedChunker()

    def encode(self, instruction: str) -> torch.Tensor:
        """Encode a single instruction into a VSA vector (dim,) binary.

        For sequential instructions ("X then Y"), encodes only the first part.
        Use encode_sequence() for full sequential encoding.
        """
        chunks = self.chunker.chunk(instruction)
        # Take only chunks before first SEQ_BREAK
        first_part = []
        for c in chunks:
            if c.role == "SEQ_BREAK":
                break
            first_part.append(c)
        return self._encode_chunks(first_part)

    def encode_sequence(self, instruction: str) -> list[torch.Tensor]:
        """Encode a (possibly sequential) instruction into a list of VSA vectors.

        "pick up the key then open the door" → [vsa_pickup_key, vsa_open_door]
        """
        chunks = self.chunker.chunk(instruction)
        segments = self._split_by_seq_break(chunks)
        return [self._encode_chunks(seg) for seg in segments]

    def decode(self, vsa_vector: torch.Tensor) -> dict[str, str]:
        """Decode a VSA vector back to action/object/attr names.

        Returns dict with keys: "action", "object", and optionally "attr".
        Unbinds each role and finds closest filler by similarity.
        """
        result: dict[str, str] = {}

        # Decode action
        action_candidate = self.cb.bind(self.cb.role(self.ROLE_ACTION), vsa_vector)
        best_action, best_sim = self._find_closest_filler(action_candidate, ACTION_TO_VSA)
        if best_action and best_sim >= 0.55:
            result["action"] = best_action

        # Decode object
        object_candidate = self.cb.bind(self.cb.role(self.ROLE_OBJECT), vsa_vector)
        best_object, best_sim = self._find_closest_filler(object_candidate, OBJECT_TO_VSA)
        if best_object and best_sim >= 0.55:
            result["object"] = best_object

        # Decode attr (optional)
        attr_candidate = self.cb.bind(self.cb.role(self.ROLE_ATTR), vsa_vector)
        best_attr, best_sim = self._find_closest_filler(attr_candidate, ATTR_TO_VSA)
        if best_attr and best_sim >= 0.55:
            result["attr"] = best_attr

        return result

    def to_subgoals(self, instruction: str) -> list[str]:
        """Convert instruction to ordered list of subgoal names.

        "pick up the key then open the door" → ["pickup_key", "open_door"]
        Returns empty list if instruction cannot be mapped.
        """
        chunks = self.chunker.chunk(instruction)
        segments = self._split_by_seq_break(chunks)
        subgoals: list[str] = []

        for seg in segments:
            action_vsa = None
            object_vsa = None
            for c in seg:
                if c.role == "ACTION":
                    action_vsa = self._resolve_action(c.text)
                elif c.role == "OBJECT":
                    object_vsa = self._resolve_object(c.text)

            if action_vsa and object_vsa:
                sg = SUBGOAL_MAP.get((action_vsa, object_vsa))
                if sg:
                    subgoals.append(sg)

        return subgoals

    # --- Internal methods ---

    def _encode_chunks(self, chunks: list[Chunk]) -> torch.Tensor:
        """Encode a list of chunks (single instruction) into a VSA vector."""
        bindings: list[torch.Tensor] = []

        for c in chunks:
            if c.role == "ACTION":
                filler_name = self._resolve_action(c.text)
                if filler_name:
                    bindings.append(self.cb.bind(
                        self.cb.role(self.ROLE_ACTION),
                        self.cb.filler(filler_name),
                    ))
            elif c.role == "OBJECT":
                filler_name = self._resolve_object(c.text)
                if filler_name:
                    bindings.append(self.cb.bind(
                        self.cb.role(self.ROLE_OBJECT),
                        self.cb.filler(filler_name),
                    ))
            elif c.role == "ATTR":
                filler_name = self._resolve_attr(c.text)
                if filler_name:
                    bindings.append(self.cb.bind(
                        self.cb.role(self.ROLE_ATTR),
                        self.cb.filler(filler_name),
                    ))

        if not bindings:
            return torch.zeros(self.cb.dim, dtype=torch.float32)

        return self.cb.bundle(bindings)

    def _find_closest_filler(
        self, candidate: torch.Tensor, word_map: dict[str, str]
    ) -> tuple[str | None, float]:
        """Find the VSA filler name most similar to candidate."""
        best_name: str | None = None
        best_sim = -1.0
        for _word, vsa_name in word_map.items():
            filler_vec = self.cb.filler(vsa_name)
            sim = self.cb.similarity(candidate, filler_vec)
            if sim > best_sim:
                best_sim = sim
                best_name = vsa_name
        return best_name, best_sim

    def _resolve_action(self, text: str) -> str | None:
        return ACTION_TO_VSA.get(text.lower().strip())

    def _resolve_object(self, text: str) -> str | None:
        return OBJECT_TO_VSA.get(text.lower().strip())

    def _resolve_attr(self, text: str) -> str | None:
        return ATTR_TO_VSA.get(text.lower().strip())

    @staticmethod
    def _split_by_seq_break(chunks: list[Chunk]) -> list[list[Chunk]]:
        """Split chunks into segments separated by SEQ_BREAK."""
        segments: list[list[Chunk]] = [[]]
        for c in chunks:
            if c.role == "SEQ_BREAK":
                segments.append([])
            else:
                segments[-1].append(c)
        return [s for s in segments if s]
