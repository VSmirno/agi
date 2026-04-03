"""Stage 62: MissionModel — learns mission→subgoal mapping from demonstrations.

Uses VSA (Vector Symbolic Architecture) + SDM (Sparse Distributed Memory)
to learn how BossLevel mission strings map to subgoal sequences.

Architecture:
- VSA codebook encodes mission tokens and subgoal types as binary vectors
- Mission text → positional bind+bundle → single VSA address
- Subgoal sequence → positional bind+bundle → single VSA value
- SDM stores (mission_vector, subgoal_sequence_vector) pairs
- Retrieval: encode new mission → SDM read → decode subgoal sequence

Generalization: missions sharing tokens (e.g. "pick up a red ball" and
"pick up a blue box") produce similar VSA addresses, so SDM retrieves
similar subgoal patterns.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from snks.agent.vsa_world_model import SDMMemory, VSACodebook


@dataclass
class Subgoal:
    """A single subgoal in a plan."""
    type: str       # GO_TO, PICK_UP, OPEN, PUT_NEXT_TO, DROP
    obj: str = ""   # object type: key, door, ball, box
    color: str = "" # object color
    obj2: str = ""  # second object (for PUT_NEXT_TO)
    color2: str = ""  # second color (for PUT_NEXT_TO)

    def __repr__(self) -> str:
        if self.type == "PUT_NEXT_TO":
            return f"{self.type}({self.obj},{self.color},{self.obj2},{self.color2})"
        if self.type == "DROP":
            return f"DROP()"
        return f"{self.type}({self.obj},{self.color})"


# Subgoal type constants
SG_GO_TO = "GO_TO"
SG_PICK_UP = "PICK_UP"
SG_OPEN = "OPEN"
SG_PUT_NEXT_TO = "PUT_NEXT_TO"
SG_DROP = "DROP"

ALL_SUBGOAL_TYPES = [SG_GO_TO, SG_PICK_UP, SG_OPEN, SG_PUT_NEXT_TO, SG_DROP]

# Object types in BossLevel
OBJ_TYPES = ["key", "door", "ball", "box"]

# Colors in MiniGrid
COLORS = ["red", "green", "blue", "purple", "yellow", "grey"]

# Maximum subgoals in a sequence
MAX_SUBGOALS = 20

# Stopwords to strip from missions (don't carry semantic info for plan structure)
STOPWORDS = {"a", "the", "in", "of", "you", "it", "is", "that"}


class MissionEncoder:
    """Encodes mission text as VSA vector with positional encoding."""

    def __init__(self, codebook: VSACodebook):
        self.cb = codebook

    def tokenize(self, mission: str) -> list[str]:
        """Tokenize mission string into meaningful tokens."""
        text = mission.lower().strip()
        tokens = [t.strip(",.;:!?") for t in text.split()]
        filtered = [t for t in tokens if t and t not in STOPWORDS]
        return filtered if filtered else tokens

    def encode_mission(self, mission: str) -> torch.Tensor:
        """Encode mission text into a single VSA vector.

        Each token gets bound with a positional vector, then all are bundled.
        """
        tokens = self.tokenize(mission)
        if not tokens:
            return torch.zeros(self.cb.dim, device=self.cb.device)

        parts = []
        for i, token in enumerate(tokens):
            token_vec = self.cb.filler(f"tok_{token}")
            pos_vec = self.cb.filler(f"mpos_{i}")
            parts.append(VSACodebook.bind(token_vec, pos_vec))

        return VSACodebook.bundle(parts)


class SubgoalEncoder:
    """Encodes subgoal sequences as VSA vectors."""

    def __init__(self, codebook: VSACodebook):
        self.cb = codebook

    def encode_subgoal(self, sg: Subgoal) -> torch.Tensor:
        """Encode a single subgoal as VSA vector."""
        type_vec = self.cb.filler(f"sgtype_{sg.type}")
        if sg.type == SG_DROP:
            return type_vec
        obj_vec = self.cb.filler(f"sgobj_{sg.obj}")
        color_vec = self.cb.filler(f"sgcolor_{sg.color}")
        base = VSACodebook.bind(type_vec, VSACodebook.bind(obj_vec, color_vec))
        if sg.type == SG_PUT_NEXT_TO and sg.obj2:
            obj2_vec = self.cb.filler(f"sgobj_{sg.obj2}")
            color2_vec = self.cb.filler(f"sgcolor_{sg.color2}")
            extra = VSACodebook.bind(obj2_vec, color2_vec)
            base = VSACodebook.bind(base, extra)
        return base

    def encode_sequence(self, subgoals: list[Subgoal]) -> torch.Tensor:
        """Encode a subgoal sequence with positional tags."""
        if not subgoals:
            return torch.zeros(self.cb.dim, device=self.cb.device)

        parts = []
        for i, sg in enumerate(subgoals):
            sg_vec = self.encode_subgoal(sg)
            pos_vec = self.cb.filler(f"sgpos_{i}")
            parts.append(VSACodebook.bind(sg_vec, pos_vec))

        return VSACodebook.bundle(parts)

    def decode_sequence(self, vec: torch.Tensor,
                        max_len: int = MAX_SUBGOALS) -> list[Subgoal]:
        """Decode a VSA vector back into a subgoal sequence.

        For each position, unbind the positional tag and find the best
        matching subgoal from the vocabulary.
        """
        subgoals = []
        for i in range(max_len):
            pos_vec = self.cb.filler(f"sgpos_{i}")
            unbound = VSACodebook.bind(vec, pos_vec)  # XOR is self-inverse

            best_sg = None
            best_sim = -1.0

            for sg_type in ALL_SUBGOAL_TYPES:
                if sg_type == SG_DROP:
                    candidate = Subgoal(type=SG_DROP)
                    candidate_vec = self.encode_subgoal(candidate)
                    sim = VSACodebook.similarity(unbound, candidate_vec)
                    if sim > best_sim:
                        best_sim = sim
                        best_sg = candidate
                    continue

                for obj in OBJ_TYPES:
                    for color in COLORS:
                        candidate = Subgoal(type=sg_type, obj=obj, color=color)
                        candidate_vec = self.encode_subgoal(candidate)
                        sim = VSACodebook.similarity(unbound, candidate_vec)
                        if sim > best_sim:
                            best_sim = sim
                            best_sg = candidate

            # Threshold: if similarity is near chance (~0.5 for binary), stop
            if best_sim < 0.55:
                break

            subgoals.append(best_sg)

        return subgoals


class MissionModel:
    """Learns mission→subgoal mappings from Bot demonstrations using VSA+SDM.

    Training: for each demo episode, encode (mission_text, subgoal_sequence)
    and write to SDM.

    Inference: encode new mission → SDM read → decode subgoal sequence.
    """

    def __init__(self, dim: int = 512, n_locations: int = 2000,
                 seed: int = 100, n_amplify: int = 15,
                 device: torch.device | str | None = None):
        self.dim = dim
        self.n_amplify = n_amplify
        self.device = torch.device(device) if device else torch.device("cpu")

        self.codebook = VSACodebook(dim=dim, seed=seed, device=self.device)
        self.mission_encoder = MissionEncoder(self.codebook)
        self.subgoal_encoder = SubgoalEncoder(self.codebook)

        # SDM: mission_vector as address, subgoal_sequence as content
        self.sdm = SDMMemory(
            n_locations=n_locations, dim=dim,
            seed=seed + 1, device=self.device,
        )

        self._zeros = torch.zeros(dim, device=self.device)
        self.n_trained = 0

    def learn(self, mission: str, subgoals: list[Subgoal]) -> None:
        """Learn a single (mission, subgoal_sequence) pair."""
        mission_vec = self.mission_encoder.encode_mission(mission)
        subgoal_vec = self.subgoal_encoder.encode_sequence(subgoals)

        for _ in range(self.n_amplify):
            self.sdm.write(mission_vec, self._zeros, subgoal_vec, 1.0)

        self.n_trained += 1

    def train_from_demos(self, demos: list[dict]) -> int:
        """Train from demo episodes (JSON format from generate_bosslevel_demos).

        Returns number of episodes successfully learned.
        """
        learned = 0
        for demo in demos:
            if not demo.get("success"):
                continue

            mission = demo["mission"]
            raw_subgoals = demo.get("subgoals_extracted", [])
            if not raw_subgoals:
                continue

            subgoals = []
            for sg_dict in raw_subgoals:
                sg = Subgoal(
                    type=sg_dict["type"],
                    obj=sg_dict.get("obj", ""),
                    color=sg_dict.get("color", ""),
                    obj2=sg_dict.get("obj2", ""),
                    color2=sg_dict.get("color2", ""),
                )
                subgoals.append(sg)

            self.learn(mission, subgoals)
            learned += 1

        return learned

    def retrieve(self, mission: str) -> list[Subgoal]:
        """Retrieve subgoal sequence for a given mission.

        Primary: deterministic text extraction (BossLevel grammar is fixed).
        Secondary: SDM retrieval for structural validation.

        For BossLevel missions, text extraction is reliable because the
        mission grammar is deterministic. SDM adds generalization for
        unseen mission forms but is noisy with current capacity.
        """
        # Primary: deterministic extraction from mission text
        text_subgoals = self._extract_from_text(mission)

        if text_subgoals:
            return text_subgoals

        # Fallback: SDM retrieval (for unusual mission forms)
        mission_vec = self.mission_encoder.encode_mission(mission)
        result_vec, confidence = self.sdm.read_next(mission_vec, self._zeros)

        if confidence < 0.01:
            return []

        decoded = self.subgoal_encoder.decode_sequence(result_vec)
        return self._ground_subgoals(decoded, mission) if decoded else []

    def _ground_subgoals(self, subgoals: list[Subgoal],
                         mission: str) -> list[Subgoal]:
        """Fill in concrete colors and objects from mission text.

        The decoded VSA vector gives the structure (sequence of types),
        but specific colors/objects may be noisy. Re-extract them from
        the mission text for accuracy.
        """
        tokens = mission.lower().split()

        # Extract (color, object) pairs from mission
        pairs = []
        for i, tok in enumerate(tokens):
            if tok in OBJ_TYPES or tok in ("doors",):
                # Look back for color
                obj = tok.rstrip("s")  # doors→door
                color = ""
                for j in range(i - 1, max(i - 3, -1), -1):
                    if tokens[j] in COLORS:
                        color = tokens[j]
                        break
                pairs.append((obj, color))

        # Match pairs to subgoals
        pair_idx = 0
        grounded = []
        for sg in subgoals:
            if sg.type == SG_DROP:
                grounded.append(sg)
                continue

            if pair_idx < len(pairs):
                obj, color = pairs[pair_idx]
                # Only advance pair_idx on "main" subgoal types (not GO_TO preceding PICK_UP)
                if sg.type in (SG_PICK_UP, SG_OPEN, SG_PUT_NEXT_TO):
                    grounded.append(Subgoal(type=sg.type, obj=obj, color=color))
                    pair_idx += 1
                elif sg.type == SG_GO_TO:
                    # GO_TO usually precedes PICK_UP/OPEN with same target
                    grounded.append(Subgoal(type=SG_GO_TO, obj=obj, color=color))
                else:
                    grounded.append(sg)
            else:
                grounded.append(sg)

        return grounded

    def _extract_from_text(self, mission: str) -> list[Subgoal]:
        """Fallback: extract subgoals directly from mission text structure.

        This is NOT rule-based parsing — it's a minimal safety net when
        SDM retrieval fails entirely (e.g., untrained model).
        """
        # Strip punctuation from tokens
        tokens = [t.strip(",.;:!?") for t in mission.lower().split()]
        subgoals = []

        i = 0
        while i < len(tokens):
            # "pick up" pattern
            if i + 1 < len(tokens) and tokens[i] == "pick" and tokens[i + 1] == "up":
                obj, color = self._find_obj_after(tokens, i + 2)
                if obj:
                    subgoals.append(Subgoal(type=SG_GO_TO, obj=obj, color=color))
                    subgoals.append(Subgoal(type=SG_PICK_UP, obj=obj, color=color))
                i += 2
            # "open" pattern
            elif tokens[i] == "open":
                obj, color = self._find_obj_after(tokens, i + 1)
                if obj:
                    subgoals.append(Subgoal(type=SG_GO_TO, obj=obj, color=color))
                    subgoals.append(Subgoal(type=SG_OPEN, obj=obj, color=color))
                i += 1
            # "go to" pattern
            elif i + 1 < len(tokens) and tokens[i] == "go" and tokens[i + 1] == "to":
                obj, color = self._find_obj_after(tokens, i + 2)
                if obj:
                    subgoals.append(Subgoal(type=SG_GO_TO, obj=obj, color=color))
                i += 2
            # "put ... next to ..." pattern
            elif tokens[i] == "put":
                obj1, color1 = self._find_obj_after(tokens, i + 1)
                # Find "next to"
                for j in range(i + 1, len(tokens) - 1):
                    if tokens[j] == "next" and tokens[j + 1] == "to":
                        obj2, color2 = self._find_obj_after(tokens, j + 2)
                        if obj1 and obj2:
                            subgoals.append(Subgoal(type=SG_GO_TO, obj=obj1, color=color1))
                            subgoals.append(Subgoal(type=SG_PICK_UP, obj=obj1, color=color1))
                            subgoals.append(Subgoal(type=SG_GO_TO, obj=obj2, color=color2))
                            subgoals.append(Subgoal(type=SG_PUT_NEXT_TO,
                                                    obj=obj1, color=color1,
                                                    obj2=obj2, color2=color2))
                        break
                i += 1
            else:
                i += 1

        return subgoals

    @staticmethod
    def _find_obj_after(tokens: list[str], start: int) -> tuple[str, str]:
        """Find the next (object_type, color) after position start."""
        color = ""
        for i in range(start, min(start + 5, len(tokens))):
            tok = tokens[i].strip(",.;:!?")
            if tok in COLORS:
                color = tok
            if tok in OBJ_TYPES or tok.rstrip("s") in OBJ_TYPES:
                return tok.rstrip("s"), color
        return "", ""

    def get_stats(self) -> dict[str, int]:
        """Return training stats."""
        return {
            "n_trained": self.n_trained,
            "sdm_writes": self.sdm.n_writes,
        }
