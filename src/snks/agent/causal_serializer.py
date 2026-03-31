"""CausalModelSerializer: save/load CausalWorldModel to JSON (Stage 26)."""

from __future__ import annotations

import json
from pathlib import Path

from snks.agent.causal_model import CausalWorldModel, _TransitionRecord, _context_hash, _split_context
from snks.daf.types import CausalAgentConfig


class CausalModelSerializer:
    """Serialize/deserialize CausalWorldModel for cross-environment transfer."""

    VERSION = 1

    @staticmethod
    def save(model: CausalWorldModel, path: str, source_env: str = "") -> None:
        """Serialize full internal state to JSON.

        Saves _TransitionRecord entries with original context_sks/effect_sks
        frozensets (as sorted lists). Hashes are recomputed on load.
        """
        transitions = []
        for (ctx_hash, action), records in model._transitions.items():
            for eff_hash, record in records.items():
                transitions.append({
                    "action": action,
                    "context_sks": sorted(record.context_sks),
                    "effect_sks": sorted(record.effect_sks),
                    "count": record.count,
                    "total_in_context": record.total_in_context,
                })

        data = {
            "version": CausalModelSerializer.VERSION,
            "config": {
                "causal_min_observations": model.config.causal_min_observations,
                "causal_confidence_threshold": model.config.causal_confidence_threshold,
                "causal_decay": model.config.causal_decay,
                "causal_context_bins": model.config.causal_context_bins,
            },
            "total_observations": model._total_observations,
            "transitions": transitions,
            "source_env": source_env,
        }

        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    @staticmethod
    def load(
        path: str,
        config: CausalAgentConfig | None = None,
    ) -> CausalWorldModel:
        """Deserialize causal model from JSON.

        Rebuilds _transitions dict by recomputing hashes from stored
        context_sks/effect_sks. Raises ValueError on version mismatch.
        """
        data = json.loads(Path(path).read_text(encoding="utf-8"))

        version = data.get("version", 0)
        if version != CausalModelSerializer.VERSION:
            raise ValueError(
                f"Causal model version mismatch: expected {CausalModelSerializer.VERSION}, "
                f"got {version}"
            )

        # Build config from saved data or use override
        if config is None:
            saved_cfg = data.get("config", {})
            config = CausalAgentConfig(
                causal_min_observations=saved_cfg.get("causal_min_observations", 3),
                causal_confidence_threshold=saved_cfg.get("causal_confidence_threshold", 0.5),
                causal_decay=saved_cfg.get("causal_decay", 0.99),
                causal_context_bins=saved_cfg.get("causal_context_bins", 64),
            )

        model = CausalWorldModel(config)
        model._total_observations = data.get("total_observations", 0)

        # Rebuild _transitions by recomputing hashes
        for entry in data.get("transitions", []):
            ctx_sks = frozenset(entry["context_sks"])
            eff_sks = frozenset(entry["effect_sks"])
            action = entry["action"]

            ctx_hash = _context_hash(ctx_sks)
            eff_hash = _context_hash(eff_sks)
            key = (ctx_hash, action)

            model._transitions[key][eff_hash] = _TransitionRecord(
                context_sks=ctx_sks,
                effect_sks=eff_sks,
                count=entry["count"],
                total_in_context=entry["total_in_context"],
            )

        return model
