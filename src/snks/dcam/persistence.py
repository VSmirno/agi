"""Persistence — save/load DCAM state via safetensors + JSON.

Tensors (HAC vectors, LSH projections, episode contexts) → safetensors.
Metadata (graph, episode info, config) → JSON.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from safetensors.torch import save_file, load_file

if TYPE_CHECKING:
    from snks.dcam.world_model import DcamWorldModel


def save(model: DcamWorldModel, path: str) -> None:
    """Save DCAM world model to <path>.safetensors + <path>.json."""
    base = Path(path)

    # --- Tensors ---
    tensors: dict[str, torch.Tensor] = {}
    tensors["hac.scalar_base"] = model.hac._scalar_base.cpu()
    tensors["lsh.projections"] = model.lsh.projections.cpu()

    # Stack episode contexts
    episodes = model.buffer.get_all_episodes()
    if episodes:
        contexts = torch.stack([e.context_hac.cpu() for e in episodes])
        tensors["episodic.contexts"] = contexts

    # LSH stored vectors
    if model.lsh._vectors:
        lsh_ids = sorted(model.lsh._vectors.keys())
        lsh_vecs = torch.stack([model.lsh._vectors[i].cpu() for i in lsh_ids])
        tensors["lsh.vectors"] = lsh_vecs
    else:
        lsh_ids = []

    save_file(tensors, str(base) + ".safetensors")

    # --- JSON metadata ---
    meta: dict = {}

    # Episodes
    meta["episodes"] = [
        {
            "episode_id": e.episode_id,
            "active_nodes": {str(k): v for k, v in e.active_nodes.items()},
            "importance": e.importance,
            "timestamp": e.timestamp,
            "consolidated": e.consolidated,
        }
        for e in episodes
    ]

    # Graph layers
    graph_data: dict[str, list] = {}
    for layer_name in ("structural", "causal", "temporal", "modulatory"):
        graph_data[layer_name] = [
            {"src": s, "dst": d, "weight": w}
            for s, d, w in model.graph.get_all_edges(layer_name)
        ]
    meta["graph"] = graph_data

    # LSH tables (hash_code → value_ids)
    meta["lsh_tables"] = [
        {str(code): ids for code, ids in table.items()}
        for table in model.lsh._tables
    ]
    meta["lsh_id_to_hashes"] = {
        str(k): v for k, v in model.lsh._id_to_hashes.items()
    }
    meta["lsh_vector_ids"] = lsh_ids

    # Buffer state
    meta["buffer_next_id"] = model.buffer._next_id
    meta["buffer_step"] = model.buffer._step

    # Cycle count
    meta["cycle_count"] = model._cycle_count

    with open(str(base) + ".json", "w") as f:
        json.dump(meta, f)


def load(model: DcamWorldModel, path: str) -> None:
    """Load DCAM world model from <path>.safetensors + <path>.json."""
    base = Path(path)
    device = model.device

    # --- Tensors ---
    tensors = load_file(str(base) + ".safetensors", device=str(device))

    model.hac._scalar_base = tensors["hac.scalar_base"].to(device)
    model.lsh.projections = tensors["lsh.projections"].to(device)

    # --- JSON metadata ---
    with open(str(base) + ".json") as f:
        meta = json.load(f)

    # Restore episodes
    model.buffer._episodes.clear()
    episodes_meta = meta["episodes"]
    contexts = tensors.get("episodic.contexts")

    for i, em in enumerate(episodes_meta):
        from snks.dcam.episodic import Episode

        ctx = contexts[i].to(device) if contexts is not None else torch.zeros(
            model.hac.dim, device=device
        )
        ep = Episode(
            episode_id=em["episode_id"],
            active_nodes={int(k): v for k, v in em["active_nodes"].items()},
            importance=em["importance"],
            timestamp=em["timestamp"],
            consolidated=em.get("consolidated", False),
            context_hac=ctx,
        )
        model.buffer._episodes[ep.episode_id] = ep

    model.buffer._next_id = meta["buffer_next_id"]
    model.buffer._step = meta["buffer_step"]

    # Restore graph
    from snks.dcam.ssg import StructuredSparseGraph

    model.graph = StructuredSparseGraph()
    for layer_name, edges in meta["graph"].items():
        for edge in edges:
            model.graph.add_edge(
                edge["src"], edge["dst"], layer_name, edge["weight"]
            )

    # Restore LSH
    from collections import defaultdict

    # Restore projections (already done above)
    # Restore tables
    model.lsh._tables = [
        defaultdict(list, {int(code): ids for code, ids in table.items()})
        for table in meta["lsh_tables"]
    ]
    model.lsh._id_to_hashes = {
        int(k): v for k, v in meta["lsh_id_to_hashes"].items()
    }

    # Restore LSH vectors
    lsh_ids = meta.get("lsh_vector_ids", [])
    lsh_vecs = tensors.get("lsh.vectors")
    model.lsh._vectors = {}
    if lsh_vecs is not None:
        for i, vid in enumerate(lsh_ids):
            model.lsh._vectors[vid] = lsh_vecs[i].to(device)

    model._cycle_count = meta.get("cycle_count", 0)
