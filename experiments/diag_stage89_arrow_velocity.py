"""Diagnostic: measure how often live arrow observations have usable velocity.

Run on minipc:
  ./scripts/minipc-run.sh stage89vdiag "from diag_stage89_arrow_velocity import main; main()"

Writes:
  _docs/diag_stage89_arrow_velocity.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

torch.backends.cudnn.enabled = False


def main() -> None:
    from snks.agent.crafter_pixel_env import CrafterPixelEnv
    from snks.agent.crafter_textbook import CrafterTextbook
    from snks.agent.perception import HomeostaticTracker
    from snks.agent.post_mortem import PostMortemLearner
    from snks.agent.vector_bootstrap import load_from_textbook
    from snks.agent.vector_mpc_agent import run_vector_mpc_episode
    from snks.agent.vector_world_model import VectorWorldModel
    from snks.encoder.tile_segmenter import load_tile_segmenter, pick_device

    root = Path(__file__).parent.parent
    out_path = root / "_docs" / "diag_stage89_arrow_velocity.json"
    textbook_path = root / "configs" / "crafter_textbook.yaml"
    checkpoint_path = root / "demos" / "checkpoints" / "exp136" / "segmenter_9x9.pt"

    device = torch.device(pick_device())
    model = VectorWorldModel(dim=16384, n_locations=50000, seed=42, device=device)
    load_from_textbook(model, textbook_path)
    segmenter = load_tile_segmenter(str(checkpoint_path), device=device)
    textbook = CrafterTextbook(str(textbook_path))
    stimuli = PostMortemLearner().build_stimuli(["health", "food", "drink", "energy"])

    episodes: list[dict] = []
    n_episodes = 5
    for ep in range(n_episodes):
        env = CrafterPixelEnv(seed=42 + ep)
        tracker = HomeostaticTracker()
        tracker.init_from_textbook(textbook.body_block)

        metrics = run_vector_mpc_episode(
            env=env,
            segmenter=segmenter,
            model=model,
            tracker=tracker,
            max_steps=1000,
            stimuli=stimuli,
            textbook=textbook,
            verbose=False,
        )
        row = {
            "episode": ep,
            "episode_steps": metrics.get("episode_steps", 0),
            "death_cause": metrics.get("death_cause", "alive"),
            "arrow_visible_steps": metrics.get("arrow_visible_steps", 0),
            "arrow_velocity_known_steps": metrics.get("arrow_velocity_known_steps", 0),
            "arrow_velocity_unknown_steps": metrics.get("arrow_velocity_unknown_steps", 0),
            "arrow_velocity_known_rate": metrics.get("arrow_velocity_known_rate", 0.0),
            "arrow_threat_steps": metrics.get("arrow_threat_steps", 0),
            "defensive_action_rate": metrics.get("defensive_action_rate", 0.0),
            "danger_prediction_error": metrics.get("danger_prediction_error", 0.0),
        }
        episodes.append(row)
        print(row)

    summary = {
        "n_episodes": n_episodes,
        "avg_arrow_visible_steps": round(float(np.mean([e["arrow_visible_steps"] for e in episodes])), 2),
        "avg_arrow_velocity_known_rate": round(
            float(np.mean([e["arrow_velocity_known_rate"] for e in episodes])), 3
        ),
        "total_arrow_visible_steps": int(sum(e["arrow_visible_steps"] for e in episodes)),
        "total_arrow_velocity_known_steps": int(
            sum(e["arrow_velocity_known_steps"] for e in episodes)
        ),
        "total_arrow_velocity_unknown_steps": int(
            sum(e["arrow_velocity_unknown_steps"] for e in episodes)
        ),
    }
    print(summary)

    out_path.write_text(json.dumps({"summary": summary, "episodes": episodes}, indent=2))
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
