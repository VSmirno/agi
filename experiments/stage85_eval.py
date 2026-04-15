"""Stage 85: Goal Selector Design — eval gate on Crafter.

Run on minipc ONLY:
  ssh minipc "cd /opt/agi && git pull origin main"
  ssh minipc "tmux new-session -d -s stage85 'cd /opt/agi && source venv/bin/activate && \
    HSA_OVERRIDE_GFX_VERSION=11.0.0 PYTHONPATH=src python experiments/stage85_eval.py 2>&1 | \
    tee _docs/stage85_eval.log'"

Gates:
  1. survival_ge_155: mean survival ≥ 155 (regression floor vs Stage 82 baseline)
  2. wood_ge_10pct:   wood ≥ 3 in ≥ 10% of episodes (goal-directed planning signal)
  3. no_total_gain:   score_trajectory has no total_gain reference (verified statically)

Results saved to _docs/stage85_eval.json.
"""

from __future__ import annotations

import ast
import inspect
import json
import time
from pathlib import Path

import numpy as np
import torch

torch.backends.cudnn.enabled = False


def check_no_total_gain() -> bool:
    """Verify score_trajectory source has no total_gain reference."""
    from snks.agent.vector_sim import score_trajectory
    source = inspect.getsource(score_trajectory)
    return "total_gain" not in source


def run_eval(
    n_episodes: int = 20,
    max_steps: int = 1000,
    model_dim: int = 16384,
    n_locations: int = 50000,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    from snks.agent.vector_world_model import VectorWorldModel
    from snks.agent.vector_bootstrap import load_from_textbook
    from snks.agent.vector_mpc_agent import run_vector_mpc_episode
    from snks.agent.stimuli import StimuliLayer, SurvivalAversion, HomeostasisStimulus
    from snks.agent.perception import HomeostaticTracker
    from snks.agent.crafter_textbook import CrafterTextbook
    from snks.agent.crafter_pixel_env import CrafterPixelEnv
    from snks.encoder.tile_segmenter import load_tile_segmenter, pick_device

    device = torch.device(pick_device())
    print(f"device={device}, dim={model_dim}, locs={n_locations}")

    checkpoint_path = Path(__file__).parent.parent / "demos" / "checkpoints" / "exp135" / "segmenter_9x9.pt"
    segmenter = load_tile_segmenter(str(checkpoint_path), device=device)
    print(f"Segmenter: {checkpoint_path}")

    model = VectorWorldModel(dim=model_dim, n_locations=n_locations, seed=seed, device=device)
    textbook_path = Path(__file__).parent.parent / "configs" / "crafter_textbook.yaml"
    stats = load_from_textbook(model, textbook_path)
    print(f"Textbook seeded: {stats}")

    tb = CrafterTextbook(str(textbook_path))
    vitals = ["health", "food", "drink", "energy"]
    stimuli = StimuliLayer([SurvivalAversion(), HomeostasisStimulus(vitals)])

    print(f"  goals_block: {tb.goals_block}")

    # --- Static gate: no_total_gain ---
    no_total_gain_ok = check_no_total_gain()
    print(f"  no_total_gain check: {'✓' if no_total_gain_ok else '✗'}")

    # --- Main eval: 20 episodes standard Crafter ---
    print(f"\n=== Main eval ({n_episodes} episodes) ===")
    results = []
    t0 = time.time()

    for ep in range(n_episodes):
        ep_seed = seed + ep
        env = CrafterPixelEnv(seed=ep_seed)
        tracker = HomeostaticTracker()
        tracker.init_from_textbook(tb.body_block)

        metrics = run_vector_mpc_episode(
            env=env,
            segmenter=segmenter,
            model=model,
            tracker=tracker,
            max_steps=max_steps,
            stimuli=stimuli,
            textbook=tb,
            verbose=verbose,
        )
        results.append(metrics)
        wood = metrics.get("final_inv", {}).get("wood", 0)
        elapsed = time.time() - t0
        eta = elapsed / (ep + 1) * (n_episodes - ep - 1)
        print(
            f"  ep{ep:2d}: len={metrics.get('avg_len', 0):4.0f} "
            f"wood={wood:2d} "
            f"cause={metrics.get('cause', '?'):8s} "
            f"[{elapsed:.0f}s eta={eta:.0f}s]"
        )

    # --- Aggregate metrics ---
    avg_len = float(np.mean([r.get("avg_len", 0) for r in results]))
    wood_counts = [r.get("final_inv", {}).get("wood", 0) for r in results]
    wood_ge3_pct = float(np.mean([w >= 3 for w in wood_counts]))

    gates = {
        "survival_ge_155": avg_len >= 155.0,
        "wood_ge_10pct": wood_ge3_pct >= 0.10,
        "no_total_gain": no_total_gain_ok,
    }

    print(f"\n=== Results ===")
    print(f"  avg_survival:  {avg_len:.1f}   (gate: ≥155) {'✓' if gates['survival_ge_155'] else '✗'}")
    print(f"  wood_ge3_pct:  {100*wood_ge3_pct:.0f}%     (gate: ≥10%) {'✓' if gates['wood_ge_10pct'] else '✗'}")
    print(f"  no_total_gain: {'yes' if no_total_gain_ok else 'no'}       (gate: True)  {'✓' if gates['no_total_gain'] else '✗'}")
    print(f"  wood counts:   {wood_counts}")

    output = {
        "avg_survival": avg_len,
        "wood_ge3_pct": wood_ge3_pct,
        "wood_counts": wood_counts,
        "gates": gates,
        "gates_passed": sum(gates.values()),
        "gates_total": len(gates),
        "n_episodes": n_episodes,
        "episodes": results,
    }
    return output


def main():
    out = run_eval(n_episodes=20, verbose=True)
    out_path = Path(__file__).parent.parent / "_docs" / "stage85_eval.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")
    passed = out["gates_passed"]
    total = out["gates_total"]
    print(f"Gates: {passed}/{total} {'✓ PASS' if passed == total else '✗ FAIL'}")


if __name__ == "__main__":
    main()
