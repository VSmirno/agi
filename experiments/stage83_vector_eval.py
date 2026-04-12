"""Stage 83: VectorWorldModel eval on Crafter.

Run on minipc ONLY:
  ssh minipc "cd /opt/agi && git pull origin main"
  ssh minipc "tmux new-session -d -s stage83 'cd /opt/agi && source venv/bin/activate && \
    HSA_OVERRIDE_GFX_VERSION=11.0.0 PYTHONPATH=src python experiments/stage83_vector_eval.py 2>&1 | \
    tee _docs/stage83_eval.log'"

Three ablations:
  1. seed_only: textbook bootstrap, no prior experience
  2. seed_experience: textbook + loaded experience from gen1
  3. multi_gen: 3 generations, each inheriting from previous

Eval gate:
  - survival ≥ 155 (Stage 82 baseline)
  - wood ≥3 in ≥10% episodes
  - mean surprise on first entity encounter > 2x after 5+ encounters
  - gen2 warmup > gen1 warmup (knowledge flow)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch


def run_eval(
    n_episodes: int = 20,
    max_steps: int = 1000,
    warmup_steps: int = 500,
    warmup_no_enemies: bool = True,
    model_dim: int = 16384,  # 16K for speed, 65K for production
    n_locations: int = 3000,
    seed: int = 42,
    experience_path: str | None = None,
    save_experience_path: str | None = None,
    label: str = "eval",
    verbose: bool = True,
) -> dict:
    """Run evaluation episodes and return metrics."""
    # Lazy imports to fail fast if deps missing
    from snks.agent.vector_world_model import VectorWorldModel
    from snks.agent.vector_bootstrap import load_from_textbook
    from snks.agent.vector_mpc_agent import run_vector_mpc_episode
    from snks.agent.perception import HomeostaticTracker
    from snks.agent.crafter_textbook import CrafterTextbook
    from snks.encoder.tile_segmenter import load_tile_segmenter, pick_device

    device = torch.device(pick_device())
    print(f"[{label}] device={device}, dim={model_dim}, locs={n_locations}")

    # --- Load segmenter ---
    checkpoint_path = Path(__file__).parent.parent / "demos" / "checkpoints" / "exp135" / "segmenter_9x9.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Segmenter checkpoint not found at {checkpoint_path}. "
            "Run Stage 75 training first or copy checkpoint from minipc."
        )
    segmenter = load_tile_segmenter(str(checkpoint_path), device=device)
    print(f"[{label}] Segmenter loaded from {checkpoint_path}")

    # --- Build model ---
    model = VectorWorldModel(
        dim=model_dim, n_locations=n_locations, seed=seed, device=device,
    )
    textbook_path = Path(__file__).parent.parent / "configs" / "crafter_textbook.yaml"
    stats = load_from_textbook(model, textbook_path)
    print(f"[{label}] Textbook seeded: {stats}")

    # --- Load experience if provided ---
    if experience_path and Path(experience_path).exists():
        loaded = model.load(experience_path)
        print(f"[{label}] Experience loaded: {loaded}")

    # --- Init tracker ---
    tb = CrafterTextbook(str(textbook_path))
    tracker = HomeostaticTracker()
    tracker.init_from_textbook(tb)

    # --- Import Crafter env ---
    try:
        from snks.agent.crafter_pixel_env import CrafterPixelEnv
    except ImportError:
        print(f"[{label}] CrafterPixelEnv not available, skipping")
        return {"error": "CrafterPixelEnv not available"}

    # --- Run episodes ---
    rng = np.random.RandomState(seed)
    results = []
    t0 = time.time()

    for ep in range(n_episodes):
        ep_seed = seed + ep
        env = CrafterPixelEnv(seed=ep_seed)

        ep_tracker = HomeostaticTracker()
        ep_tracker.init_from_textbook(tb)

        metrics = run_vector_mpc_episode(
            env=env,
            segmenter=segmenter,
            model=model,
            tracker=ep_tracker,
            rng=np.random.RandomState(ep_seed),
            max_steps=max_steps,
            horizon=10,
            beam_width=5,
            max_depth=3,
            verbose=verbose,
        )
        metrics["episode"] = ep
        metrics["seed"] = ep_seed
        results.append(metrics)

        elapsed = time.time() - t0
        eta = elapsed / (ep + 1) * (n_episodes - ep - 1)
        print(
            f"[{label}] ep{ep:3d}/{n_episodes} "
            f"len={metrics['avg_len']:3d} "
            f"cause={metrics['cause']:8s} "
            f"surprise={metrics['mean_surprise']:.3f} "
            f"wood={metrics['final_inv'].get('wood', 0)} "
            f"entropy={metrics['action_entropy']:.2f} "
            f"[{elapsed:.0f}s, ETA {eta:.0f}s]"
        )

    # --- Save experience ---
    if save_experience_path:
        model.save(save_experience_path)
        print(f"[{label}] Experience saved to {save_experience_path}")

    # --- Aggregate ---
    avg_len = np.mean([r["avg_len"] for r in results])
    wood_counts = [r["final_inv"].get("wood", 0) for r in results]
    wood_ge3_pct = sum(1 for w in wood_counts if w >= 3) / len(results)
    avg_surprise = np.mean([r["mean_surprise"] for r in results])
    avg_entropy = np.mean([r["action_entropy"] for r in results])

    summary = {
        "label": label,
        "n_episodes": n_episodes,
        "avg_len": round(float(avg_len), 1),
        "wood_ge3_pct": round(wood_ge3_pct, 3),
        "avg_wood": round(float(np.mean(wood_counts)), 2),
        "avg_surprise": round(float(avg_surprise), 4),
        "avg_entropy": round(float(avg_entropy), 3),
        "total_time_s": round(time.time() - t0, 1),
        "episodes": results,
    }

    # --- Gate checks ---
    gates = {
        "survival_ge_155": avg_len >= 155,
        "wood_ge3_10pct": wood_ge3_pct >= 0.10,
        "entropy_not_collapsed": avg_entropy > 0.5,
    }
    summary["gates"] = gates
    print(f"\n[{label}] === SUMMARY ===")
    print(f"  avg_len: {avg_len:.1f}")
    print(f"  wood≥3: {wood_ge3_pct*100:.1f}%")
    print(f"  avg_surprise: {avg_surprise:.4f}")
    print(f"  avg_entropy: {avg_entropy:.3f}")
    for gate, passed in gates.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {gate}: {status}")

    return summary


def main():
    """Run 3 ablations."""
    output_dir = Path("_docs")
    output_dir.mkdir(exist_ok=True)

    all_results = {}

    # --- Ablation 1: seed only ---
    print("\n" + "=" * 60)
    print("ABLATION 1: seed_only (textbook bootstrap, no experience)")
    print("=" * 60)
    r1 = run_eval(
        n_episodes=20,
        label="seed_only",
        save_experience_path=str(output_dir / "stage83_gen1_experience.bin"),
    )
    all_results["seed_only"] = r1

    # --- Ablation 2: seed + experience from gen1 ---
    print("\n" + "=" * 60)
    print("ABLATION 2: seed_experience (textbook + gen1 experience)")
    print("=" * 60)
    r2 = run_eval(
        n_episodes=20,
        label="seed_experience",
        experience_path=str(output_dir / "stage83_gen1_experience.bin"),
        save_experience_path=str(output_dir / "stage83_gen2_experience.bin"),
        seed=1042,  # different seed for fair comparison
    )
    all_results["seed_experience"] = r2

    # --- Ablation 3: multi-gen (3 generations) ---
    print("\n" + "=" * 60)
    print("ABLATION 3: multi_gen (3 generations)")
    print("=" * 60)
    exp_path = str(output_dir / "stage83_gen2_experience.bin")
    r3 = run_eval(
        n_episodes=20,
        label="multi_gen",
        experience_path=exp_path,
        seed=2042,
    )
    all_results["multi_gen"] = r3

    # --- Save all results ---
    # Strip episode details for summary
    summary = {}
    for k, v in all_results.items():
        s = {kk: vv for kk, vv in v.items() if kk != "episodes"}
        summary[k] = s

    result_path = output_dir / "stage83_eval_results.json"
    with open(result_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {result_path}")

    # Full results with episodes
    full_path = output_dir / "stage83_eval_full.json"
    with open(full_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Full results saved to {full_path}")

    # --- Knowledge flow gate ---
    if "seed_only" in all_results and "seed_experience" in all_results:
        g1 = all_results["seed_only"]["avg_len"]
        g2 = all_results["seed_experience"]["avg_len"]
        flow_pass = g2 > g1
        print(f"\nKnowledge flow: gen1={g1:.1f}, gen2={g2:.1f} → {'PASS' if flow_pass else 'FAIL'}")


if __name__ == "__main__":
    main()
