"""Experiment 93: Curriculum Learning (Stage 36).

AutonomousAgent uses progressive curriculum 5x5 → 6x6 → 8x8 → 16x16
with causal knowledge transfer between grid sizes.

Gates:
    success_5x5 >= 0.8
    success_8x8 >= 0.5
    success_16x16 >= 0.1    (stretch goal: any success on 16x16)
"""

from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding="utf-8")

from snks.language.autonomous_agent import AutonomousAgent


def main():
    print("=" * 60)
    print("Experiment 93: Curriculum Learning (Stage 36)")
    print("=" * 60)

    agent = AutonomousAgent(
        levels=[5, 6, 8, 16],
        advance_threshold=0.4,
    )

    def _callback(ep, grid_size, result):
        if (ep + 1) % 10 == 0:
            stats = agent.curriculum.get_stats(grid_size)
            print(f"  ep={ep+1} grid={grid_size}x{grid_size} "
                  f"success={stats.success_rate:.2f} ({stats.successes}/{stats.episodes}) "
                  f"links={agent.causal_model.n_links}")

    cr = agent.run_curriculum(total_episodes=500, callback=_callback)

    print(f"\n--- Results ---")
    print(f"  Total episodes: {cr.total_episodes}")
    print(f"  Total steps: {cr.total_steps}")
    print(f"  Elapsed: {cr.elapsed_seconds:.1f}s")
    print(f"  Final grid: {cr.final_grid_size}x{cr.final_grid_size}")
    print(f"  Causal links: {cr.causal_links}")

    for size, stats in sorted(cr.level_stats.items()):
        sr = stats["success_rate"]
        eps = stats["episodes"]
        succ = stats["successes"]
        print(f"  Grid {size}x{size}: {sr:.3f} ({succ}/{eps})")

    # Gates
    sr5 = cr.level_stats.get(5, {}).get("success_rate", 0)
    sr8 = cr.level_stats.get(8, {}).get("success_rate", 0)
    sr16 = cr.level_stats.get(16, {}).get("success_rate", 0)

    gate_5x5 = sr5 >= 0.8
    gate_8x8 = sr8 >= 0.5
    gate_16x16 = sr16 >= 0.1

    print(f"\n{'=' * 60}")
    print(f"GATE: 5x5 success {'PASS' if gate_5x5 else 'FAIL'} ({sr5:.3f} >= 0.800)")
    print(f"GATE: 8x8 success {'PASS' if gate_8x8 else 'FAIL'} ({sr8:.3f} >= 0.500)")
    print(f"GATE: 16x16 success {'PASS' if gate_16x16 else 'FAIL'} ({sr16:.3f} >= 0.100)")
    print(f"{'=' * 60}")

    all_pass = gate_5x5 and gate_8x8
    # 16x16 is stretch goal — not required for overall pass
    if all_pass:
        print("*** ALL GATES PASS ***")
    else:
        print("*** SOME GATES FAIL ***")


if __name__ == "__main__":
    main()
