"""Stage 18 / exp41 — Multi-Env Baseline.

Protocol: train EmbodiedAgent(N=50K) independently on each of 15 MiniGrid envs.
Parallelism: 5 workers via multiprocessing.Pool, each trains 3 envs sequentially.
Gate: coverage_ratio > 0.3 for all easy/medium envs (difficulty in {"easy","medium"}).
"""
from __future__ import annotations

import json
import multiprocessing
import os
import sys
import time

import numpy as np

from snks.experiments.stage18_utils import (
    ENV_DIFFICULTY,
    ENV_STEPS,
    build_agent_config,
    checkpoint_path,
    coverage_ratio,
    get_device,
    img,
    make_env,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COVERAGE_GATE = 0.3  # required for easy + medium envs

WORKER_GROUPS: list[list[str]] = [
    [
        "MiniGrid-Empty-5x5-v0",
        "MiniGrid-FourRooms-v0",
        "MiniGrid-DoorKey-8x8-v0",
    ],
    [
        "MiniGrid-Empty-8x8-v0",
        "MiniGrid-MultiRoom-N2-S4-v0",
        "MiniGrid-MultiRoom-N4-S5-v0",
    ],
    [
        "MiniGrid-LavaCrossingS9N1-v0",
        "MiniGrid-LavaCrossingS9N2-v0",
        "MiniGrid-DoorKey-5x5-v0",
    ],
    [
        "MiniGrid-SimpleCrossingS9N1-v0",
        "MiniGrid-KeyCorridorS3R1-v0",
        "MiniGrid-Unlock-v0",
    ],
    [
        "MiniGrid-UnlockPickup-v0",
        "MiniGrid-MemoryS7-v0",
        "MiniGrid-ObstructedMaze-1Dlhb-v0",
    ],
]


# ---------------------------------------------------------------------------
# Single-env training
# ---------------------------------------------------------------------------

def _train_single_env(
    env_id: str,
    device: str,
    checkpoint_every: int = 10_000,
) -> dict:
    """Train a fresh EmbodiedAgent on one MiniGrid environment.

    Args:
        env_id: MiniGrid environment ID.
        device: PyTorch device string.
        checkpoint_every: Save a checkpoint every this many total steps.

    Returns:
        Dict with keys:
            env_id, n_steps, coverage_ratio, goal_seeking_steps,
            steps_per_sec, coverage_curve.
    """
    import random
    import torch

    # Independent seed per env so results are reproducible but not correlated.
    seed = abs(hash(env_id)) % (2**31)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))

    n_train_steps = ENV_STEPS[env_id]
    config = build_agent_config(device)

    from snks.agent.embodied_agent import EmbodiedAgent

    agent = EmbodiedAgent(config)
    env = make_env(env_id)

    total_steps = 0
    goal_seeking_steps = 0
    all_visited: set[tuple] = set()
    coverage_curve: list[tuple[int, float]] = []
    next_checkpoint = checkpoint_every
    next_curve_record = 1_000

    t_start = time.perf_counter()

    while total_steps < n_train_steps:
        _obs, _ = env.reset(seed=total_steps)
        obs = img(_obs)
        done = False
        ep_visited: set[tuple] = set()

        while not done and total_steps < n_train_steps:
            action = agent.step(obs)

            # Track goal-seeking mode
            result = agent.causal_agent.pipeline.last_cycle_result
            if result is not None and result.configurator_action is not None:
                if result.configurator_action.mode == "goal_seeking":
                    goal_seeking_steps += 1

            # Track visited positions
            pos = tuple(env.unwrapped.agent_pos)
            ep_visited.add(pos)

            _obs_next, _, terminated, truncated, _ = env.step(action)
            obs_next = img(_obs_next)
            done = terminated or truncated
            total_steps += 1

            agent.observe_result(obs_next)
            obs = obs_next

            # Coverage curve recording
            if total_steps >= next_curve_record:
                all_visited.update(ep_visited)
                cr = coverage_ratio(all_visited, env)
                coverage_curve.append((total_steps, round(cr, 4)))
                next_curve_record += 1_000

            # Periodic checkpoint
            if total_steps >= next_checkpoint:
                ckpt = checkpoint_path("exp41", env_id, total_steps)
                os.makedirs(os.path.dirname(ckpt), exist_ok=True)
                agent.save_checkpoint(ckpt)
                next_checkpoint += checkpoint_every

        all_visited.update(ep_visited)
        agent.end_episode()

    t_elapsed = time.perf_counter() - t_start
    steps_per_sec = total_steps / t_elapsed if t_elapsed > 0 else 0.0

    # Final coverage
    final_cr = coverage_ratio(all_visited, env)

    # Final checkpoint
    ckpt_final = checkpoint_path("exp41", env_id, "final")
    os.makedirs(os.path.dirname(ckpt_final), exist_ok=True)
    agent.save_checkpoint(ckpt_final)

    env.close()

    return {
        "env_id": env_id,
        "n_steps": total_steps,
        "coverage_ratio": round(final_cr, 4),
        "goal_seeking_steps": goal_seeking_steps,
        "steps_per_sec": round(steps_per_sec, 2),
        "coverage_curve": coverage_curve,
    }


# ---------------------------------------------------------------------------
# Worker group
# ---------------------------------------------------------------------------

def _train_group(worker_id: int, env_ids: list[str], device: str) -> list[dict]:
    """Train each env in the group sequentially.

    This function is a module-level function (not a closure) so it can be
    pickled by multiprocessing.Pool.

    Args:
        worker_id: Index of this worker (0–4), used for logging.
        env_ids: List of env IDs to train sequentially.
        device: PyTorch device string.

    Returns:
        List of result dicts from _train_single_env.
    """
    results: list[dict] = []
    for env_id in env_ids:
        print(f"[worker {worker_id}] starting {env_id}", flush=True)
        result = _train_single_env(env_id, device)
        print(
            f"[worker {worker_id}] done {env_id}: "
            f"coverage={result['coverage_ratio']:.3f}  "
            f"steps/sec={result['steps_per_sec']:.1f}",
            flush=True,
        )
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

_DIFFICULTY_COLORS = {
    "easy":   "#4caf50",   # green
    "medium": "#ff9800",   # orange
    "hard":   "#f44336",   # red
}


def _build_html_report(all_results: list[dict]) -> str:
    """Build a Plotly HTML report for exp41.

    Args:
        all_results: List of result dicts from _train_single_env.

    Returns:
        HTML string.
    """
    # Sort by difficulty ordering for display
    _diff_order = {"easy": 0, "medium": 1, "hard": 2}
    sorted_results = sorted(
        all_results,
        key=lambda r: (_diff_order.get(ENV_DIFFICULTY.get(r["env_id"], "hard"), 2), r["env_id"]),
    )

    # Coverage curves chart data (one trace per env)
    curve_traces = []
    for r in sorted_results:
        env_id = r["env_id"]
        difficulty = ENV_DIFFICULTY.get(env_id, "hard")
        color = _DIFFICULTY_COLORS[difficulty]
        curve = r["coverage_curve"]
        xs = [c[0] for c in curve]
        ys = [c[1] for c in curve]
        curve_traces.append(
            {
                "x": xs,
                "y": ys,
                "name": env_id.replace("MiniGrid-", ""),
                "mode": "lines",
                "line": {"color": color, "width": 2},
                "hovertemplate": f"{env_id}<br>step: %{{x}}<br>coverage: %{{y:.3f}}<extra></extra>",
            }
        )

    # Bar chart: final coverage per env
    bar_envs = [r["env_id"].replace("MiniGrid-", "") for r in sorted_results]
    bar_values = [r["coverage_ratio"] for r in sorted_results]
    bar_colors = [_DIFFICULTY_COLORS[ENV_DIFFICULTY.get(r["env_id"], "hard")] for r in sorted_results]

    bar_trace = {
        "x": bar_envs,
        "y": bar_values,
        "type": "bar",
        "marker": {"color": bar_colors},
        "hovertemplate": "%{x}<br>coverage: %{y:.3f}<extra></extra>",
        "name": "Final coverage",
    }

    # Gate line at 0.3
    gate_shape = {
        "type": "line",
        "x0": -0.5,
        "x1": len(bar_envs) - 0.5,
        "y0": COVERAGE_GATE,
        "y1": COVERAGE_GATE,
        "line": {"color": "#ffffff", "width": 1, "dash": "dash"},
    }

    # Summary table rows
    table_rows_html = ""
    for r in sorted_results:
        env_id = r["env_id"]
        diff = ENV_DIFFICULTY.get(env_id, "hard")
        color = _DIFFICULTY_COLORS[diff]
        gate_ok = diff in ("easy", "medium") and r["coverage_ratio"] > COVERAGE_GATE
        gate_cell = (
            f'<td style="color:#4caf50">PASS</td>'
            if gate_ok or diff == "hard"
            else f'<td style="color:#f44336">FAIL</td>'
        )
        table_rows_html += (
            f"<tr>"
            f'<td>{env_id}</td>'
            f'<td style="color:{color}">{diff}</td>'
            f"<td>{r['n_steps']:,}</td>"
            f"<td>{r['coverage_ratio']:.3f}</td>"
            f"<td>{r['goal_seeking_steps']:,}</td>"
            f"<td>{r['steps_per_sec']:.1f}</td>"
            f"{gate_cell}"
            f"</tr>\n"
        )

    import json as _json

    curve_traces_json = _json.dumps(curve_traces)
    bar_trace_json = _json.dumps([bar_trace])
    bar_layout_json = _json.dumps(
        {
            "title": "Final Coverage Ratio per Environment",
            "paper_bgcolor": "#1a1a2e",
            "plot_bgcolor": "#16213e",
            "font": {"color": "#e0e0e0"},
            "xaxis": {"tickangle": -40, "tickfont": {"size": 10}},
            "yaxis": {"title": "coverage_ratio", "range": [0, 1]},
            "shapes": [gate_shape],
            "showlegend": False,
            "margin": {"b": 160},
        }
    )
    curve_layout_json = _json.dumps(
        {
            "title": "Coverage Curves (all 15 envs)",
            "paper_bgcolor": "#1a1a2e",
            "plot_bgcolor": "#16213e",
            "font": {"color": "#e0e0e0"},
            "xaxis": {"title": "total steps"},
            "yaxis": {"title": "coverage_ratio", "range": [0, 1]},
            "legend": {"font": {"size": 9}},
        }
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <title>Stage 18 exp41 -- Multi-Env Baseline</title>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <style>
    body {{ background: #1a1a2e; color: #e0e0e0; font-family: monospace; margin: 24px; }}
    h1 {{ color: #90caf9; }}
    h2 {{ color: #b0bec5; margin-top: 32px; }}
    .chart {{ width: 100%; height: 500px; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 16px; }}
    th {{ background: #0d1117; color: #90caf9; padding: 8px 12px; text-align: left; }}
    td {{ padding: 6px 12px; border-bottom: 1px solid #263238; }}
    tr:hover {{ background: #1e2a3a; }}
  </style>
</head>
<body>
  <h1>Stage 18 exp41 -- Multi-Env Baseline</h1>
  <p>N=50K DAF nodes, replay=uniform, 5 workers x 3 envs, 15 MiniGrid environments.</p>
  <p>Gate: coverage_ratio &gt; {COVERAGE_GATE} for all easy + medium envs.</p>

  <h2>Coverage Curves</h2>
  <p>
    <span style="color:#4caf50">green = easy</span> &nbsp;
    <span style="color:#ff9800">orange = medium</span> &nbsp;
    <span style="color:#f44336">red = hard</span>
  </p>
  <div id="chart-curves" class="chart"></div>

  <h2>Final Coverage per Environment</h2>
  <p>Dashed white line = gate threshold ({COVERAGE_GATE}).</p>
  <div id="chart-bars" class="chart"></div>

  <h2>Summary Table</h2>
  <table>
    <thead>
      <tr>
        <th>env_id</th>
        <th>difficulty</th>
        <th>n_steps</th>
        <th>final_coverage</th>
        <th>goal_seeking_steps</th>
        <th>steps_per_sec</th>
        <th>gate</th>
      </tr>
    </thead>
    <tbody>
      {table_rows_html}
    </tbody>
  </table>

  <script>
    Plotly.newPlot('chart-curves', {curve_traces_json}, {curve_layout_json}, {{responsive: true}});
    Plotly.newPlot('chart-bars',   {bar_trace_json},   {bar_layout_json},   {{responsive: true}});
  </script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------

def run(device: str | None = None) -> dict:
    """Run exp41: train 15 MiniGrid envs in parallel (5 workers x 3 envs).

    Args:
        device: PyTorch device. If None, auto-detected via get_device().

    Returns:
        Dict with keys: passed, results, gate_details.
    """
    if device is None:
        device = get_device()

    print(f"exp41: device={device}, workers=5, envs=15", flush=True)

    # Build starmap args: (worker_id, env_ids, device)
    pool_args = [
        (i, WORKER_GROUPS[i], device)
        for i in range(len(WORKER_GROUPS))
    ]

    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=5) as pool:
        group_results: list[list[dict]] = pool.starmap(_train_group, pool_args)

    # Flatten results list
    all_results: list[dict] = [r for group in group_results for r in group]

    # Evaluate gate: coverage > 0.3 for easy + medium envs
    gate_details: dict[str, bool] = {}
    passed = True
    for r in all_results:
        env_id = r["env_id"]
        diff = ENV_DIFFICULTY.get(env_id, "hard")
        if diff in ("easy", "medium"):
            ok = r["coverage_ratio"] > COVERAGE_GATE
            gate_key = (
                f"{env_id} [{diff}]: coverage({r['coverage_ratio']:.4f}) > {COVERAGE_GATE}"
            )
            gate_details[gate_key] = ok
            if not ok:
                passed = False

    # Persist results JSON
    os.makedirs("results", exist_ok=True)
    results_path = os.path.join("results", "exp41_baseline.json")
    with open(results_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "passed": passed,
                "gate_details": gate_details,
                "results": {r["env_id"]: r for r in all_results},
            },
            fh,
            indent=2,
        )
    print(f"exp41: results saved to {results_path}", flush=True)

    # Persist HTML report
    os.makedirs("reports", exist_ok=True)
    report_path = os.path.join("reports", "exp41_baseline.html")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(_build_html_report(all_results))
    print(f"exp41: report saved to {report_path}", flush=True)

    # Console summary
    print(f"\n{'='*60}", flush=True)
    print("exp41 — Multi-Env Baseline", flush=True)
    print(f"{'='*60}", flush=True)
    for r in sorted(all_results, key=lambda x: x["env_id"]):
        diff = ENV_DIFFICULTY.get(r["env_id"], "hard")
        flag = (
            "[GATE]" if diff in ("easy", "medium") else "      "
        )
        print(
            f"  {flag} {r['env_id']:45s}  "
            f"cov={r['coverage_ratio']:.3f}  "
            f"gs={r['goal_seeking_steps']:6d}  "
            f"{r['steps_per_sec']:.1f} sps",
            flush=True,
        )
    print("\nGate details:", flush=True)
    for k, v in gate_details.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}", flush=True)
    print(f"\n{'PASS' if passed else 'FAIL'}", flush=True)

    return {
        "passed": passed,
        "results": {r["env_id"]: r for r in all_results},
        "gate_details": gate_details,
    }


# ---------------------------------------------------------------------------
# pytest test
# ---------------------------------------------------------------------------

def test_exp41_multi_env_baseline() -> None:
    """Pytest test: train 15 envs and assert gate passes."""
    result = run()
    assert result["passed"], (
        f"exp41 FAIL: gate details={result['gate_details']}"
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Required on Windows to prevent recursive process spawning.
    multiprocessing.set_start_method("spawn", force=True)

    device_arg = sys.argv[1] if len(sys.argv) > 1 else None
    outcome = run(device=device_arg)
    sys.exit(0 if outcome["passed"] else 1)
