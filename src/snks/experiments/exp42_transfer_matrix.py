"""Stage 18 / exp42 — Transfer Matrix.

Protocol: load exp41 checkpoints, evaluate zero-shot transfer across 30 curated pairs.
Gate: transfer_score > 1.0 for >= 20% pairs (6 out of 30).
"""
from __future__ import annotations

import json
import multiprocessing
import os
import sys
import warnings

import numpy as np

from snks.experiments.stage18_utils import (
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

TRANSFER_PAIRS: list[tuple[str, list[str]]] = [
    (
        "MiniGrid-Empty-5x5-v0",
        [
            "MiniGrid-FourRooms-v0",
            "MiniGrid-DoorKey-5x5-v0",
            "MiniGrid-LavaCrossingS9N1-v0",
            "MiniGrid-MultiRoom-N2-S4-v0",
            "MiniGrid-SimpleCrossingS9N1-v0",
        ],
    ),
    (
        "MiniGrid-FourRooms-v0",
        [
            "MiniGrid-Empty-8x8-v0",
            "MiniGrid-DoorKey-5x5-v0",
            "MiniGrid-MultiRoom-N2-S4-v0",
            "MiniGrid-LavaCrossingS9N1-v0",
            "MiniGrid-KeyCorridorS3R1-v0",
        ],
    ),
    (
        "MiniGrid-DoorKey-5x5-v0",
        [
            "MiniGrid-DoorKey-8x8-v0",
            "MiniGrid-Unlock-v0",
            "MiniGrid-UnlockPickup-v0",
            "MiniGrid-KeyCorridorS3R1-v0",
            "MiniGrid-FourRooms-v0",
        ],
    ),
    (
        "MiniGrid-LavaCrossingS9N1-v0",
        [
            "MiniGrid-LavaCrossingS9N2-v0",
            "MiniGrid-SimpleCrossingS9N1-v0",
            "MiniGrid-Empty-5x5-v0",
            "MiniGrid-FourRooms-v0",
            "MiniGrid-ObstructedMaze-1Dlhb-v0",
        ],
    ),
    (
        "MiniGrid-MultiRoom-N2-S4-v0",
        [
            "MiniGrid-MultiRoom-N4-S5-v0",
            "MiniGrid-FourRooms-v0",
            "MiniGrid-DoorKey-5x5-v0",
            "MiniGrid-KeyCorridorS3R1-v0",
            "MiniGrid-MemoryS7-v0",
        ],
    ),
    (
        "MiniGrid-KeyCorridorS3R1-v0",
        [
            "MiniGrid-Unlock-v0",
            "MiniGrid-DoorKey-5x5-v0",
            "MiniGrid-UnlockPickup-v0",
            "MiniGrid-DoorKey-8x8-v0",
            "MiniGrid-MultiRoom-N2-S4-v0",
        ],
    ),
]

_EVAL_STEPS = 2_000

# Minimum fraction of pairs that must pass.
_GATE_FRACTION = 0.20
_TOTAL_PAIRS = 30
_GATE_N_POSITIVE = 6  # 20% of 30


# ---------------------------------------------------------------------------
# Helper: check if exp41 checkpoint exists for a source env
# ---------------------------------------------------------------------------

def _checkpoint_exists(source_env_id: str) -> bool:
    """Return True if exp41 final checkpoint exists for source_env_id.

    The checkpoint created by save_checkpoint() produces at minimum
    a ``{base_path}_daf.safetensors`` file.

    Args:
        source_env_id: MiniGrid environment ID used as the source.

    Returns:
        True if the DAF safetensors file is present on disk.
    """
    ckpt = checkpoint_path("exp41", source_env_id, "final")
    return os.path.isfile(ckpt + "_daf.safetensors")


# ---------------------------------------------------------------------------
# Eval helpers (module-level for multiprocessing pickling)
# ---------------------------------------------------------------------------

def _eval_agent(agent, env, n_steps: int) -> list[float]:
    """Run agent for n_steps, return per-step coverage values.

    Learning is expected to be disabled by the caller before this function
    is invoked. This function does not toggle enable_learning itself.

    Args:
        agent: EmbodiedAgent instance (learning already disabled).
        env: Gymnasium MiniGrid env.
        n_steps: Total number of environment steps to execute.

    Returns:
        List of float coverage_ratio values, one per step.
    """
    visited: set[tuple] = set()
    coverage_curve: list[float] = []

    obs, _ = env.reset(seed=42)

    for _ in range(n_steps):
        action = agent.step(img(obs))

        # Track agent position
        pos = getattr(env.unwrapped, "agent_pos", None)
        if pos is not None:
            visited.add(tuple(pos))

        obs_next, _, term, trunc, info = env.step(action)

        # info["agent_pos"] takes priority over unwrapped attr
        pos_info = info.get("agent_pos") if isinstance(info, dict) else None
        if pos_info is not None:
            visited.add(tuple(pos_info) if not isinstance(pos_info, tuple) else pos_info)

        coverage_curve.append(coverage_ratio(visited, env))

        agent.observe_result(img(obs_next))

        if term or trunc:
            agent.end_episode()
            obs, _ = env.reset()
        else:
            obs = obs_next

    return coverage_curve


def _compute_baseline_auc(target_env_id: str, device: str) -> float:
    """Random (no learning) agent baseline AUC in target environment.

    Creates a fresh agent with STDP learning disabled and runs it for
    _EVAL_STEPS steps to obtain a coverage curve, then returns the
    area-under-curve normalized by the number of steps.

    Args:
        target_env_id: MiniGrid environment ID to evaluate in.
        device: PyTorch device string.

    Returns:
        Normalized AUC (float), representing average coverage across steps.
    """
    from snks.agent.embodied_agent import EmbodiedAgent

    agent = EmbodiedAgent(build_agent_config(device))
    # Disable STDP learning so this is a pure random-walk baseline.
    agent.causal_agent.pipeline.engine.enable_learning = False

    env = make_env(target_env_id)
    coverage_values = _eval_agent(agent, env, _EVAL_STEPS)
    env.close()

    if len(coverage_values) == 0:
        return 0.0
    return float(np.trapz(coverage_values) / len(coverage_values))


def _eval_transfer_pair(
    source_env_id: str,
    target_env_id: str,
    baseline_auc: float,
    device: str,
) -> dict:
    """Load exp41 checkpoint for source, evaluate zero-shot in target env.

    STDP learning is frozen so we measure pure transfer from the source
    checkpoint without additional adaptation.

    Args:
        source_env_id: Source environment whose checkpoint to load.
        target_env_id: Target environment to evaluate in.
        baseline_auc: Normalised AUC of a no-learning agent in the target env.
        device: PyTorch device string.

    Returns:
        Dict with keys:
            source, target, adaptation_auc, baseline_auc, transfer_score,
            final_coverage.
    """
    from snks.agent.embodied_agent import EmbodiedAgent

    agent = EmbodiedAgent(build_agent_config(device))
    ckpt = checkpoint_path("exp41", source_env_id, "final")
    agent.load_checkpoint(ckpt)
    # Freeze STDP so we measure zero-shot transfer, not further adaptation.
    agent.causal_agent.pipeline.engine.enable_learning = False

    env = make_env(target_env_id)
    coverage_values = _eval_agent(agent, env, _EVAL_STEPS)
    env.close()

    if len(coverage_values) == 0:
        adaptation_auc = 0.0
    else:
        adaptation_auc = float(np.trapz(coverage_values) / len(coverage_values))

    transfer_score = adaptation_auc / baseline_auc if baseline_auc > 0 else 0.0

    return {
        "source": source_env_id,
        "target": target_env_id,
        "adaptation_auc": round(adaptation_auc, 6),
        "baseline_auc": round(baseline_auc, 6),
        "transfer_score": round(transfer_score, 6),
        "final_coverage": round(coverage_values[-1], 6) if coverage_values else 0.0,
    }


# ---------------------------------------------------------------------------
# Worker: evaluate one (source, targets_list) group
# ---------------------------------------------------------------------------

def _worker_eval_group(
    worker_id: int,
    source_env_id: str,
    target_env_ids: list[str],
    baseline_aucs: dict[str, float],
    device: str,
) -> list[dict]:
    """Evaluate all targets for a single source env.

    This is a module-level function (not a closure) so multiprocessing
    can pickle it.

    Args:
        worker_id: Worker index for logging.
        source_env_id: Source environment ID.
        target_env_ids: Target environment IDs to evaluate.
        baseline_aucs: Pre-computed baseline AUC per target env.
        device: PyTorch device string.

    Returns:
        List of result dicts from _eval_transfer_pair (skipped pairs omitted).
    """
    results: list[dict] = []

    if not _checkpoint_exists(source_env_id):
        warnings.warn(
            f"[worker {worker_id}] exp41 checkpoint missing for {source_env_id}, "
            "skipping all its target pairs.",
            stacklevel=2,
        )
        print(
            f"[worker {worker_id}] SKIP source={source_env_id} (no checkpoint)",
            flush=True,
        )
        return results

    for target_env_id in target_env_ids:
        print(
            f"[worker {worker_id}] evaluating {source_env_id} -> {target_env_id}",
            flush=True,
        )
        baseline_auc = baseline_aucs.get(target_env_id, 0.0)
        result = _eval_transfer_pair(source_env_id, target_env_id, baseline_auc, device)
        print(
            f"[worker {worker_id}] done {source_env_id} -> {target_env_id}: "
            f"transfer_score={result['transfer_score']:.4f}  "
            f"adapt_auc={result['adaptation_auc']:.4f}  "
            f"base_auc={result['baseline_auc']:.4f}",
            flush=True,
        )
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def _build_html_report(all_results: list[dict]) -> str:
    """Build a Plotly HTML report with a 6x5 transfer-score heatmap.

    Cells with transfer_score > 1.0 are shown in green, below 1.0 in red.
    Skipped pairs (missing checkpoints) are shown as grey N/A cells.

    Args:
        all_results: List of result dicts from _eval_transfer_pair.

    Returns:
        HTML string with embedded Plotly CDN script.
    """
    import json as _json

    # Build source x target matrix
    sources = [s for s, _ in TRANSFER_PAIRS]
    all_targets: list[str] = []
    seen: set[str] = set()
    for _, targets in TRANSFER_PAIRS:
        for t in targets:
            if t not in seen:
                all_targets.append(t)
                seen.add(t)

    # Map (source, target) -> score
    score_map: dict[tuple[str, str], float | None] = {}
    for r in all_results:
        score_map[(r["source"], r["target"])] = r["transfer_score"]

    # Build z matrix (sources as rows, targets as columns — first 5 per source)
    source_labels = [s.replace("MiniGrid-", "") for s in sources]
    # Use per-source target order to build 6x5 grid
    z_matrix: list[list[float]] = []
    target_label_cols: list[str] = []
    # Collect column labels from first source group
    for t in TRANSFER_PAIRS[0][1]:
        target_label_cols.append(t.replace("MiniGrid-", ""))

    # For heatmap we need a rectangular z. Use each source's own 5 targets.
    # Build a separate table in HTML for the full details.
    # Heatmap: 6 rows (sources) x 5 columns (positional targets per source)
    for source_env_id, target_env_ids in TRANSFER_PAIRS:
        row: list[float] = []
        for t in target_env_ids:
            val = score_map.get((source_env_id, t))
            row.append(val if val is not None else 0.0)
        z_matrix.append(row)

    # Build column labels from each source's own targets (positional)
    col_labels_per_row = []
    for _, targets in TRANSFER_PAIRS:
        col_labels_per_row.append([t.replace("MiniGrid-", "") for t in targets])

    # Custom colorscale: red at 0, white at 1.0, green above 1.0
    colorscale = [
        [0.0, "#c62828"],
        [0.5, "#ff8a65"],
        [1.0 / 2.0, "#ffffff"],
        [1.0, "#2e7d32"],
    ]

    # Build heatmap trace
    heatmap_trace = {
        "type": "heatmap",
        "z": z_matrix,
        "x": [f"T{i+1}" for i in range(5)],
        "y": source_labels,
        "colorscale": [
            [0.0, "#c62828"],
            [0.3, "#ef9a9a"],
            [0.5, "#ffffff"],
            [0.8, "#a5d6a7"],
            [1.0, "#2e7d32"],
        ],
        "zmid": 1.0,
        "zmin": 0.0,
        "zmax": 2.0,
        "colorbar": {
            "title": "transfer_score",
            "tickvals": [0.0, 0.5, 1.0, 1.5, 2.0],
            "ticktext": ["0.0", "0.5", "1.0 (gate)", "1.5", "2.0"],
        },
        "text": [[f"{v:.3f}" for v in row] for row in z_matrix],
        "texttemplate": "%{text}",
        "hovertemplate": "source: %{y}<br>target col: %{x}<br>score: %{z:.4f}<extra></extra>",
    }

    heatmap_layout = {
        "title": "Transfer Score Heatmap (6 sources x 5 targets each)",
        "paper_bgcolor": "#1a1a2e",
        "plot_bgcolor": "#16213e",
        "font": {"color": "#e0e0e0"},
        "xaxis": {"title": "Target slot (T1-T5)", "tickangle": -30},
        "yaxis": {"title": "Source environment", "autorange": "reversed"},
        "height": 420,
        "margin": {"l": 220, "r": 80, "b": 80, "t": 60},
    }

    # Summary table rows
    table_rows_html = ""
    for r in sorted(all_results, key=lambda x: (x["source"], x["target"])):
        score = r["transfer_score"]
        score_color = "#4caf50" if score > 1.0 else "#f44336"
        table_rows_html += (
            f"<tr>"
            f'<td>{r["source"].replace("MiniGrid-", "")}</td>'
            f'<td>{r["target"].replace("MiniGrid-", "")}</td>'
            f'<td>{r["adaptation_auc"]:.4f}</td>'
            f'<td>{r["baseline_auc"]:.4f}</td>'
            f'<td style="color:{score_color};font-weight:bold">{score:.4f}</td>'
            f'<td style="color:{score_color}">{"PASS" if score > 1.0 else "FAIL"}</td>'
            f"</tr>\n"
        )

    n_positive = sum(1 for r in all_results if r["transfer_score"] > 1.0)
    n_total = len(all_results)
    passed = n_positive >= _GATE_N_POSITIVE

    heatmap_trace_json = _json.dumps([heatmap_trace])
    heatmap_layout_json = _json.dumps(heatmap_layout)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <title>Stage 18 exp42 -- Transfer Matrix</title>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <style>
    body {{ background: #1a1a2e; color: #e0e0e0; font-family: monospace; margin: 24px; }}
    h1 {{ color: #90caf9; }}
    h2 {{ color: #b0bec5; margin-top: 32px; }}
    .chart {{ width: 100%; height: 460px; }}
    .status-pass {{ color: #4caf50; font-weight: bold; font-size: 1.2em; }}
    .status-fail {{ color: #f44336; font-weight: bold; font-size: 1.2em; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 16px; }}
    th {{ background: #0d1117; color: #90caf9; padding: 8px 12px; text-align: left; }}
    td {{ padding: 6px 12px; border-bottom: 1px solid #263238; }}
    tr:hover {{ background: #1e2a3a; }}
  </style>
</head>
<body>
  <h1>Stage 18 exp42 -- Transfer Matrix</h1>
  <p>Zero-shot transfer: load exp41 source checkpoint, evaluate {_EVAL_STEPS:,} steps in target env (no learning).</p>
  <p>Gate: transfer_score &gt; 1.0 for &gt;= {_GATE_N_POSITIVE} / {_TOTAL_PAIRS} pairs ({int(_GATE_FRACTION*100)}%).</p>
  <p>Result: <strong>{n_positive}</strong> / {n_total} pairs pass &nbsp;
    <span class="{'status-pass' if passed else 'status-fail'}">{'PASS' if passed else 'FAIL'}</span>
  </p>

  <h2>Transfer Score Heatmap</h2>
  <p>
    <span style="color:#2e7d32">Green = score &gt; 1.0 (transfers well)</span> &nbsp;|&nbsp;
    <span style="color:#c62828">Red = score &lt; 1.0 (worse than random)</span>
  </p>
  <p>Column labels T1–T5 map to each source's target order as listed in TRANSFER_PAIRS.</p>
  <div id="chart-heatmap" class="chart"></div>

  <h2>Pair Details</h2>
  <table>
    <thead>
      <tr>
        <th>source</th>
        <th>target</th>
        <th>adapt_auc</th>
        <th>baseline_auc</th>
        <th>transfer_score</th>
        <th>gate</th>
      </tr>
    </thead>
    <tbody>
      {table_rows_html}
    </tbody>
  </table>

  <script>
    Plotly.newPlot('chart-heatmap', {heatmap_trace_json}, {heatmap_layout_json}, {{responsive: true}});
  </script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------

def run(device: str | None = None) -> dict:
    """Run exp42: evaluate zero-shot transfer across 30 curated pairs.

    Steps:
        1. Compute baselines for all unique target envs (no-learning agents).
        2. Run 30 transfer pairs via Pool(5), one source per worker (5 targets each).
        3. Count pairs with transfer_score > 1.0.
        4. Gate: n_positive >= 6 (20% of 30).
        5. Save results/exp42_transfer.json and reports/exp42_transfer.html.

    Args:
        device: PyTorch device string. Auto-detected if None.

    Returns:
        Dict with keys: passed, n_positive, n_total, results.
    """
    if device is None:
        device = get_device()

    print(f"exp42: device={device}, pairs={_TOTAL_PAIRS}, workers=5", flush=True)

    # Step 1: collect unique target env IDs
    unique_targets: list[str] = []
    seen: set[str] = set()
    for _, targets in TRANSFER_PAIRS:
        for t in targets:
            if t not in seen:
                unique_targets.append(t)
                seen.add(t)

    print(f"exp42: computing baselines for {len(unique_targets)} unique target envs...", flush=True)
    baseline_aucs: dict[str, float] = {}
    for target_env_id in unique_targets:
        auc = _compute_baseline_auc(target_env_id, device)
        baseline_aucs[target_env_id] = auc
        print(f"  baseline {target_env_id}: auc={auc:.6f}", flush=True)

    # Step 2: run 30 pairs — 5 workers, each handles 1 source (5 targets)
    pool_args = [
        (i, source_env_id, target_env_ids, baseline_aucs, device)
        for i, (source_env_id, target_env_ids) in enumerate(TRANSFER_PAIRS)
    ]

    print("exp42: launching Pool(5) for transfer evaluation...", flush=True)
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=5) as pool:
        group_results: list[list[dict]] = pool.starmap(_worker_eval_group, pool_args)

    all_results: list[dict] = [r for group in group_results for r in group]

    # Step 3: count pairs passing the gate
    n_positive = sum(1 for r in all_results if r["transfer_score"] > 1.0)
    n_total = len(all_results)
    passed = n_positive >= _GATE_N_POSITIVE

    print(f"\n{'='*60}", flush=True)
    print("exp42 — Transfer Matrix", flush=True)
    print(f"{'='*60}", flush=True)
    for r in sorted(all_results, key=lambda x: (x["source"], x["target"])):
        flag = "[PASS]" if r["transfer_score"] > 1.0 else "[FAIL]"
        print(
            f"  {flag} {r['source'].replace('MiniGrid-', ''):30s}"
            f" -> {r['target'].replace('MiniGrid-', ''):30s}"
            f"  score={r['transfer_score']:.4f}",
            flush=True,
        )
    print(f"\nPositive pairs: {n_positive} / {n_total}", flush=True)
    print(f"Gate: >= {_GATE_N_POSITIVE} required", flush=True)
    print(f"{'PASS' if passed else 'FAIL'}", flush=True)

    # Step 5: persist JSON
    os.makedirs("results", exist_ok=True)
    results_path = os.path.join("results", "exp42_transfer.json")
    payload = {
        "passed": passed,
        "n_positive": n_positive,
        "n_total": n_total,
        "gate_n_positive": _GATE_N_POSITIVE,
        "gate_fraction": _GATE_FRACTION,
        "baseline_aucs": baseline_aucs,
        "results": all_results,
    }
    with open(results_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"exp42: results saved to {results_path}", flush=True)

    # HTML report
    os.makedirs("reports", exist_ok=True)
    report_path = os.path.join("reports", "exp42_transfer.html")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(_build_html_report(all_results))
    print(f"exp42: report saved to {report_path}", flush=True)

    return {
        "passed": passed,
        "n_positive": n_positive,
        "n_total": n_total,
        "results": all_results,
    }


# ---------------------------------------------------------------------------
# pytest test
# ---------------------------------------------------------------------------

def test_exp42_transfer_matrix() -> None:
    """Pytest test: evaluate 30 transfer pairs and assert gate passes."""
    result = run()
    assert result["passed"], (
        f"exp42 FAIL: {result['n_positive']}/{result['n_total']} pairs pass "
        f"(need >= {_GATE_N_POSITIVE})"
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    device_arg = sys.argv[1] if len(sys.argv) > 1 else None
    outcome = run(device=device_arg)
    sys.exit(0 if outcome["passed"] else 1)
