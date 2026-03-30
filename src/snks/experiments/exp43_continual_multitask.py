"""Stage 18 / exp43 — Continual Learning & Multitask.

Protocol:
  43a (Sequential): Train EmbodiedAgent sequentially A→B→C on 5 chains of 3 envs.
       Measure retention_ratio = coverage_after_later / coverage_after_initial.
       Gate: retention_ratio >= 0.8 for >= 80% of transitions (8 out of 10).
  43b (Multitask): Single EmbodiedAgent trained on random env per step for 200K steps.
       Every 10K steps evaluate 500 steps per env (no STDP).
       Gate: mean(multitask_coverage[env] / baseline_coverage[env]) >= 0.5.
"""
from __future__ import annotations

import json
import multiprocessing
import os
import random
import sys
import time
from typing import Optional

import numpy as np

from stage18_utils import (
    ENVS,
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

# 43a: 5 sequential chains, each A→B→C
CHAINS: list[tuple[str, str, str]] = [
    (
        "MiniGrid-Empty-5x5-v0",
        "MiniGrid-FourRooms-v0",
        "MiniGrid-DoorKey-5x5-v0",
    ),
    (
        "MiniGrid-Empty-8x8-v0",
        "MiniGrid-MultiRoom-N2-S4-v0",
        "MiniGrid-MultiRoom-N4-S5-v0",
    ),
    (
        "MiniGrid-SimpleCrossingS9N1-v0",
        "MiniGrid-LavaCrossingS9N1-v0",
        "MiniGrid-LavaCrossingS9N2-v0",
    ),
    (
        "MiniGrid-Unlock-v0",
        "MiniGrid-DoorKey-5x5-v0",
        "MiniGrid-UnlockPickup-v0",
    ),
    (
        "MiniGrid-KeyCorridorS3R1-v0",
        "MiniGrid-DoorKey-8x8-v0",
        "MiniGrid-ObstructedMaze-1Dlhb-v0",
    ),
]

_EVAL_STEPS_RETENTION = 1_000   # steps per retention eval (no learning)
_MULTITASK_STEPS = 200_000      # total training steps for 43b
_MULTITASK_EVAL_EVERY = 10_000  # eval frequency in 43b
_MULTITASK_EVAL_STEPS = 500     # steps per env per eval in 43b

_RETENTION_GATE = 0.8   # retention_ratio threshold
_RETENTION_MIN_PASS = 8  # out of 10 transitions must pass
_MULTITASK_RATIO_GATE = 0.5  # multitask / baseline >= 0.5


# ---------------------------------------------------------------------------
# Shared episode loop helpers
# ---------------------------------------------------------------------------

def _train_env(agent, env, n_steps: int, seed_offset: int = 0) -> float:
    """Train agent on env for n_steps. Returns final coverage_ratio.

    Args:
        agent: EmbodiedAgent instance.
        env: Gymnasium environment.
        n_steps: Total step budget.
        seed_offset: Added to episode seed for reproducibility.

    Returns:
        Final coverage_ratio over all visited positions.
    """
    import torch

    total_steps = 0
    all_visited: set[tuple] = set()

    while total_steps < n_steps:
        _obs, _ = env.reset(seed=total_steps + seed_offset)
        obs = img(_obs)
        done = False
        ep_visited: set[tuple] = set()

        while not done and total_steps < n_steps:
            action = agent.step(obs)
            pos = tuple(env.unwrapped.agent_pos)
            ep_visited.add(pos)

            _obs_next, _, terminated, truncated, _ = env.step(action)
            obs_next = img(_obs_next)
            done = terminated or truncated
            total_steps += 1

            agent.observe_result(obs_next)
            obs = obs_next

        all_visited.update(ep_visited)
        agent.end_episode()

    return coverage_ratio(all_visited, env)


def _eval_env(agent, env, n_steps: int = _EVAL_STEPS_RETENTION) -> float:
    """Evaluate agent on env for n_steps WITHOUT STDP learning.

    Args:
        agent: EmbodiedAgent instance.
        env: Gymnasium environment.
        n_steps: Number of eval steps.

    Returns:
        coverage_ratio over visited positions.
    """
    engine = agent.causal_agent.pipeline.engine
    was_learning = engine.enable_learning
    engine.enable_learning = False

    try:
        all_visited: set[tuple] = set()
        total_steps = 0

        while total_steps < n_steps:
            _obs, _ = env.reset(seed=1000 + total_steps)
            obs = img(_obs)
            done = False
            ep_visited: set[tuple] = set()

            while not done and total_steps < n_steps:
                action = agent.step(obs)
                pos = tuple(env.unwrapped.agent_pos)
                ep_visited.add(pos)

                _obs_next, _, terminated, truncated, _ = env.step(action)
                obs_next = img(_obs_next)
                done = terminated or truncated
                total_steps += 1

                agent.observe_result(obs_next)
                obs = obs_next

            all_visited.update(ep_visited)
            agent.end_episode()

        return coverage_ratio(all_visited, env)
    finally:
        engine.enable_learning = was_learning


# ---------------------------------------------------------------------------
# 43a: Sequential continual learning
# ---------------------------------------------------------------------------

def _run_chain(args: tuple) -> dict:
    """Run one A→B→C chain. Module-level for multiprocessing pickling.

    Args:
        args: (chain_idx, chain_tuple, device)

    Returns:
        Dict with retention metrics per transition.
    """
    chain_idx, chain, device = args
    env_a, env_b, env_c = chain

    torch_seed = chain_idx * 1000 + 42
    import torch
    torch.manual_seed(torch_seed)
    random.seed(torch_seed)
    np.random.seed(torch_seed)

    from snks.agent.embodied_agent import EmbodiedAgent
    agent = EmbodiedAgent(build_agent_config(device))

    envs = {
        env_a: make_env(env_a),
        env_b: make_env(env_b),
        env_c: make_env(env_c),
    }

    print(f"[chain {chain_idx}] Training on A={env_a}", flush=True)
    _train_env(agent, envs[env_a], ENV_STEPS[env_a], seed_offset=chain_idx * 10000)
    cov_a_after_a = _eval_env(agent, envs[env_a])
    print(f"[chain {chain_idx}] A coverage after A: {cov_a_after_a:.3f}", flush=True)

    print(f"[chain {chain_idx}] Training on B={env_b}", flush=True)
    _train_env(agent, envs[env_b], ENV_STEPS[env_b], seed_offset=chain_idx * 10000 + 100)
    cov_a_after_b = _eval_env(agent, envs[env_a])
    cov_b_after_b = _eval_env(agent, envs[env_b])
    print(
        f"[chain {chain_idx}] A retention after B: {cov_a_after_b:.3f} "
        f"(ratio={cov_a_after_b / cov_a_after_a:.3f if cov_a_after_a > 0 else 'N/A'})",
        flush=True,
    )

    print(f"[chain {chain_idx}] Training on C={env_c}", flush=True)
    _train_env(agent, envs[env_c], ENV_STEPS[env_c], seed_offset=chain_idx * 10000 + 200)
    cov_a_after_c = _eval_env(agent, envs[env_a])
    cov_b_after_c = _eval_env(agent, envs[env_b])
    cov_c_after_c = _eval_env(agent, envs[env_c])
    print(
        f"[chain {chain_idx}] Retention after C: "
        f"A={cov_a_after_c:.3f} B={cov_b_after_c:.3f} C={cov_c_after_c:.3f}",
        flush=True,
    )

    for env in envs.values():
        env.close()

    # 2 transitions per chain: A→B and B→C
    # Retention = coverage_prev_after_later / coverage_prev_after_initial
    def safe_ratio(after: float, before: float) -> float:
        return after / before if before > 1e-6 else 1.0

    return {
        "chain_idx": chain_idx,
        "envs": [env_a, env_b, env_c],
        "coverage": {
            f"{env_a}_after_{env_a}": round(cov_a_after_a, 4),
            f"{env_a}_after_{env_b}": round(cov_a_after_b, 4),
            f"{env_a}_after_{env_c}": round(cov_a_after_c, 4),
            f"{env_b}_after_{env_b}": round(cov_b_after_b, 4),
            f"{env_b}_after_{env_c}": round(cov_b_after_c, 4),
            f"{env_c}_after_{env_c}": round(cov_c_after_c, 4),
        },
        "transitions": {
            f"A→B (retain {env_a})": {
                "retention_ratio": round(safe_ratio(cov_a_after_b, cov_a_after_a), 4),
                "passed": safe_ratio(cov_a_after_b, cov_a_after_a) >= _RETENTION_GATE,
            },
            f"B→C (retain {env_a})": {
                "retention_ratio": round(safe_ratio(cov_a_after_c, cov_a_after_a), 4),
                "passed": safe_ratio(cov_a_after_c, cov_a_after_a) >= _RETENTION_GATE,
            },
        },
    }


# ---------------------------------------------------------------------------
# 43b: Multitask training
# ---------------------------------------------------------------------------

def _run_multitask(device: str, baseline_coverages: dict[str, float]) -> dict:
    """Train one agent on randomly selected envs for _MULTITASK_STEPS steps.

    Args:
        device: PyTorch device string.
        baseline_coverages: Dict env_id → coverage from exp41 (or fallback 0.1).

    Returns:
        Dict with eval history and final ratios per env.
    """
    import torch

    torch.manual_seed(7)
    random.seed(7)
    np.random.seed(7)

    from snks.agent.embodied_agent import EmbodiedAgent
    agent = EmbodiedAgent(build_agent_config(device))

    all_env_ids = [env_id for env_id, _, _ in ENVS]
    envs = {env_id: make_env(env_id) for env_id in all_env_ids}

    # Pre-reset all envs
    current_obs = {}
    for env_id, env in envs.items():
        obs, _ = env.reset(seed=0)
        current_obs[env_id] = img(obs)

    total_steps = 0
    next_eval = _MULTITASK_EVAL_EVERY
    eval_history: list[dict] = []
    t_start = time.perf_counter()

    print(f"[43b] Starting multitask training for {_MULTITASK_STEPS:,} steps", flush=True)

    while total_steps < _MULTITASK_STEPS:
        # Pick random env
        env_id = random.choice(all_env_ids)
        env = envs[env_id]
        obs = current_obs[env_id]

        action = agent.step(obs)
        obs_next_raw, _, term, trunc, _ = env.step(action)
        obs_next = img(obs_next_raw)
        agent.observe_result(obs_next)

        if term or trunc:
            agent.end_episode()
            obs_reset, _ = env.reset(seed=total_steps)
            current_obs[env_id] = img(obs_reset)
        else:
            current_obs[env_id] = obs_next

        total_steps += 1

        # Periodic evaluation
        if total_steps >= next_eval:
            elapsed = time.perf_counter() - t_start
            sps = total_steps / elapsed if elapsed > 0 else 0
            print(
                f"[43b] step={total_steps:,}  steps/sec={sps:.1f}  — evaluating all envs...",
                flush=True,
            )
            eval_coverages: dict[str, float] = {}
            for eid, eenv in envs.items():
                cov = _eval_env(agent, eenv, _MULTITASK_EVAL_STEPS)
                eval_coverages[eid] = round(cov, 4)
            eval_history.append({"step": total_steps, "coverage": eval_coverages})
            next_eval += _MULTITASK_EVAL_EVERY

    for env in envs.values():
        env.close()

    # Final coverage from last eval
    if eval_history:
        final_coverages = eval_history[-1]["coverage"]
    else:
        final_coverages = {eid: 0.0 for eid in all_env_ids}

    # Ratio vs baseline
    ratios: dict[str, float] = {}
    for env_id in all_env_ids:
        baseline = baseline_coverages.get(env_id, 0.1)
        mt_cov = final_coverages.get(env_id, 0.0)
        ratios[env_id] = round(mt_cov / baseline if baseline > 1e-6 else 0.0, 4)

    mean_ratio = float(np.mean(list(ratios.values())))
    passed = mean_ratio >= _MULTITASK_RATIO_GATE

    print(
        f"[43b] Done. mean_ratio={mean_ratio:.3f}  passed={passed}",
        flush=True,
    )

    return {
        "mean_ratio": round(mean_ratio, 4),
        "passed": passed,
        "ratios": ratios,
        "final_coverages": final_coverages,
        "eval_history": eval_history,
    }


# ---------------------------------------------------------------------------
# HTML reports
# ---------------------------------------------------------------------------

def _build_continual_html(chain_results: list[dict]) -> str:
    """Build dark-theme plotly HTML report for 43a retention results."""
    import json as _json

    rows_html = ""
    for cr in chain_results:
        for trans_name, trans_data in cr["transitions"].items():
            color = "#4eff7a" if trans_data["passed"] else "#f44336"
            rows_html += (
                f"<tr>"
                f"<td>Chain {cr['chain_idx'] + 1}: {' → '.join(e.replace('MiniGrid-','') for e in cr['envs'])}</td>"
                f"<td>{trans_name}</td>"
                f"<td style='color:{color}'>{trans_data['retention_ratio']:.3f}</td>"
                f"<td style='color:{color}'>{'PASS' if trans_data['passed'] else 'FAIL'}</td>"
                f"</tr>"
            )

    # Bar chart data: all transitions
    bar_labels = []
    bar_values = []
    bar_colors_list = []
    for cr in chain_results:
        for trans_name, trans_data in cr["transitions"].items():
            label = f"C{cr['chain_idx']+1}: {trans_name[:20]}"
            bar_labels.append(label)
            bar_values.append(trans_data["retention_ratio"])
            bar_colors_list.append("#4eff7a" if trans_data["passed"] else "#f44336")

    bar_data = _json.dumps([{
        "x": bar_labels,
        "y": bar_values,
        "type": "bar",
        "marker": {"color": bar_colors_list},
        "hovertemplate": "%{x}<br>retention: %{y:.3f}<extra></extra>",
        "name": "Retention ratio",
    }])

    gate_shape = _json.dumps([{
        "type": "line",
        "x0": -0.5, "x1": len(bar_labels) - 0.5,
        "y0": _RETENTION_GATE, "y1": _RETENTION_GATE,
        "line": {"color": "#ffffff", "width": 1, "dash": "dash"},
    }])

    layout = _json.dumps({
        "title": "Stage 18 exp43a — Retention Ratios per Transition",
        "paper_bgcolor": "#1a1a2e",
        "plot_bgcolor": "#12122b",
        "font": {"color": "#eee"},
        "yaxis": {"title": "retention_ratio", "range": [0, 1.4]},
        "xaxis": {"tickangle": -30},
        "shapes": _json.loads(gate_shape),
        "showlegend": False,
        "margin": {"b": 160},
    })

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Stage 18 exp43a — Continual Learning</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  body {{ background: #1a1a2e; color: #eee; font-family: monospace; margin: 0; padding: 20px; }}
  h1 {{ color: #4a9eff; }}
  table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
  th, td {{ border: 1px solid #333; padding: 8px 12px; text-align: left; }}
  th {{ background: #12122b; color: #4a9eff; }}
  tr:hover {{ background: #1e1e3f; }}
  #chart {{ width: 100%; height: 480px; }}
</style>
</head>
<body>
<h1>Stage 18 / exp43a — Sequential Continual Learning</h1>
<p>5 chains × A→B→C. Gate: retention_ratio &ge; {_RETENTION_GATE} for &ge; {_RETENTION_MIN_PASS}/10 transitions.</p>
<div id="chart"></div>
<table>
  <tr><th>Chain</th><th>Transition</th><th>Retention Ratio</th><th>Result</th></tr>
  {rows_html}
</table>
<script>
  Plotly.newPlot('chart', {bar_data}, {layout}, {{responsive: true}});
</script>
</body>
</html>"""


def _build_multitask_html(multitask_result: dict, baseline_coverages: dict) -> str:
    """Build dark-theme plotly HTML report for 43b multitask results."""
    import json as _json

    env_ids = list(multitask_result["ratios"].keys())
    ratios = [multitask_result["ratios"][e] for e in env_ids]
    mt_covs = [multitask_result["final_coverages"].get(e, 0) for e in env_ids]
    base_covs = [baseline_coverages.get(e, 0) for e in env_ids]
    short_names = [e.replace("MiniGrid-", "") for e in env_ids]

    ratio_colors = ["#4eff7a" if r >= _MULTITASK_RATIO_GATE else "#ff9800" for r in ratios]

    bar_data = _json.dumps([
        {
            "x": short_names, "y": mt_covs, "type": "bar",
            "name": "Multitask coverage", "marker": {"color": "#4a9eff"},
            "hovertemplate": "%{x}<br>multitask: %{y:.3f}<extra></extra>",
        },
        {
            "x": short_names, "y": base_covs, "type": "bar",
            "name": "exp41 baseline", "marker": {"color": "#888"},
            "hovertemplate": "%{x}<br>baseline: %{y:.3f}<extra></extra>",
        },
    ])

    ratio_data = _json.dumps([{
        "x": short_names, "y": ratios, "type": "bar",
        "marker": {"color": ratio_colors},
        "name": "Ratio multitask/baseline",
        "hovertemplate": "%{x}<br>ratio: %{y:.3f}<extra></extra>",
    }])

    layout1 = _json.dumps({
        "title": "Coverage: Multitask vs Baseline",
        "paper_bgcolor": "#1a1a2e", "plot_bgcolor": "#12122b",
        "font": {"color": "#eee"}, "barmode": "group",
        "yaxis": {"title": "coverage_ratio"},
        "xaxis": {"tickangle": -30},
        "margin": {"b": 160},
    })

    layout2 = _json.dumps({
        "title": f"Multitask/Baseline Ratio (gate={_MULTITASK_RATIO_GATE})",
        "paper_bgcolor": "#1a1a2e", "plot_bgcolor": "#12122b",
        "font": {"color": "#eee"},
        "yaxis": {"title": "ratio", "range": [0, max(max(ratios, default=1), 1.5)]},
        "xaxis": {"tickangle": -30},
        "shapes": [{"type": "line", "x0": -0.5, "x1": len(env_ids) - 0.5,
                    "y0": _MULTITASK_RATIO_GATE, "y1": _MULTITASK_RATIO_GATE,
                    "line": {"color": "#fff", "width": 1, "dash": "dash"}}],
        "margin": {"b": 160},
    })

    status_color = "#4eff7a" if multitask_result["passed"] else "#f44336"
    status_text = "PASS" if multitask_result["passed"] else "FAIL"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Stage 18 exp43b — Multitask</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  body {{ background: #1a1a2e; color: #eee; font-family: monospace; margin: 0; padding: 20px; }}
  h1 {{ color: #4a9eff; }}
  .stat {{ display: inline-block; background: #12122b; padding: 12px 24px;
           margin: 8px; border-radius: 8px; border: 1px solid #333; }}
  .stat-value {{ font-size: 2em; color: {status_color}; }}
  #chart1, #chart2 {{ width: 100%; height: 420px; margin-bottom: 20px; }}
</style>
</head>
<body>
<h1>Stage 18 / exp43b — Multitask Training</h1>
<p>1 agent × {_MULTITASK_STEPS:,} steps on random envs. Gate: mean ratio &ge; {_MULTITASK_RATIO_GATE}.</p>
<div class="stat"><div>Mean Ratio</div><div class="stat-value">{multitask_result['mean_ratio']:.3f}</div></div>
<div class="stat"><div>Gate</div><div class="stat-value">{status_text}</div></div>
<div id="chart1"></div>
<div id="chart2"></div>
<script>
  Plotly.newPlot('chart1', {bar_data}, {layout1}, {{responsive: true}});
  Plotly.newPlot('chart2', {ratio_data}, {layout2}, {{responsive: true}});
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------

def run(device: Optional[str] = None) -> dict:
    """Run both 43a (sequential) and 43b (multitask) experiments.

    Args:
        device: PyTorch device string. Auto-detected if None.

    Returns:
        Dict: {passed, passed_43a, passed_43b, continual_results, multitask_result}
    """
    if device is None:
        device = get_device()

    print(f"[exp43] device={device}", flush=True)

    # ---- 43a: parallel chains ----
    print("[exp43a] Running 5 sequential chains in parallel...", flush=True)
    chain_args = [(i, chain, device) for i, chain in enumerate(CHAINS)]
    with multiprocessing.Pool(processes=5) as pool:
        chain_results = pool.map(_run_chain, chain_args)

    # Count passing transitions
    all_transitions = []
    for cr in chain_results:
        for trans_data in cr["transitions"].values():
            all_transitions.append(trans_data["passed"])

    n_transitions = len(all_transitions)
    n_pass_transitions = sum(all_transitions)
    passed_43a = n_pass_transitions >= _RETENTION_MIN_PASS

    print(
        f"[exp43a] {n_pass_transitions}/{n_transitions} transitions passed "
        f"(gate: {_RETENTION_MIN_PASS}/{n_transitions})  passed={passed_43a}",
        flush=True,
    )

    # Save 43a results
    os.makedirs("results", exist_ok=True)
    continual_out = {
        "passed": passed_43a,
        "n_pass_transitions": n_pass_transitions,
        "n_transitions": n_transitions,
        "retention_gate": _RETENTION_GATE,
        "min_pass": _RETENTION_MIN_PASS,
        "chains": chain_results,
    }
    with open("results/exp43_continual.json", "w") as f:
        json.dump(continual_out, f, indent=2)
    print("[exp43a] Results saved to results/exp43_continual.json", flush=True)

    # Build 43a HTML
    os.makedirs("reports", exist_ok=True)
    html_cont = _build_continual_html(chain_results)
    with open("reports/exp43_continual.html", "w") as f:
        f.write(html_cont)
    print("[exp43a] Report saved to reports/exp43_continual.html", flush=True)

    # ---- Load baselines from exp41 ----
    baseline_coverages: dict[str, float] = {}
    exp41_path = "results/exp41_baseline.json"
    if os.path.exists(exp41_path):
        with open(exp41_path) as f:
            exp41_data = json.load(f)
        for r in exp41_data.get("results", []):
            baseline_coverages[r["env_id"]] = r.get("coverage_ratio", 0.1)
        print(f"[exp43b] Loaded {len(baseline_coverages)} baselines from exp41", flush=True)
    else:
        # Fallback: use small constant baseline
        print("[exp43b] exp41 results not found — using fallback baseline 0.1", flush=True)
        baseline_coverages = {env_id: 0.1 for env_id, _, _ in ENVS}

    # ---- 43b: multitask ----
    print("[exp43b] Running multitask training...", flush=True)
    multitask_result = _run_multitask(device, baseline_coverages)
    passed_43b = multitask_result["passed"]

    # Save 43b results
    with open("results/exp43_multitask.json", "w") as f:
        json.dump(multitask_result, f, indent=2)
    print("[exp43b] Results saved to results/exp43_multitask.json", flush=True)

    # Build 43b HTML
    html_multi = _build_multitask_html(multitask_result, baseline_coverages)
    with open("reports/exp43_multitask.html", "w") as f:
        f.write(html_multi)
    print("[exp43b] Report saved to reports/exp43_multitask.html", flush=True)

    # ---- Overall gate ----
    passed = passed_43a and passed_43b
    print(
        f"[exp43] DONE  passed_43a={passed_43a}  passed_43b={passed_43b}  passed={passed}",
        flush=True,
    )

    return {
        "passed": passed,
        "passed_43a": passed_43a,
        "passed_43b": passed_43b,
        "gate_details": {
            "43a_n_pass": n_pass_transitions,
            "43a_n_total": n_transitions,
            "43a_gate": f">= {_RETENTION_MIN_PASS}/{n_transitions} with ratio >= {_RETENTION_GATE}",
            "43b_mean_ratio": multitask_result["mean_ratio"],
            "43b_gate": f">= {_MULTITASK_RATIO_GATE}",
        },
        "continual_results": continual_out,
        "multitask_result": multitask_result,
    }


# ---------------------------------------------------------------------------
# pytest entry point
# ---------------------------------------------------------------------------

def test_exp43_continual_multitask() -> None:
    """Stage 18 / exp43 gate test."""
    result = run()
    assert result["passed"], (
        f"exp43 gate failed: "
        f"43a={result['passed_43a']} (transitions {result['gate_details']['43a_n_pass']}/{result['gate_details']['43a_n_total']}), "
        f"43b={result['passed_43b']} (mean_ratio={result['gate_details']['43b_mean_ratio']:.3f})"
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    if sys.platform == "win32":
        multiprocessing.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(description="Stage 18 exp43 — Continual + Multitask")
    parser.add_argument("--device", default=None, help="cuda or cpu (auto-detected if omitted)")
    args = parser.parse_args()

    # Add src/ to path for imports
    src_dir = os.path.join(os.path.dirname(__file__), "..", "..")
    if src_dir not in sys.path:
        sys.path.insert(0, os.path.abspath(src_dir))

    result = run(device=args.device)
    print(f"\nResult: passed={result['passed']}")
    print(f"  43a: {result['gate_details']['43a_n_pass']}/{result['gate_details']['43a_n_total']} transitions retained")
    print(f"  43b: mean_ratio={result['gate_details']['43b_mean_ratio']:.3f}")
