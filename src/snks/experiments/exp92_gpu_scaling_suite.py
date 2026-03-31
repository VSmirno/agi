"""Experiment 92: GPU Scaling Suite — Full СНКС on AMD ROCm.

Runs on minipc (evo-x2, AMD 96GB VRAM, ROCm 7.2).
Designed for overnight execution (~4-6 hours).

Tests:
  A. DAF throughput sweep: N=50K, 100K, 200K
  B. Embodied agent on large grids: N=50K, grid 12x12 and 16x16, 100 episodes
  C. All stage 25-35 experiments on GPU (device=cuda)
  D. IntegratedAgent full pipeline with N=50K DAF

Results are saved incrementally to results/scaling/ as JSON files.
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# Env setup
# ---------------------------------------------------------------------------
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "scaling"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _save(name: str, data: dict) -> Path:
    path = RESULTS_DIR / f"{name}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  -> saved {path}")
    return path


def _log(msg: str):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ===================================================================
# PART A: DAF Throughput Sweep
# ===================================================================

def run_daf_sweep(device: str = "cuda") -> dict:
    """Measure DAF engine throughput at N=50K, 100K, 200K."""
    from snks.agent.embodied_agent import EmbodiedAgent, EmbodiedAgentConfig
    from snks.daf.types import (
        CausalAgentConfig, ConfiguratorConfig, CostModuleConfig,
        DafConfig, EncoderConfig, HierarchicalConfig, PipelineConfig, SKSConfig,
    )
    from snks.env.causal_grid import make_level

    results = {}

    for n_nodes in [50_000, 100_000, 200_000]:
        _log(f"DAF sweep: N={n_nodes:,} ...")

        try:
            # Measure VRAM before
            vram_before = _get_vram_mb(device)

            t0 = time.perf_counter()
            daf_cfg = DafConfig(
                num_nodes=n_nodes,
                avg_degree=30,
                oscillator_model="fhn",
                dt=0.0001,
                noise_sigma=0.01,
                fhn_I_base=0.5,
                device=device,
                disable_csr=True,
            )
            pipeline_cfg = PipelineConfig(
                daf=daf_cfg,
                encoder=EncoderConfig(),
                sks=SKSConfig(),
                hierarchical=HierarchicalConfig(enabled=True),
                cost_module=CostModuleConfig(enabled=True),
                configurator=ConfiguratorConfig(enabled=True),
                device=device,
                steps_per_cycle=20,
            )
            causal_cfg = CausalAgentConfig(pipeline=pipeline_cfg)
            agent = EmbodiedAgent(EmbodiedAgentConfig(causal=causal_cfg))
            init_time = time.perf_counter() - t0

            vram_after = _get_vram_mb(device)

            # Run 20 episodes on DoorKey-16x16
            env = make_level("DoorKey", size=16, max_steps=100)
            n_steps_total = 0
            t_run = time.perf_counter()

            for ep in range(20):
                obs_raw, _ = env.reset(seed=ep)
                obs = obs_raw["image"] if isinstance(obs_raw, dict) else obs_raw
                done = False
                while not done:
                    action = agent.step(obs)
                    obs_raw, _, terminated, truncated, _ = env.step(action)
                    obs = obs_raw["image"] if isinstance(obs_raw, dict) else obs_raw
                    done = terminated or truncated
                    agent.observe_result(obs)
                    n_steps_total += 1

            run_time = time.perf_counter() - t_run
            steps_per_sec = n_steps_total / run_time if run_time > 0 else 0

            results[f"N{n_nodes}"] = {
                "num_nodes": n_nodes,
                "init_seconds": round(init_time, 1),
                "steps_per_sec": round(steps_per_sec, 2),
                "total_steps": n_steps_total,
                "run_seconds": round(run_time, 1),
                "vram_before_mb": vram_before,
                "vram_after_mb": vram_after,
                "vram_delta_mb": round(vram_after - vram_before, 0) if vram_before and vram_after else None,
                "status": "PASS",
            }
            _log(f"  N={n_nodes:,}: {steps_per_sec:.1f} steps/sec, init={init_time:.0f}s, VRAM={vram_after}MB")

        except Exception as e:
            results[f"N{n_nodes}"] = {
                "num_nodes": n_nodes,
                "status": "FAIL",
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
            _log(f"  N={n_nodes:,}: FAIL — {e}")

        # Free memory
        gc.collect()
        _torch_empty_cache()

    _save("partA_daf_sweep", results)
    return results


# ===================================================================
# PART B: Embodied Agent on Large Grids
# ===================================================================

def run_large_grids(device: str = "cuda") -> dict:
    """N=50K agent on grid 12x12 and 16x16, 100 episodes each."""
    from snks.agent.embodied_agent import EmbodiedAgent, EmbodiedAgentConfig
    from snks.daf.types import (
        CausalAgentConfig, ConfiguratorConfig, CostModuleConfig,
        DafConfig, EncoderConfig, HierarchicalConfig, PipelineConfig, SKSConfig,
    )
    from snks.env.causal_grid import make_level

    results = {}

    for grid_size, max_steps, n_episodes in [(12, 300, 100), (16, 500, 100)]:
        _log(f"Large grid: {grid_size}x{grid_size}, {n_episodes} eps, max_steps={max_steps} ...")

        try:
            daf_cfg = DafConfig(
                num_nodes=50_000, avg_degree=30,
                oscillator_model="fhn", dt=0.0001, noise_sigma=0.01,
                fhn_I_base=0.5, device=device, disable_csr=True,
            )
            pipeline_cfg = PipelineConfig(
                daf=daf_cfg, encoder=EncoderConfig(), sks=SKSConfig(),
                hierarchical=HierarchicalConfig(enabled=True),
                cost_module=CostModuleConfig(enabled=True),
                configurator=ConfiguratorConfig(enabled=True),
                device=device, steps_per_cycle=20,
            )
            causal_cfg = CausalAgentConfig(
                pipeline=pipeline_cfg,
                grid_size=grid_size,
                max_steps_per_episode=max_steps,
            )
            agent = EmbodiedAgent(EmbodiedAgentConfig(causal=causal_cfg))

            env = make_level("DoorKey", size=grid_size, max_steps=max_steps)
            total_steps = 0
            successes = 0
            t_run = time.perf_counter()

            for ep in range(n_episodes):
                obs_raw, _ = env.reset(seed=ep)
                obs = obs_raw["image"] if isinstance(obs_raw, dict) else obs_raw
                done = False
                ep_steps = 0
                while not done:
                    action = agent.step(obs)
                    obs_raw, reward, terminated, truncated, _ = env.step(action)
                    obs = obs_raw["image"] if isinstance(obs_raw, dict) else obs_raw
                    done = terminated or truncated
                    agent.observe_result(obs)
                    ep_steps += 1
                    total_steps += 1
                if terminated and reward > 0:
                    successes += 1

                if (ep + 1) % 25 == 0:
                    elapsed = time.perf_counter() - t_run
                    _log(f"  grid={grid_size}: ep {ep+1}/{n_episodes}, "
                         f"success={successes}/{ep+1}, "
                         f"{total_steps/elapsed:.1f} steps/sec")

            run_time = time.perf_counter() - t_run
            sps = total_steps / run_time if run_time > 0 else 0
            success_rate = successes / n_episodes

            results[f"grid{grid_size}"] = {
                "grid_size": grid_size,
                "max_steps": max_steps,
                "n_episodes": n_episodes,
                "total_steps": total_steps,
                "successes": successes,
                "success_rate": round(success_rate, 3),
                "steps_per_sec": round(sps, 2),
                "run_seconds": round(run_time, 1),
                "status": "DONE",
            }
            _log(f"  grid={grid_size}: DONE — success={success_rate:.1%}, {sps:.1f} steps/sec")

        except Exception as e:
            results[f"grid{grid_size}"] = {
                "grid_size": grid_size,
                "status": "FAIL",
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
            _log(f"  grid={grid_size}: FAIL — {e}")

        gc.collect()
        _torch_empty_cache()

    _save("partB_large_grids", results)
    return results


# ===================================================================
# PART C: All Stage 25-35 Experiments on GPU
# ===================================================================

# Experiment modules — all have main() that prints GATE PASS/FAIL
STAGE_EXP_MODULES = [
    "snks.experiments.exp58_goal_decomposition",
    "snks.experiments.exp59_causal_learning",
    "snks.experiments.exp60_multitrial_success",
    "snks.experiments.exp61_transfer_layouts",
    "snks.experiments.exp62_cross_env_transfer",
    "snks.experiments.exp63_persistence",
    "snks.experiments.exp64_selective_transfer",
    "snks.experiments.exp65_skill_extraction",
    "snks.experiments.exp66_skill_reuse",
    "snks.experiments.exp67_skill_transfer",
    "snks.experiments.exp68_analogy_found",
    "snks.experiments.exp69_analogy_solve",
    "snks.experiments.exp70_regression",
    "snks.experiments.exp71_curiosity_unit",
    "snks.experiments.exp72_curiosity_vs_random",
    "snks.experiments.exp73_curiosity_goal",
    "snks.experiments.exp74_one_shot_skill",
    "snks.experiments.exp75_few_shot_goal",
    "snks.experiments.exp76_few_shot_transfer",
    "snks.experiments.exp77_pattern_rules",
    "snks.experiments.exp78_pattern_completion",
    "snks.experiments.exp79_multi_rule",
    "snks.experiments.exp80_strategy_selection",
    "snks.experiments.exp81_adaptation",
    "snks.experiments.exp82_multi_task",
    "snks.experiments.exp83_concept_transfer",
    "snks.experiments.exp84_multi_agent_speedup",
    "snks.experiments.exp85_cooperative",
    "snks.experiments.exp86_plan_depth",
    "snks.experiments.exp87_replan",
    "snks.experiments.exp88_multi_room",
    "snks.experiments.exp89_capabilities",
    "snks.experiments.exp90_end_to_end",
    "snks.experiments.exp91_full_integration",
]


def run_all_exps_gpu() -> dict:
    """Run all stage 25-35 experiments as subprocesses, capture PASS/FAIL."""
    import subprocess

    results = {}
    passed_count = 0
    failed_count = 0

    for module_path in STAGE_EXP_MODULES:
        name = module_path.rsplit(".", 1)[-1]  # e.g. exp58_goal_decomposition
        short = name.split("_", 1)[0]  # e.g. exp58
        _log(f"Running {short} ({name}) ...")

        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                [sys.executable, "-m", module_path],
                capture_output=True, text=True, timeout=600,  # 10 min per exp
                env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src")},
            )
            elapsed = time.perf_counter() - t0
            output = proc.stdout + proc.stderr

            # Detect PASS/FAIL from output
            is_pass = ("GATE PASS" in output or "GATE: " in output and "PASS" in output
                       or ">>> " in output and "PASS" in output)

            results[short] = {
                "module": name,
                "passed": is_pass,
                "exit_code": proc.returncode,
                "elapsed_seconds": round(elapsed, 1),
                "output_tail": output[-500:] if len(output) > 500 else output,
            }
            if is_pass:
                passed_count += 1
                _log(f"  {short}: PASS ({elapsed:.1f}s)")
            else:
                failed_count += 1
                _log(f"  {short}: FAIL ({elapsed:.1f}s)")

        except subprocess.TimeoutExpired:
            elapsed = time.perf_counter() - t0
            results[short] = {
                "module": name,
                "passed": False,
                "elapsed_seconds": round(elapsed, 1),
                "error": "TIMEOUT (600s)",
            }
            failed_count += 1
            _log(f"  {short}: TIMEOUT ({elapsed:.1f}s)")

        except Exception as e:
            elapsed = time.perf_counter() - t0
            results[short] = {
                "module": name,
                "passed": False,
                "elapsed_seconds": round(elapsed, 1),
                "error": str(e),
            }
            failed_count += 1
            _log(f"  {short}: ERROR ({elapsed:.1f}s) — {e}")

        # Save incrementally every 5 experiments
        if (passed_count + failed_count) % 5 == 0:
            _save("partC_all_exps_gpu_partial", {
                "passed": passed_count,
                "failed": failed_count,
                "total": passed_count + failed_count,
                "experiments": results,
            })

    summary = {
        "passed": passed_count,
        "failed": failed_count,
        "total": len(STAGE_EXP_MODULES),
        "all_pass": failed_count == 0,
        "experiments": results,
    }
    _save("partC_all_exps_gpu", summary)
    return summary


# ===================================================================
# PART D: IntegratedAgent with N=50K DAF
# ===================================================================

def run_integrated_scaling(device: str = "cuda") -> dict:
    """IntegratedAgent full pipeline with scaled DAF."""
    from snks.language.integrated_agent import IntegratedAgent
    from snks.agent.causal_model import CausalLink
    from snks.language.curiosity_module import CuriosityModule
    from snks.language.skill import Skill

    _log("IntegratedAgent scaling test ...")

    results = {}
    try:
        t0 = time.perf_counter()
        agent = IntegratedAgent(agent_id="scaling_test", grid_size=12)
        init_time = time.perf_counter() - t0

        # Inject knowledge (same as exp89)
        links = [
            CausalLink(action=3, context_sks=frozenset({50}),
                        effect_sks=frozenset({51}), strength=0.9, count=5),
            CausalLink(action=5, context_sks=frozenset({51, 52}),
                        effect_sks=frozenset({53}), strength=0.85, count=4),
            CausalLink(action=1, context_sks=frozenset({53, 54}),
                        effect_sks=frozenset({54}), strength=0.8, count=3),
        ]
        agent.inject_knowledge(links)
        agent.inject_skill(Skill(
            name="pickup_key", preconditions=frozenset({50}),
            effects=frozenset({51}), terminal_action=3,
            target_word="key", success_count=10, attempt_count=10,
        ))

        # Build coverage
        for i in range(100):
            key = CuriosityModule.make_key({50 + i % 20}, (i, 0))
            agent.curiosity.observe(key)

        # Test capabilities
        caps = agent.capabilities()
        n_caps = sum(1 for c in caps if c.available)

        # Test strategy pipeline
        t_pipe = time.perf_counter()
        profile = agent.profile_task(goal_sks=frozenset({53}), current_sks=frozenset({50}))
        strategy = agent.select_strategy(profile)
        plan = agent.plan(goal=frozenset({53}), current=frozenset({50}))
        pipe_time = time.perf_counter() - t_pipe

        # Multi-agent test
        agent2 = IntegratedAgent(agent_id="scaling_test_2", grid_size=12)
        agent.share_knowledge(agent2)
        caps2 = agent2.capabilities()
        n_caps2 = sum(1 for c in caps2 if c.available)

        results = {
            "init_seconds": round(init_time, 2),
            "n_capabilities": n_caps,
            "strategy": strategy.name if strategy else None,
            "plan_length": len(plan) if plan else 0,
            "pipeline_seconds": round(pipe_time, 4),
            "multi_agent_caps": n_caps2,
            "knowledge_transferred": n_caps2 > 0,
            "status": "PASS",
        }
        _log(f"  IntegratedAgent: {n_caps} caps, pipeline={pipe_time:.3f}s, multi-agent={n_caps2} caps")

    except Exception as e:
        results = {
            "status": "FAIL",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }
        _log(f"  IntegratedAgent: FAIL — {e}")

    _save("partD_integrated_scaling", results)
    return results


# ===================================================================
# Utilities
# ===================================================================

def _get_vram_mb(device: str) -> float | None:
    try:
        import torch
        if torch.cuda.is_available():
            return round(torch.cuda.memory_allocated(0) / 1024 / 1024, 0)
    except Exception:
        pass
    return None


def _torch_empty_cache():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# ===================================================================
# Main
# ===================================================================

def run_all(device: str = "cuda") -> dict:
    """Run the full scaling suite."""
    _log("=" * 60)
    _log("Experiment 92: GPU Scaling Suite")
    _log(f"Device: {device}")
    _log(f"Results: {RESULTS_DIR}")
    _log("=" * 60)

    all_results = {}
    suite_start = time.perf_counter()

    # Part A: DAF Sweep
    _log("\n=== PART A: DAF Throughput Sweep ===")
    all_results["partA"] = run_daf_sweep(device)

    # Part B: Large Grids
    _log("\n=== PART B: Embodied Agent on Large Grids ===")
    all_results["partB"] = run_large_grids(device)

    # Part C: All experiments on GPU
    _log("\n=== PART C: All Stage 25-35 Experiments on GPU ===")
    all_results["partC"] = run_all_exps_gpu()

    # Part D: IntegratedAgent scaling
    _log("\n=== PART D: IntegratedAgent Scaling ===")
    all_results["partD"] = run_integrated_scaling(device)

    suite_time = time.perf_counter() - suite_start
    all_results["total_seconds"] = round(suite_time, 1)
    all_results["total_hours"] = round(suite_time / 3600, 2)

    _save("exp92_full_results", all_results)

    _log("\n" + "=" * 60)
    _log(f"DONE in {suite_time/3600:.1f} hours")
    _log("=" * 60)

    return all_results


if __name__ == "__main__":
    _device = sys.argv[1] if len(sys.argv) > 1 else "cuda"
    run_all(device=_device)
