"""Experiment 98: Curriculum Learning + Adaptive Exploration (Stage 39).

Tests curriculum training mechanisms (epsilon decay, PE exploration,
curriculum progression) and verifies learning improvement.

CPU tests verify mechanisms work correctly.
GPU tests verify absolute performance improvements.

Gates:
    exp98a: curriculum mechanism — scheduler promotes when threshold met
    exp98b: epsilon decay — final_epsilon < initial_epsilon
    exp98c: PE exploration — unique_states > random baseline
    exp98d: curriculum trainer — runs without error on MiniGrid envs
    exp98e: [GPU only] DoorKey-5x5 success_rate >= 0.25
"""

from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

from snks.agent.curriculum import (
    CurriculumScheduler,
    CurriculumStage,
    CurriculumTrainer,
    EpsilonScheduler,
    PredictionErrorExplorer,
)
from snks.agent.pure_daf_agent import PureDafAgent, PureDafConfig


def _detect_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _make_config(n_actions: int = 7, small: bool | None = None) -> PureDafConfig:
    """Create config."""
    cfg = PureDafConfig()
    cfg.n_actions = n_actions
    cfg.max_episode_steps = 100

    if small is None:
        small = _detect_device() == "cpu"

    if small:
        cfg.causal.pipeline.daf.num_nodes = 2000
        cfg.causal.pipeline.daf.avg_degree = 15
        cfg.causal.pipeline.daf.device = "cpu"
        cfg.causal.pipeline.daf.disable_csr = True
        cfg.causal.pipeline.daf.dt = 0.005
        cfg.causal.pipeline.steps_per_cycle = 200
        cfg.causal.pipeline.encoder.image_size = 32
        cfg.causal.pipeline.encoder.pool_h = 5
        cfg.causal.pipeline.encoder.pool_w = 5
        cfg.causal.pipeline.encoder.n_orientations = 4
        cfg.causal.pipeline.encoder.n_frequencies = 2
        cfg.causal.pipeline.encoder.sdr_size = 1600
        cfg.causal.pipeline.sks.min_cluster_size = 3
        cfg.causal.pipeline.sks.coherence_mode = "cofiring"
        cfg.causal.pipeline.sks.top_k = 200
        cfg.causal.motor_sdr_size = 200
        cfg.causal.pipeline.daf.fhn_I_base = 0.3
        cfg.causal.pipeline.daf.coupling_strength = 0.05
    else:
        cfg.causal.pipeline.daf.device = "auto"
        cfg.causal.pipeline.daf.disable_csr = True
        cfg.causal.pipeline.daf.dt = 0.002
        cfg.causal.pipeline.steps_per_cycle = 200
        cfg.causal.pipeline.sks.coherence_mode = "cofiring"
        cfg.causal.pipeline.sks.top_k = 2000
        cfg.causal.pipeline.daf.fhn_I_base = 0.3
        cfg.causal.pipeline.daf.coupling_strength = 0.05

    cfg.exploration_epsilon = 0.7 if small else 0.3
    cfg.reward_scale = 3.0
    cfg.trace_length = 5
    cfg.n_sim_steps = 3 if small else 10
    return cfg


def _make_env(env_name: str):
    """Factory for MiniGrid environments."""
    from snks.env.adapter import MiniGridAdapter
    return MiniGridAdapter(env_name)


def exp98a_curriculum_mechanism():
    """Exp 98a: CurriculumScheduler promotes correctly.

    Gate: promotion happens when success threshold is met.
    """
    print("\n--- Exp 98a: Curriculum Mechanism ---")

    # Test 1: Scheduler promotes after threshold met
    stages = [
        CurriculumStage("stage_a", gate_threshold=0.5, min_episodes=4),
        CurriculumStage("stage_b", gate_threshold=0.3, min_episodes=3),
    ]
    sched = CurriculumScheduler(stages)

    # Record 2 successes, 2 failures = 50% → meets 0.5 threshold
    sched.record_episode(success=True)
    sched.record_episode(success=False)
    sched.record_episode(success=True)
    promoted_early = sched.current_stage_idx > 0
    sched.record_episode(success=True)  # 3/4 = 0.75 ≥ 0.5
    promoted = sched.current_stage_idx == 1

    print(f"  Promoted early (before min_episodes): {promoted_early}")
    print(f"  Promoted after threshold met: {promoted}")

    # Test 2: Doesn't promote below threshold
    sched2 = CurriculumScheduler(stages)
    for _ in range(4):
        sched2.record_episode(success=False)
    not_promoted = sched2.current_stage_idx == 0

    print(f"  Stays at stage 0 when failing: {not_promoted}")

    # Test 3: is_complete works
    stages_one = [CurriculumStage("only", gate_threshold=0.5, min_episodes=2)]
    sched3 = CurriculumScheduler(stages_one)
    sched3.record_episode(success=True)
    sched3.record_episode(success=True)
    is_done = sched3.is_complete
    print(f"  Marks complete at final stage: {is_done}")

    gate_pass = promoted and not_promoted and not promoted_early and is_done
    print(f"  Gate: {'PASS' if gate_pass else 'FAIL'}")

    return {"gate_pass": gate_pass}


def exp98b_epsilon_decay():
    """Exp 98b: Epsilon decay works correctly.

    Gate: final_epsilon < initial_epsilon AND floor is respected
    """
    print("\n--- Exp 98b: Epsilon Decay ---")

    eps = EpsilonScheduler(initial=0.7, decay=0.95, floor=0.1)
    initial = eps.value

    values = [initial]
    for _ in range(50):
        eps.step()
        values.append(eps.value)

    final = eps.value
    monotonic = all(values[i] >= values[i+1] for i in range(len(values)-1))
    floor_ok = final >= 0.1

    print(f"  Initial: {initial:.3f}")
    print(f"  Final:   {final:.3f}")
    print(f"  Monotonic decrease: {monotonic}")
    print(f"  Floor respected:    {floor_ok}")

    gate_pass = final < initial and monotonic and floor_ok
    print(f"  Gate: {'PASS' if gate_pass else 'FAIL'}")

    return {
        "initial_epsilon": initial,
        "final_epsilon": final,
        "gate_pass": gate_pass,
    }


def exp98c_pe_exploration():
    """Exp 98c: PE explorer biases action distribution based on PE.

    Gate: After recording high PE for one action, that action gets
    significantly higher selection probability than uniform.
    """
    print("\n--- Exp 98c: PE Exploration Bias ---")

    n_actions = 5
    pe = PredictionErrorExplorer(n_actions=n_actions, window_size=10, temperature=3.0)

    # Record high PE for action 0, low PE for others
    for _ in range(20):
        pe.record(0, 0.9)
        pe.record(1, 0.1)
        pe.record(2, 0.1)
        pe.record(3, 0.1)
        pe.record(4, 0.1)

    bonuses = pe.action_bonuses()
    print(f"  Bonuses: {[f'{b:.3f}' for b in bonuses]}")
    print(f"  Action 0 bonus: {bonuses[0]:.3f}")
    print(f"  Uniform would be: {1/n_actions:.3f}")

    # Action 0 should be heavily favored
    biased = bonuses[0] > 1.0 / n_actions * 1.5  # at least 1.5x uniform

    # Count empirical selections
    counts = [0] * n_actions
    n_trials = 1000
    for _ in range(n_trials):
        a = pe.select_with_bonus()
        counts[a] += 1

    action_0_ratio = counts[0] / n_trials
    print(f"  Action 0 selected {counts[0]}/{n_trials} = {action_0_ratio:.3f}")
    print(f"  Expected uniform:  {1/n_actions:.3f}")

    selection_biased = action_0_ratio > 1.0 / n_actions * 1.3  # selected 1.3x more than uniform

    gate_pass = biased and selection_biased
    print(f"  Bonus bias: {biased}")
    print(f"  Selection bias: {selection_biased}")
    print(f"  Gate: {'PASS' if gate_pass else 'FAIL'}")

    return {
        "action_0_bonus": bonuses[0],
        "action_0_selection_ratio": action_0_ratio,
        "gate_pass": gate_pass,
    }


def exp98d_trainer_runs():
    """Exp 98d: CurriculumTrainer runs without error on real envs.

    Gate: completes training loop without exceptions
    """
    print("\n--- Exp 98d: Curriculum Trainer Smoke Test ---")

    try:
        from snks.env.adapter import MiniGridAdapter
        _ = MiniGridAdapter("MiniGrid-Empty-5x5-v0")
    except ImportError:
        print("  SKIP: MiniGrid not installed")
        return {"status": "SKIP", "reason": "MiniGrid not installed"}

    cfg = _make_config(n_actions=7)
    stages = [
        CurriculumStage("MiniGrid-Empty-5x5-v0", gate_threshold=0.9, min_episodes=3),
    ]

    errors = []
    try:
        trainer = CurriculumTrainer(
            config=cfg, stages=stages,
            epsilon_initial=0.7, epsilon_decay=0.95,
        )
        # Run 3 episodes — enough to verify pipeline works
        results = []
        env = _make_env("MiniGrid-Empty-5x5-v0")
        for _ in range(3):
            r = trainer.train_episode(env)
            results.append(r)
            print(f"    Episode: steps={r.steps} reward={r.reward:.1f} sks={r.sks_count}")

        # Verify results have sensible structure
        assert all(r.steps > 0 for r in results), "Zero steps"
        assert all(r.sks_count >= 0 for r in results), "Negative SKS"
        assert trainer.epsilon.value < 0.7, "Epsilon didn't decay"

        print(f"  Epsilon: {trainer.epsilon.value:.3f}")
        print(f"  Stats: {trainer.stats}")

    except Exception as e:
        errors.append(str(e))
        print(f"  ERROR: {e}")

    gate_pass = len(errors) == 0
    print(f"  Gate (no errors): {'PASS' if gate_pass else 'FAIL'}")

    return {"errors": errors, "gate_pass": gate_pass}


def exp98e_gpu_doorkey():
    """Exp 98e: [GPU only] Curriculum training improves DoorKey-5x5.

    Gate: success_rate >= 0.25 (only meaningful with 50K nodes on GPU)
    """
    print("\n--- Exp 98e: GPU Curriculum → DoorKey-5x5 ---")

    device = _detect_device()
    if device == "cpu":
        print("  SKIP: GPU required for meaningful success rate test")
        print("  (2K nodes on CPU → success ≈ 0% due to insufficient representation)")
        return {"status": "SKIP", "reason": "GPU required"}

    try:
        from snks.env.adapter import MiniGridAdapter
        _ = MiniGridAdapter("MiniGrid-Empty-5x5-v0")
    except ImportError:
        print("  SKIP: MiniGrid not installed")
        return {"status": "SKIP", "reason": "MiniGrid not installed"}

    cfg = _make_config(n_actions=7, small=False)
    stages = [
        CurriculumStage("MiniGrid-Empty-5x5-v0", gate_threshold=0.3, min_episodes=5),
        CurriculumStage("MiniGrid-Empty-8x8-v0", gate_threshold=0.3, min_episodes=5),
        CurriculumStage("MiniGrid-DoorKey-5x5-v0", gate_threshold=0.15, min_episodes=5),
    ]

    trainer = CurriculumTrainer(
        config=cfg, stages=stages,
        epsilon_initial=0.5, epsilon_decay=0.95, epsilon_floor=0.1,
    )
    results = trainer.train(_make_env, total_episodes=50, max_steps=cfg.max_episode_steps)

    # Evaluate on DoorKey
    eval_env = _make_env("MiniGrid-DoorKey-5x5-v0")
    eval_results = [trainer.train_episode(eval_env) for _ in range(15)]

    successes = sum(1 for r in eval_results if r.success)
    success_rate = successes / len(eval_results)

    print(f"  Curriculum DAF: success={success_rate:.3f} ({successes}/{len(eval_results)})")
    print(f"  Final epsilon:  {trainer.epsilon.value:.3f}")
    print(f"  Gate (>=0.25): {'PASS' if success_rate >= 0.25 else 'FAIL'}")

    return {
        "success_rate": success_rate,
        "gate_pass": success_rate >= 0.25,
    }


def main():
    print("=" * 60)
    print("Experiment 98: Curriculum Learning (Stage 39)")
    print("=" * 60)
    print()

    results = {}

    results["98a"] = exp98a_curriculum_mechanism()
    results["98b"] = exp98b_epsilon_decay()
    results["98c"] = exp98c_pe_exploration()
    results["98d"] = exp98d_trainer_runs()
    results["98e"] = exp98e_gpu_doorkey()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for key, res in results.items():
        status = res.get("status", "")
        if status == "SKIP":
            print(f"  Exp {key}: SKIP — {res.get('reason', '')}")
        else:
            gate = "PASS" if res.get("gate_pass", False) else "FAIL"
            print(f"  Exp {key}: {gate}")

    all_pass = all(
        r.get("gate_pass", False) or r.get("status") == "SKIP"
        for r in results.values()
    )
    print(f"\nOverall: {'ALL GATES PASS' if all_pass else 'SOME GATES FAIL'}")
    return results


if __name__ == "__main__":
    main()
