# Stage91 Rescue Robustness Root-Cause Findings

## Inspected Artifact Paths

- Remote run root: `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_rescue_robustness_20260502T113506Z`
- Commands:
  - `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_rescue_robustness_20260502T113506Z/commands/bounded_compare_seed7.cmd`
  - `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_rescue_robustness_20260502T113506Z/commands/eval_seed_7.cmd`
  - `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_rescue_robustness_20260502T113506Z/commands/eval_seed_17.cmd`
  - `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_rescue_robustness_20260502T113506Z/commands/eval_seed_27.cmd`
- Logs:
  - `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_rescue_robustness_20260502T113506Z/logs/runner.log`
  - `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_rescue_robustness_20260502T113506Z/logs/seed7_bounded_eval.log`
  - `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_rescue_robustness_20260502T113506Z/logs/seed_7_eval.log`
  - `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_rescue_robustness_20260502T113506Z/logs/seed_17_eval.log`
  - `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_rescue_robustness_20260502T113506Z/logs/seed_27_eval.log`
- Raw eval JSON:
  - `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_rescue_robustness_20260502T113506Z/raw/seed7_bounded_eval.json`
  - `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_rescue_robustness_20260502T113506Z/raw/seed_7_eval.json`
  - `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_rescue_robustness_20260502T113506Z/raw/seed_17_eval.json`
  - `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_rescue_robustness_20260502T113506Z/raw/seed_27_eval.json`
  - `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_rescue_robustness_20260502T113506Z/raw/seed_37_eval.json`
  - `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_rescue_robustness_20260502T113506Z/raw/seed_47_eval.json`
- Summary:
  - `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_rescue_robustness_20260502T113506Z/summaries/multiseed_compare_summary.json`
- Remote forensic output created during this investigation:
  - `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/hyper_stage91_rescue_robustness_20260502T113506Z/forensics_root_cause_20260502/compact_report.json`

## Findings

1. Bounded seed 7 and multiseed seed 7 used the same evaluation command shape; the only argument difference is the output filename.
   Evidence:
   - Local mirror of the bounded command [bounded_compare_seed7.cmd](/home/yorick/agi-stage90r-world-model-guardrails/_docs/stage91_root_cause_forensics_20260502/artifacts/commands/bounded_compare_seed7.cmd:1)
   - Local mirror of the multiseed seed 7 command [eval_seed_7.cmd](/home/yorick/agi-stage90r-world-model-guardrails/_docs/stage91_root_cause_forensics_20260502/artifacts/commands/eval_seed_7.cmd:1)
   - Both run `experiments/stage90r_eval_local_policy.py --mode mixed_control_rescue --n-episodes 4 --max-steps 220 --seed 7 --perception-mode symbolic --local-evaluator ...stage90r_seed7_actor_selection_probe3.pt --enable-planner-rescue --smoke-lite`.

2. The bounded-vs-multiseed seed 7 mismatch is a real behavioral difference in the raw eval payloads, not a summary bug.
   Evidence:
   - Bounded seed 7 log shows episode lengths `172, 220, 162, 220` and rescue counts `68, 164, 32, 122` [seed7_bounded_eval.log](/home/yorick/agi-stage90r-world-model-guardrails/_docs/stage91_root_cause_forensics_20260502/artifacts/logs/seed7_bounded_eval.log:1).
   - Multiseed seed 7 log shows episode lengths `78, 182, 220, 45` and rescue counts `3, 40, 52, 19` [seed_7_eval.log](/home/yorick/agi-stage90r-world-model-guardrails/_docs/stage91_root_cause_forensics_20260502/artifacts/logs/seed_7_eval.log:1).
   - Raw JSON summary values extracted from those files:
     - bounded seed 7: `avg_survival=193.5`, `rescue_rate=0.499`, `learner_control_fraction=0.185`, `controller_distribution.emergency_safety=386`
     - multiseed seed 7: `avg_survival=131.25`, `rescue_rate=0.217`, `learner_control_fraction=0.305`, `controller_distribution.emergency_safety=114`
   - The multiseed top summary matches the raw per-seed values, so the summary path is not inventing the regression:
     - seed 7 `131.25`, seed 17 `199.75`, seed 27 `143.0`, seed 37 `171.25`, seed 47 `173.0`

3. The weaker seeds are characterized by fewer emergency-safety interventions and lower rescue rates, not by hostile deaths without rescue.
   Evidence:
   - Seed 17 raw summary: `avg_survival=199.75`, `rescue_rate=0.404`, `learner_control_fraction=0.224`, `emergency_safety=323`, death causes `arrow=2,zombie=1,skeleton=1`.
   - Seed 27 raw summary: `avg_survival=143.0`, `rescue_rate=0.327`, `learner_control_fraction=0.243`, `emergency_safety=187`, death causes `arrow=1,zombie=1,skeleton=1,dehydration=1`.
   - Seed 7 raw summary: `avg_survival=131.25`, `rescue_rate=0.217`, `learner_control_fraction=0.305`, `emergency_safety=114`, death causes `unknown=2,skeleton=1,zombie=1`.
   - All of these runs still report `early_hostile_deaths_without_rescue=0` and `hostile_deaths_without_rescue=0`.

4. The evaluation path contains an unseeded RNG exactly at the actor/planner arbitration point, so identical eval commands can produce different control trajectories.
   Evidence:
   - `run_vector_mpc_episode()` creates a fresh RNG with `np.random.RandomState()` whenever no RNG is passed [vector_mpc_agent.py](/home/yorick/agi-stage90r-world-model-guardrails/src/snks/agent/vector_mpc_agent.py:634).
   - The mixed-control actor handoff uses that RNG on every step: `if actor_ranked and float(rng.rand()) < float(mixed_control_actor_share): ... control_origin = "learner_actor"` [vector_mpc_agent.py](/home/yorick/agi-stage90r-world-model-guardrails/src/snks/agent/vector_mpc_agent.py:1004).
   - The Stage91 eval does not pass an RNG into `run_vector_mpc_episode()`; it only seeds the environment and runtime [stage90r_eval_local_policy.py](/home/yorick/agi-stage90r-world-model-guardrails/experiments/stage90r_eval_local_policy.py:620) and [stage90r_eval_local_policy.py](/home/yorick/agi-stage90r-world-model-guardrails/experiments/stage90r_eval_local_policy.py:623).

5. The saved per-episode traces are truncated to the first 8 local steps and first 8 rescue events, so they are useful for early rescue-pattern comparison but not for terminal-step diagnosis of the `unknown` deaths.
   Evidence:
   - `rescue_trace` and `local_trace_excerpt` are both sliced with `[: args.max_explanations_per_episode]` [stage90r_eval_local_policy.py](/home/yorick/agi-stage90r-world-model-guardrails/experiments/stage90r_eval_local_policy.py:683) and [stage90r_eval_local_policy.py](/home/yorick/agi-stage90r-world-model-guardrails/experiments/stage90r_eval_local_policy.py:684).
   - `--max-explanations-per-episode` defaults to `8` [stage90r_eval_local_policy.py](/home/yorick/agi-stage90r-world-model-guardrails/experiments/stage90r_eval_local_policy.py:210).

## Best-Supported Hypothesis

The Stage91 regression below the frozen `9083357` baseline is primarily an evaluation reproducibility failure caused by unseeded mixed-control arbitration, not a summary bug and not evidence that the emergency controller failed to prevent hostile deaths without rescue.

- Identical seed-7 commands produced materially different raw behavior.
- The raw behavior differences line up with step-level controller allocation changes: weak runs give the learner actor control more often and trigger emergency safety much less often.
- The code path has an explicit unseeded RNG at the exact arbitration site responsible for those control-allocation changes.

## Minimal Next Step

Make the mixed-control RNG deterministic for evaluation only by plumbing a seeded `rng` from `args.seed` into `run_vector_mpc_episode()`, then rerun just seed 7 twice on HyperPC with distinct output paths.

- Expected confirmation signal: both reruns match each other on `avg_survival`, rescue counts, and controller distribution, and the bounded-vs-multiseed seed-7 discrepancy disappears.
