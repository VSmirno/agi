# Phase B vs HEAD planner diagnostic — was wandering present before Stage 91?

Date: 2026-05-10
Seed: 17, episode 0, max_steps=220, strict-determinism wrapper.

## Method

**Phase B** = the planner / controller stack at commit `71d1e29`
("feat(stage90r): add emergency safety controller") — i.e. the last commit
before the Stage 91 forensics work that produced HEAD's local_affordance /
feasibility-label / blocked_h / movement / do-facing fixes — **plus** the two
post-71d1e29 determinism patches needed to run under
`torch.use_deterministic_algorithms(True)` on CUDA 12.1 / torch 2.5.1.

**Isolated checkout** (no original modified):
`/opt/cuda/agi-stage91-phaseB-video-20260510T065623Z`, hardlink-cloned from
`/opt/cuda/agi-stage91-feasibility-label-fix-20260507T181856Z`, then:

- **Reverted to 71d1e29 versions:**
  - `experiments/stage90r_eval_local_policy.py` (with `_eval_episode_rng` helper
    re-added — the recorder needs the deterministic 4-line RNG factory; this is
    pure determinism scaffolding, not Stage 91 logic)
  - `src/snks/agent/vector_bootstrap.py`
  - `src/snks/agent/vector_mpc_agent.py`
  - `src/snks/agent/vector_sim.py`
  - `src/snks/agent/stage90r_emergency_controller.py`
  - `src/snks/agent/stage90r_local_policy.py`
- **Deleted** (didn't exist at 71d1e29):
  - `src/snks/agent/stage90r_local_affordances.py`
- **Kept HEAD versions** (determinism patches only):
  - `src/snks/agent/vector_world_model.py` — `dists_flat.cpu().kthvalue(...)`
  - `src/snks/agent/crafter_pixel_env.py` — `_install_crafter_determinism_patch`
- **Kept HEAD** `experiments/record_stage91_seed_video.py`.

**Verification markers (remote checkout):**
- `effective_displacement_h` in vector_mpc_agent.py: 0 ✅ (Stage 91 field absent)
- `_move_target_blocked` in vector_sim.py: 0 ✅ (Stage 91 method absent)
- `dists_flat.cpu().kthvalue` in vector_world_model.py: 1 ✅
- `_install_crafter_determinism_patch` in crafter_pixel_env.py: 2 ✅
- `record_stage91_seed_video.py`: present ✅
- `stage90r_local_affordances.py`: absent ✅

**Run command (HyperPC, gpu 0, strict-determinism env):**
```
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0
PYTHONPATH=$CO/src:$CO:$CO/experiments \
python experiments/record_stage91_seed_video.py \
  --seed 17 --episode-index 0 \
  --local-evaluator /opt/cuda/agi-stage91-feasibility-label-fix-20260507T181856Z/_docs/stage90r_seed7_actor_selection_probe3.pt \
  --max-steps 220 --fps 10 --out _docs/debug_videos/seed17_ep0_phaseB.mp4
```

Artifacts:
- mp4 → `/home/yorick/.ductor/workspace/output_to_user/seed17_ep0_phaseB.mp4`
- json → `_docs/stage91_root_cause_forensics_20260502/transfer/seed17_phaseB_video_metrics.json`

## Phase B numbers

- `episode_steps = 184`, `death_cause = zombie`, frames = 185.
- **Pre-hostile (steps 0..119) controller distribution:**
  `{planner_bootstrap: 120}` — 0 emergency_safety.
- **Pre-hostile plan_origin distribution:**
  - `self:motion_chain:move_left+move_left` — 77
  - `self:move_left` — 37
  - `baseline` — 5
  - `single:tree:do` — 1
- **Pre-hostile action distribution:**
  `move_up: 35, move_down: 33, move_right: 27, move_left: 24, do: 1`.
- **`do` events in pre-hostile phase:** 1 (step 114 → first wood at step 115).
- **First-resource-collected step:** 115 (wood = 1).

**Pre-hostile resource opportunities** (`near_concept ∈ {tree,water,stone,cow,coal,iron,diamond}`):
11 instances, all `near=tree`:

| step | near | action      | plan_origin                                  |
|-----:|:-----|:------------|:---------------------------------------------|
|   17 | tree | move_right  | self:motion_chain:move_left+move_left        |
|   19 | tree | move_up     | self:motion_chain:move_left+move_left        |
|   20 | tree | move_down   | self:motion_chain:move_left+move_left        |
|   21 | tree | move_down   | self:motion_chain:move_left+move_left        |
|   59 | tree | move_up     | self:motion_chain:move_left+move_left        |
|   60 | tree | move_left   | self:motion_chain:move_left+move_left        |
|   61 | tree | move_right  | self:motion_chain:move_left+move_left        |
|  109 | tree | move_left   | self:motion_chain:move_left+move_left        |
|  113 | tree | move_left   | self:move_left                                |
|  114 | tree | **do**      | **single:tree:do**                            |
|  115 | tree | move_down   | self:move_left                                |

10 of 11 tree-adjacent steps the agent moved away rather than acted.

## HEAD comparison

| metric                                | Phase B (71d1e29 + det) | HEAD (de0e54b) |
|---------------------------------------|-------------------------|----------------|
| episode_steps                         | 184                     | 195            |
| death_cause                           | zombie                  | zombie         |
| pre-hostile controllers               | 120 planner_bootstrap   | 120 planner_bootstrap |
| pre-hostile motion_chain L+L          | 77                      | 77             |
| pre-hostile self:move_left            | 37                      | 37             |
| pre-hostile baseline                  | 5                       | 5              |
| pre-hostile single:tree:do            | 1                       | 1              |
| pre-hostile action `do` count         | 1                       | 1              |
| first-resource step (wood)            | 115                     | 115            |
| pre-hostile near=tree opportunities   | 11 (same steps)         | 11 (same steps) |
| **pre-hostile bit-identical steps**   | **120/120**             | —              |
| full-episode `do` total               | 2                       | 9              |
| full-episode emergency_safety         | 27                      | 35             |
| full-episode controllers              | 157 boot / 27 emerg     | 160 boot / 35 emerg |

The pre-hostile trajectory is **bit-identical**: every (action, plan_origin,
player_pos_after) tuple matches across all 120 steps. The eleven tree-adjacent
near-misses occur at the same steps with the same outgoing actions.

Stage 91 fixes do change behaviour, but only **after** hostile contact:
- HEAD survives 11 more steps (195 vs 184).
- HEAD lands 9 `do` actions vs Phase B's 2 (so post-hostile resource
  interactions actually went **up** under Stage 91, not down).
- HEAD spends more time in `emergency_safety` (35 vs 27) — consistent with
  the post-71d1e29 emergency-gating tweaks behaving more aggressively.

## Conclusion

**The pre-Stage 91 wandering hypothesis is confirmed.** On seed 17 ep 0, the
planner_bootstrap stack at 71d1e29 already:
- ignored 10 of 11 tree-adjacent opportunities in the pre-hostile phase,
- locked into `self:motion_chain:move_left+move_left` for 77 of 120 pre-hostile
  steps even when actively next to a tree,
- produced exactly one productive `do` (step 114) before the first hostile.

The Stage 91 fixes (movement / blocked-move / feasibility-label /
local-affordances / do-facing / null-target) did **not** introduce the
wandering — every byte of the pre-hostile trajectory is identical between
71d1e29+determinism and HEAD. The wandering is a **pre-existing
planner-bootstrap / motion-chain ranking issue** that was already there in
the emergency-controller-only baseline; we simply did not have the perception
overlay until Phase A to notice it.

If anything, Stage 91 measurably *improved* post-hostile interaction (9 `do`
vs 2) and survival (+11 steps) on this seed.

The next concrete step is therefore not in the Stage 91 fix layer; it lives
inside `vector_mpc_agent.run_vector_mpc_episode` plan ranking (or its
upstream goal/motion-chain generator). Specifically: when `near_concept ==
tree` and `single:tree:do` is a candidate, the bootstrap motion_chain
`move_left+move_left` is winning by aggregate stimulus score against the
single-step productive plan. That ranking is what to instrument next.
