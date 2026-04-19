# Stage 89 Local Video Diagnostic Design

## Goal

Add a local diagnostic runner that records clean Crafter GUI video so a human can inspect agent behavior directly on the real environment instead of inferring failures only from aggregate metrics.

## Scope

First version stays intentionally narrow:

- record normal Crafter GUI frames only
- no overlays, labels, or HUD additions
- run one seed at a time
- produce paired MP4 outputs for:
  - current `stage89` agent with `exp137` segmenter
  - a trivial comparison policy (`random_move`)

This is a debugging tool, not a benchmark or report generator.

## Approach

Implement a standalone experiment script, `experiments/diag_stage89_video.py`, that:

1. Builds the current Stage 89 stack using the same textbook/model/segmenter path as `stage89_eval.py`.
2. Wraps `CrafterPixelEnv` in a small recording environment that captures `env._env.render((900, 900))`:
   - once after reset
   - after every action step
3. Writes frames to MP4 incrementally.
4. Supports two modes:
   - `agent`
   - `random`
5. Supports a paired run that renders both modes on the same seed to `_docs/debug_videos/`.

## Non-Goals

- no live viewer
- no side-by-side compositing
- no extra agent overlays yet
- no minipc-only workflow; this should run locally

## Success Criteria

- local command produces playable MP4 files
- the `agent` video uses the real current Stage 89 stack
- the `random` video uses the same environment seed and duration
- no changes are required to the core planning loop beyond wrapping the env for recording

## Risks

- local runtime may be slow if full Stage 89 planning is used at long horizons
- MP4 writing may depend on available codec backend; script should fail clearly if the writer backend is unavailable
- recorded frames must reflect the real Crafter GUI render, not the 64x64 tensor input
