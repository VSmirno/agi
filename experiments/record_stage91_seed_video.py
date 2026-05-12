"""Record a Crafter video for one seed/episode under the Stage 91 mixed_control_rescue path,
with optional perception overlay.

Outputs:
  - <out>.mp4: video with optional overlay (semantic labels per tile, status strip per step)
  - <out>.json: full metrics dict (rescue_trace, local_trace, controller_distribution, etc.)

Determinism env vars MUST be set before launch:
  CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")
torch.use_deterministic_algorithms(True)

ROOT = Path(__file__).parent.parent

SEMANTIC_NAMES = {
    0: "?", 1: "wat", 2: "gr", 3: "stn", 4: "pth", 5: "snd",
    6: "TR", 7: "lva", 8: "COA", 9: "IRN", 10: "DIA",
    11: "tbl", 12: "frn",
    13: "P", 14: "cow", 15: "ZB", 16: "SK", 17: "AR",
}
HOSTILE_IDS = {15, 16, 17}
RESOURCE_IDS = {1, 6, 8, 9, 10, 14}
BLOCKER_IDS = {3, 7}  # stone, lava


class FrameCapturingEnv:
    """Wrap CrafterPixelEnv; capture per-step (frame, info_dict) pairs in memory."""

    def __init__(self, env, render_size: tuple[int, int]) -> None:
        self._env = env
        self._render_size = render_size
        self.frames: list[np.ndarray] = []
        self.infos: list[dict] = []
        self.actions: list[str | None] = []

    @property
    def n_actions(self) -> int:
        return self._env.n_actions

    @property
    def action_names(self) -> list[str]:
        return self._env.action_names

    def reset(self):
        out = self._env.reset()
        self._capture(action=None)
        return out

    def step(self, action):
        out = self._env.step(action)
        action_name = action if isinstance(action, str) else self._env.action_names[int(action)]
        self._capture(action=action_name)
        return out

    def observe(self):
        return self._env.observe()

    def _capture(self, action: str | None) -> None:
        frame = np.array(self._env._env.render(self._render_size))
        info = self._env._last_info or {}
        self.frames.append(frame)
        self.infos.append({
            "player_pos": list(map(int, info.get("player_pos", [0, 0]))),
            "semantic": np.asarray(info.get("semantic", np.zeros((1, 1)))).copy(),
            "inventory": dict(info.get("inventory", {})),
        })
        self.actions.append(action)


def _draw_overlay(
    frame: np.ndarray,
    info: dict,
    action: str | None,
    step_idx: int,
    rescue_event: dict | None,
    local_trace_step: dict | None,
) -> np.ndarray:
    """Compose perception overlay onto a single rendered frame."""
    from PIL import Image, ImageDraw, ImageFont

    h, w = frame.shape[:2]
    # Crafter renders world tiles in the TOP 7 rows of the 9-row visible area;
    # the BOTTOM 2 rows are inventory HUD (post-transpose `local_view | item_view`).
    # So the world view is 9 wide × 7 tall, with the player at (vi=4, vj=3).
    view_w, view_h_world = 9, 7
    tile_w = w // 9
    tile_h = w // 9  # tiles are square, 100px each in a 900x900 render
    world_h_px = view_h_world * tile_h  # 700 of 900

    # Extend the canvas to give the status strip its own area below the HUD.
    strip_h = 80
    canvas = np.zeros((h + strip_h, w, 3), dtype=np.uint8)
    canvas[:h] = frame
    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img, "RGBA")

    semantic = np.asarray(info.get("semantic"))
    px, py = info.get("player_pos", [0, 0])

    try:
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 14)
        font_med = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 18)
    except Exception:
        font_small = ImageFont.load_default()
        font_med = ImageFont.load_default()

    if semantic.ndim == 2 and semantic.size > 1:
        # For each visible tile, look up world coord and draw semantic label.
        # Player sits at view-center (vi=4, vj=3) of the 9w×7h world grid.
        center_vi, center_vj = 4, 3
        for vi in range(view_w):
            for vj in range(view_h_world):
                wx = px + (vi - center_vi)
                wy = py + (vj - center_vj)
                if not (0 <= wx < semantic.shape[0] and 0 <= wy < semantic.shape[1]):
                    continue
                tile_id = int(semantic[wx, wy])
                # Skip player's tile (already visible).
                if vi == center_vi and vj == center_vj:
                    continue
                label = SEMANTIC_NAMES.get(tile_id, str(tile_id))
                # Pixel coords for tile top-left corner.
                tx = vi * tile_w
                ty = vj * tile_h
                # Semi-transparent backdrop so text reads over varied tile pixels.
                if tile_id in HOSTILE_IDS:
                    bg = (255, 0, 0, 200)
                    fg = (255, 255, 255, 255)
                elif tile_id in RESOURCE_IDS:
                    bg = (0, 200, 0, 180)
                    fg = (255, 255, 255, 255)
                elif tile_id in BLOCKER_IDS:
                    bg = (60, 60, 60, 180)
                    fg = (255, 255, 255, 255)
                else:
                    bg = (0, 0, 0, 130)
                    fg = (255, 255, 200, 255)
                # Draw label in top-left of tile.
                pad = 2
                draw.rectangle(
                    [tx + pad, ty + pad, tx + pad + tile_w // 2, ty + pad + 18],
                    fill=bg,
                )
                draw.text((tx + pad + 2, ty + pad), label, fill=fg, font=font_small)

        # Highlight hostile tiles with thick red border (world rows only).
        for vi in range(view_w):
            for vj in range(view_h_world):
                wx = px + (vi - center_vi)
                wy = py + (vj - center_vj)
                if not (0 <= wx < semantic.shape[0] and 0 <= wy < semantic.shape[1]):
                    continue
                if int(semantic[wx, wy]) in HOSTILE_IDS:
                    tx = vi * tile_w
                    ty = vj * tile_h
                    draw.rectangle(
                        [tx, ty, tx + tile_w - 1, ty + tile_h - 1],
                        outline=(255, 0, 0, 255), width=4,
                    )

    # Status strip in the extra area below the world+HUD rows.
    strip_top = h
    draw.rectangle([0, strip_top, w, h + strip_h], fill=(0, 0, 0, 255))

    inv = info.get("inventory", {})
    health = inv.get("health", "?")
    food = inv.get("food", "?")
    drink = inv.get("drink", "?")
    energy = inv.get("energy", "?")
    line1 = f"step {step_idx:3d}  action: {action or '-':<12}  pos {px},{py}  H{health} F{food} W{drink} E{energy}"
    draw.text((8, strip_top + 4), line1, fill=(255, 255, 255), font=font_med)

    line2 = ""
    if local_trace_step is not None:
        ctrl = local_trace_step.get("controller", "?")
        line2 = f"controller: {ctrl}"
    if rescue_event is not None:
        trig = rescue_event.get("trigger", "?")
        src = rescue_event.get("override_source", rescue_event.get("rescue_policy", "?"))
        line2 += f"   RESCUE trig={trig} src={src}"
    if line2:
        draw.text((8, strip_top + 28), line2, fill=(255, 220, 100), font=font_small)

    # Per-direction blocked_h prediction badges (top of strip, right side).
    if local_trace_step is not None:
        cfo = local_trace_step.get("local_counterfactual_outcomes") or local_trace_step.get("counterfactual_outcomes")
        if cfo is not None:
            move_actions = ["move_left", "move_right", "move_up", "move_down"]
            outcome_by_action = {str(o.get("action")): o for o in cfo}
            x_off = w - 4
            for act in reversed(move_actions):
                o = outcome_by_action.get(act)
                if o is None:
                    continue
                label_blocked = bool(o.get("label", {}).get("blocked_h", False))
                color = (255, 80, 80) if label_blocked else (80, 220, 80)
                short = {"move_left": "L", "move_right": "R", "move_up": "U", "move_down": "D"}[act]
                badge = f"[{short}]"
                bbox = draw.textbbox((0, 0), badge, font=font_med)
                bw = bbox[2] - bbox[0]
                x_off -= bw + 6
                draw.text((x_off, strip_top + 50), badge, fill=color, font=font_med)

    return np.array(img)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=220)
    parser.add_argument("--local-evaluator", type=Path, required=True)
    parser.add_argument("--perception-mode", default="symbolic")
    parser.add_argument("--actor-share", type=float, default=0.0)
    parser.add_argument("--enable-planner-rescue", action="store_true", default=True)
    parser.add_argument("--terminal-trace-steps", type=int, default=32)
    parser.add_argument("--max-explanations-per-episode", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--render-width", type=int, default=900)
    parser.add_argument("--render-height", type=int, default=900)
    parser.add_argument("--no-overlay", action="store_true",
                        help="Skip perception overlay; output plain mp4 only")
    parser.add_argument("--full-profile", action="store_true",
                        help="Use full SDM profile (16384/50000) instead of smoke-lite (2048/5000). Slower, but planning is reliable.")
    parser.add_argument("--enable-outcome-learning", action="store_true",
                        help="Enable cross-episode outcome-role learning (PCCS step 1). Off by default.")
    parser.add_argument("--world-model-path", type=Path, default=None,
                        help="Path to load/save VectorWorldModel.save() snapshot for this seed. Required for cross-episode persistence.")
    parser.add_argument("--outcome-horizon", type=int, default=5,
                        help="Env steps after each decision before its outcome is written back to the world model.")
    parser.add_argument("--outcome-weight", type=float, default=1.0,
                        help="OutcomeStimulus weight in score_trajectory.base.")
    args = parser.parse_args()

    from snks.agent.crafter_pixel_env import CrafterPixelEnv
    from snks.agent.perception import HomeostaticTracker
    from snks.agent.stage90r_local_model import load_local_evaluator_artifact
    from snks.agent.vector_mpc_agent import run_vector_mpc_episode
    from stage90_quick_slice import _build_runtime
    from stage90r_eval_local_policy import (
        _device, _enforce_offline_gate, _eval_episode_rng, _runtime_profile,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    device = _device()

    evaluator, evaluator_artifact = load_local_evaluator_artifact(args.local_evaluator, device=device)
    _enforce_offline_gate(
        mode="mixed_control_rescue",
        offline_gate=evaluator_artifact.get("offline_gate"),
        allow_override=False,
    )
    runtime_profile = _runtime_profile(smoke_lite=not args.full_profile)
    model, segmenter, tb, _, runtime = _build_runtime(
        seed=args.seed,
        checkpoint=None,
        crop_world=True,
        model_dim=int(runtime_profile["model_dim"]),
        n_locations=int(runtime_profile["n_locations"]),
    )
    config = runtime["config"]
    stimuli = runtime["stimuli"]

    ep = args.episode_index
    env_seed = args.seed + ep
    base_env = CrafterPixelEnv(seed=env_seed)
    cap_env = FrameCapturingEnv(base_env, render_size=(args.render_width, args.render_height))
    tracker = HomeostaticTracker()
    tracker.init_from_textbook(tb.body_block)

    print(f"recording: env_seed={env_seed} max_steps={args.max_steps} fps={args.fps}")

    metrics = run_vector_mpc_episode(
        env=cap_env,
        segmenter=segmenter,
        model=model,
        tracker=tracker,
        max_steps=args.max_steps,
        horizon=int(runtime_profile["planner_horizon"]),
        beam_width=int(runtime_profile["beam_width"]),
        max_depth=int(runtime_profile["max_depth"]),
        stimuli=stimuli,
        textbook=tb,
        verbose=False,
        enable_dynamic_threat_model=config["enable_dynamic_threat_model"],
        enable_dynamic_threat_goals=config["enable_dynamic_threat_goals"],
        enable_motion_plans=config["enable_motion_plans"],
        enable_motion_chains=config["enable_motion_chains"],
        enable_post_plan_passive_rollout=bool(runtime_profile["enable_post_plan_passive_rollout"]),
        perception_mode=args.perception_mode,
        local_actor_policy=evaluator,
        local_advisory_device=device,
        mixed_control_actor_share=float(args.actor_share),
        enable_planner_rescue=bool(args.enable_planner_rescue),
        enable_outcome_learning=bool(args.enable_outcome_learning),
        world_model_path=args.world_model_path,
        outcome_horizon=int(args.outcome_horizon),
        outcome_stimulus_weight=float(args.outcome_weight),
        record_death_bundle=True,
        record_local_trace=True,
        record_local_counterfactuals="salient_only",
        local_counterfactual_horizon=1,
        death_capture_steps=max(int(args.terminal_trace_steps), int(args.max_explanations_per_episode)),
        rng=_eval_episode_rng(base_seed=args.seed, episode_index=ep),
    )

    print(
        f"done: episode_steps={metrics.get('episode_steps')} "
        f"death={metrics.get('death_cause')}  frames={len(cap_env.frames)}"
    )

    # Build per-step lookups from metrics.
    local_trace = list(metrics.get("local_trace") or [])
    local_trace_by_step: dict[int, dict] = {}
    for entry in local_trace:
        s = entry.get("step")
        if s is not None:
            local_trace_by_step[int(s)] = entry
    rescue_trace = list(metrics.get("rescue_trace") or [])
    rescue_by_step: dict[int, dict] = {}
    for ev in rescue_trace:
        s = ev.get("step")
        if s is not None:
            rescue_by_step[int(s)] = ev

    writer = imageio.get_writer(
        str(args.out), format="FFMPEG", fps=args.fps,
        codec="libx264", quality=8, macro_block_size=1,
    )
    try:
        for i, (frame, info) in enumerate(zip(cap_env.frames, cap_env.infos)):
            if args.no_overlay:
                writer.append_data(frame)
                continue
            rendered = _draw_overlay(
                frame=frame,
                info=info,
                action=cap_env.actions[i],
                step_idx=i,
                rescue_event=rescue_by_step.get(i),
                local_trace_step=local_trace_by_step.get(i),
            )
            writer.append_data(rendered)
    finally:
        writer.close()

    # Save metrics JSON next to the mp4.
    metrics_path = args.out.with_suffix(".json")
    serialisable = {
        "episode_steps": int(metrics.get("episode_steps", 0)),
        "death_cause": metrics.get("death_cause"),
        "controller_distribution": dict(metrics.get("controller_distribution", {})),
        "n_rescue_events": len(rescue_trace),
        "rescue_trace": rescue_trace,
        "local_trace": local_trace,
        "death_trace_bundle": metrics.get("death_trace_bundle", {}),
    }

    def _default(obj):
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return str(obj)

    metrics_path.write_text(json.dumps(serialisable, indent=2, default=_default))
    print(f"video: {args.out}")
    print(f"metrics: {metrics_path}")


if __name__ == "__main__":
    main()
