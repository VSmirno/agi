"""Local Crafter video diagnostic for Stage 89.

Records clean Crafter GUI video for human inspection on a single seed.

Default output:
  - _docs/debug_videos/seedXXXX_agent.mp4
  - _docs/debug_videos/seedXXXX_random.mp4
  - _docs/debug_videos/seedXXXX_summary.json

This is a debugging tool, not a benchmark runner.
"""

from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

ROOT = Path(__file__).parent.parent
OUT_DIR = ROOT / "_docs" / "debug_videos"


class RecordingCrafterEnv:
    """Wrap CrafterPixelEnv and record real GUI renders after reset/step."""

    def __init__(
        self,
        env,
        video_path: Path,
        fps: int,
        render_size: tuple[int, int],
    ) -> None:
        self._env = env
        self._video_path = video_path
        self._fps = fps
        self._render_size = render_size
        try:
            import imageio_ffmpeg  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "MP4 recording requires imageio-ffmpeg in the active environment. "
                "Install it with: pip install imageio-ffmpeg"
            ) from exc
        self._writer = imageio.get_writer(
            str(video_path),
            format="FFMPEG",
            fps=fps,
            codec="libx264",
            quality=8,
            macro_block_size=1,
        )

    @property
    def n_actions(self) -> int:
        return self._env.n_actions

    @property
    def action_names(self) -> list[str]:
        return self._env.action_names

    def reset(self):
        pixels, info = self._env.reset()
        self._append_frame()
        return pixels, info

    def step(self, action):
        pixels, reward, done, info = self._env.step(action)
        self._append_frame()
        return pixels, reward, done, info

    def observe(self):
        return self._env.observe()

    def close(self) -> None:
        self._writer.close()

    def _append_frame(self) -> None:
        frame = self._env._env.render(self._render_size)
        self._writer.append_data(frame)


def _run_random_episode(env, rng: np.random.RandomState, max_steps: int) -> dict:
    pixels, info = env.reset()
    move_actions = ["move_left", "move_right", "move_up", "move_down"]
    steps = 0
    done = False
    for step in range(max_steps):
        steps = step + 1
        action = str(rng.choice(move_actions))
        pixels, _reward, done, info = env.step(action)
        if done:
            break
    return {
        "episode_steps": steps,
        "death_cause": "alive" if not done else "unknown",
    }


def _build_stage89_agent(
    device: torch.device,
    checkpoint_path: Path,
    crop_world: bool,
    perception_mode: str,
):
    from stage89_eval import _build_model_and_segmenter, _stage89_mode_config
    from snks.agent.crafter_textbook import CrafterTextbook
    from snks.agent.perception import HomeostaticTracker
    from snks.agent.post_mortem import PostMortemLearner

    model_dim = 16384
    n_locations = 50000
    seed = 42
    config = _stage89_mode_config("current")
    textbook_path: Path
    if perception_mode == "pixel":
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                "Pixel perception requested but checkpoint is missing: "
                f"{checkpoint_path}"
            )
        model, segmenter, textbook_path = _build_model_and_segmenter(
            model_dim=model_dim,
            n_locations=n_locations,
            seed=seed,
            device=device,
            checkpoint_path=checkpoint_path,
            crop_world=crop_world,
        )
    elif perception_mode == "symbolic":
        from snks.agent.vector_bootstrap import load_from_textbook
        from snks.agent.vector_world_model import VectorWorldModel

        textbook_path = ROOT / "configs" / "crafter_textbook.yaml"
        model = VectorWorldModel(
            dim=model_dim,
            n_locations=n_locations,
            seed=seed,
            device=device,
        )
        stats = load_from_textbook(model, textbook_path)
        segmenter = None
        print(f"Symbolic perception  Textbook seeded: {stats}")
    else:
        raise ValueError(f"Unknown perception_mode: {perception_mode}")
    tb = CrafterTextbook(str(textbook_path))
    tracker = HomeostaticTracker()
    tracker.init_from_textbook(tb.body_block)
    learner = PostMortemLearner()
    stimuli = learner.build_stimuli(
        ["health", "food", "drink", "energy"],
        include_vital_delta=config["include_vital_delta"],
    )
    return {
        "model": model,
        "segmenter": segmenter,
        "textbook": tb,
        "tracker": tracker,
        "stimuli": stimuli,
        "config": config,
        "perception_mode": perception_mode,
    }


def _run_agent_episode(
    env,
    agent_bundle: dict,
    max_steps: int,
) -> dict:
    from snks.agent.vector_mpc_agent import run_vector_mpc_episode

    return run_vector_mpc_episode(
        env=env,
        segmenter=agent_bundle["segmenter"],
        model=agent_bundle["model"],
        tracker=agent_bundle["tracker"],
        max_steps=max_steps,
        stimuli=agent_bundle["stimuli"],
        textbook=agent_bundle["textbook"],
        verbose=True,
        enable_dynamic_threat_model=agent_bundle["config"]["enable_dynamic_threat_model"],
        enable_dynamic_threat_goals=agent_bundle["config"]["enable_dynamic_threat_goals"],
        enable_motion_plans=agent_bundle["config"]["enable_motion_plans"],
        enable_motion_chains=agent_bundle["config"]["enable_motion_chains"],
        enable_post_plan_passive_rollout=agent_bundle["config"]["enable_post_plan_passive_rollout"],
        perception_mode=agent_bundle["perception_mode"],
    )


def _record_mode(
    mode: str,
    seed: int,
    max_steps: int,
    fps: int,
    render_size: tuple[int, int],
    checkpoint_path: Path,
    crop_world: bool,
    device: torch.device,
    perception_mode: str,
) -> dict:
    from snks.agent.crafter_pixel_env import CrafterPixelEnv

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    video_path = OUT_DIR / f"seed{seed:04d}_{mode}.mp4"
    env = RecordingCrafterEnv(
        CrafterPixelEnv(seed=seed),
        video_path=video_path,
        fps=fps,
        render_size=render_size,
    )
    try:
        if mode == "agent":
            bundle = _build_stage89_agent(
                device=device,
                checkpoint_path=checkpoint_path,
                crop_world=crop_world,
                perception_mode=perception_mode,
            )
            metrics = _run_agent_episode(env=env, agent_bundle=bundle, max_steps=max_steps)
        elif mode == "random":
            rng = np.random.RandomState(seed)
            metrics = _run_random_episode(env=env, rng=rng, max_steps=max_steps)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    finally:
        env.close()

    return {
        "mode": mode,
        "seed": seed,
        "video_path": str(video_path),
        "metrics": {
            "episode_steps": int(metrics.get("episode_steps", 0)),
            "death_cause": metrics.get("death_cause", "unknown"),
            "action_counts": metrics.get("action_counts", {}),
            "arrow_threat_steps": metrics.get("arrow_threat_steps", 0),
            "defensive_action_rate": metrics.get("defensive_action_rate", 0.0),
        },
    }


def main() -> None:
    import argparse
    from snks.encoder.tile_segmenter import pick_device

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--render-width", type=int, default=900)
    parser.add_argument("--render-height", type=int, default=900)
    parser.add_argument(
        "--modes",
        type=str,
        default="agent,random",
        help="Comma-separated subset of: agent,random",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "demos" / "checkpoints" / "exp137" / "segmenter_9x9.pt",
    )
    parser.add_argument("--crop-world", action="store_true", default=True)
    parser.add_argument(
        "--perception-mode",
        choices=("pixel", "symbolic"),
        default="pixel",
    )
    args = parser.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    render_size = (args.render_width, args.render_height)
    device = torch.device(pick_device())

    print(
        f"diag_stage89_video: seed={args.seed} max_steps={args.max_steps} "
        f"fps={args.fps} modes={modes} device={device}"
    )
    results = []
    for mode in modes:
        print(f"recording mode={mode} ...")
        result = _record_mode(
            mode=mode,
            seed=args.seed,
            max_steps=args.max_steps,
            fps=args.fps,
            render_size=render_size,
            checkpoint_path=args.checkpoint,
            crop_world=args.crop_world,
            device=device,
            perception_mode=args.perception_mode,
        )
        results.append(result)
        print(
            f"done mode={mode}: steps={result['metrics']['episode_steps']} "
            f"death={result['metrics']['death_cause']} "
            f"video={result['video_path']}"
        )

    summary_path = OUT_DIR / f"seed{args.seed:04d}_summary.json"
    payload = {
        "seed": args.seed,
        "max_steps": args.max_steps,
        "fps": args.fps,
        "render_size": list(render_size),
        "checkpoint": str(args.checkpoint),
        "crop_world": bool(args.crop_world),
        "perception_mode": args.perception_mode,
        "results": results,
    }
    summary_path.write_text(json.dumps(payload, indent=2))
    print(f"saved summary: {summary_path}")


if __name__ == "__main__":
    main()
