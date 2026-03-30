"""
Inference Viewer — FastAPI server for live MiniGrid inference.

Usage:
    python -m snks.viz.inference_viewer \\
        --checkpoint checkpoints/exp41/MiniGrid-FourRooms-v0/final \\
        --env MiniGrid-FourRooms-v0 \\
        --device cuda \\
        --port 8765
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

STATIC_DIR = Path(__file__).parent / "static"
CHECKPOINTS_ROOT = Path(__file__).resolve().parents[3] / "checkpoints" / "exp41"


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class InferenceState:
    running: bool = False
    step: int = 0
    current_obs: Optional[np.ndarray] = None
    visited: set = field(default_factory=set)
    coverage_history: list = field(default_factory=list)
    env_id: str = "MiniGrid-FourRooms-v0"
    checkpoint_step: str = "final"


_state = InferenceState()
_ws_clients: list[WebSocket] = []
_agent = None          # EmbodiedAgent instance
_env = None            # gymnasium env instance
_loop_task: Optional[asyncio.Task] = None
_device: str = "cuda"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _img(obs) -> np.ndarray:
    """Extract image array from env observation dict."""
    return obs["image"] if isinstance(obs, dict) else obs


def _encode_frame(frame_rgb: np.ndarray) -> str:
    """Encode numpy RGB array as base64 PNG string."""
    from PIL import Image
    img = Image.fromarray(frame_rgb.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _make_env(env_id: str, render: bool = True):
    """Create a MiniGrid env with rgb_array render mode."""
    import minigrid  # noqa: F401 — registers MiniGrid envs
    import gymnasium
    render_mode = "rgb_array" if render else None
    return gymnasium.make(env_id, render_mode=render_mode)


def _build_agent(checkpoint_path: Optional[Path], device: str):
    """Build EmbodiedAgent. If checkpoint_path is given and has a .pt file,
    load the agent state. Otherwise return a fresh agent."""
    from snks.agent.embodied_agent import EmbodiedAgent, EmbodiedAgentConfig
    from snks.daf.types import (
        CausalAgentConfig,
        ConfiguratorConfig,
        CostModuleConfig,
        DafConfig,
        EncoderConfig,
        HierarchicalConfig,
        PipelineConfig,
        SKSConfig,
    )

    daf_cfg = DafConfig(
        num_nodes=5_000,
        avg_degree=30,
        oscillator_model="fhn",
        dt=0.0001,
        noise_sigma=0.01,
        fhn_I_base=0.5,
        device=device,
        disable_csr=(device != "cpu"),
    )
    pipeline_cfg = PipelineConfig(
        daf=daf_cfg,
        encoder=EncoderConfig(),
        sks=SKSConfig(),
        hierarchical=HierarchicalConfig(enabled=True),
        cost_module=CostModuleConfig(enabled=True),
        configurator=ConfiguratorConfig(
            enabled=True,
            explore_epistemic_threshold=-0.01,
            explore_cost_threshold=0.40,
        ),
        steps_per_cycle=20,
        device=device,
    )
    causal_cfg = CausalAgentConfig(pipeline=pipeline_cfg)
    agent = EmbodiedAgent(EmbodiedAgentConfig(causal=causal_cfg))

    # Attempt checkpoint load if path provided
    if checkpoint_path is not None:
        import torch
        ckpt_file = checkpoint_path / "agent.pt"
        if ckpt_file.exists():
            try:
                saved = torch.load(str(ckpt_file), map_location=device)
                if hasattr(agent, "load_state_dict"):
                    agent.load_state_dict(saved)
                    print(f"[inference_viewer] Loaded checkpoint: {ckpt_file}", flush=True)
                else:
                    print(
                        f"[inference_viewer] Checkpoint found at {ckpt_file} but "
                        "agent has no load_state_dict — running with fresh weights.",
                        flush=True,
                    )
            except Exception as exc:
                print(
                    f"[inference_viewer] Warning: failed to load {ckpt_file}: {exc}",
                    flush=True,
                )
        else:
            print(
                f"[inference_viewer] No checkpoint file at {ckpt_file} — "
                "running with fresh agent.",
                flush=True,
            )

    return agent


def _resolve_checkpoint(env_id: str, step: str) -> Optional[Path]:
    """Return checkpoint Path if it exists, else None."""
    path = CHECKPOINTS_ROOT / env_id / step
    return path if path.exists() else None


async def _broadcast(data: dict) -> None:
    """Send JSON to all connected WebSocket clients."""
    dead: list[WebSocket] = []
    for ws in _ws_clients:
        try:
            await ws.send_json(data)
        except Exception:
            dead.append(ws)
    for ws in dead:
        if ws in _ws_clients:
            _ws_clients.remove(ws)


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

async def _agent_loop() -> None:
    """Background task: step agent, broadcast frames."""
    global _agent, _env, _state

    while True:
        if not _state.running:
            await asyncio.sleep(0.05)
            continue

        if _agent is None or _env is None:
            await asyncio.sleep(0.1)
            continue

        if _state.current_obs is None:
            await asyncio.sleep(0.05)
            continue

        loop = asyncio.get_event_loop()

        # --- Run agent step in executor to avoid blocking the event loop ---
        obs_for_agent = _state.current_obs

        def _do_step():
            action = _agent.step(obs_for_agent)
            result = _agent.causal_agent.pipeline.last_cycle_result
            fsm = "neutral"
            if result is not None and result.configurator_action is not None:
                fsm = getattr(result.configurator_action, "mode", "neutral")
            return action, fsm

        try:
            action, fsm = await loop.run_in_executor(None, _do_step)
        except Exception as exc:
            print(f"[inference_viewer] agent.step error: {exc}", flush=True)
            await asyncio.sleep(0.1)
            continue

        # --- Step environment ---
        try:
            obs_next, _, term, trunc, info = _env.step(action)
        except Exception as exc:
            print(f"[inference_viewer] env.step error: {exc}", flush=True)
            await asyncio.sleep(0.1)
            continue

        # observe_result in executor
        obs_next_img = _img(obs_next)
        await loop.run_in_executor(None, _agent.observe_result, obs_next_img)

        # --- Render frame ---
        try:
            frame_rgb = _env.render()
            frame_b64 = _encode_frame(frame_rgb)
        except Exception as exc:
            print(f"[inference_viewer] render error: {exc}", flush=True)
            frame_b64 = ""

        # --- Coverage tracking ---
        pos = info.get("agent_pos") if info else None
        if pos is None:
            try:
                pos = getattr(_env.unwrapped, "agent_pos", None)
            except Exception:
                pos = None
        if pos is not None:
            _state.visited.add(tuple(pos))

        try:
            total_cells = _env.unwrapped.width * _env.unwrapped.height
        except Exception:
            total_cells = 64
        coverage = len(_state.visited) / max(total_cells, 1)
        _state.coverage_history.append(coverage)
        _state.step += 1

        # --- SKS count ---
        sks_count = 0
        try:
            result = _agent.causal_agent.pipeline.last_cycle_result
            if result is not None:
                sks_count = len(result.sks_clusters)
        except Exception:
            pass

        # --- Broadcast ---
        msg = {
            "type": "step",
            "frame": frame_b64,
            "coverage": round(coverage, 4),
            "fsm": str(fsm).upper(),
            "sks_count": sks_count,
            "step": _state.step,
        }
        await _broadcast(msg)

        # --- Episode boundary ---
        if term or trunc:
            try:
                _agent.end_episode()
            except Exception:
                pass
            try:
                obs_reset, _ = _env.reset()
                _state.current_obs = _img(obs_reset)
                _state.visited.clear()
            except Exception as exc:
                print(f"[inference_viewer] env.reset error: {exc}", flush=True)
        else:
            _state.current_obs = obs_next_img

        await asyncio.sleep(0.05)  # ~20 fps cap


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="SNKS Inference Viewer")

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.on_event("startup")
async def _startup():
    global _loop_task
    _loop_task = asyncio.create_task(_agent_loop())


@app.on_event("shutdown")
async def _shutdown():
    global _loop_task
    if _loop_task is not None:
        _loop_task.cancel()
        try:
            await _loop_task
        except asyncio.CancelledError:
            pass
        _loop_task = None


@app.get("/")
async def index():
    """Serve the inference viewer UI."""
    page = STATIC_DIR / "inference.html"
    if page.exists():
        return FileResponse(str(page))
    return JSONResponse({"message": "inference.html not found"}, status_code=404)


@app.get("/checkpoints")
async def list_checkpoints():
    """List all available checkpoint directories under checkpoints/exp41/."""
    if not CHECKPOINTS_ROOT.exists():
        return JSONResponse({"envs": [], "message": "checkpoints/exp41/ not found"})
    envs: dict[str, list[str]] = {}
    for env_dir in sorted(CHECKPOINTS_ROOT.iterdir()):
        if env_dir.is_dir():
            steps = [s.name for s in sorted(env_dir.iterdir()) if s.is_dir()]
            if steps:
                envs[env_dir.name] = steps
    return JSONResponse({"envs": envs})


class LoadRequest(BaseModel):
    env_id: str
    step: str = "final"


@app.post("/load")
async def load_checkpoint(req: LoadRequest):
    """Load agent + env for the given env_id and checkpoint step."""
    global _agent, _env, _state, _loop_task

    was_running = _state.running
    _state.running = False

    # Tear down old env
    if _env is not None:
        try:
            _env.close()
        except Exception:
            pass

    _state.env_id = req.env_id
    _state.checkpoint_step = req.step
    _state.step = 0
    _state.visited = set()
    _state.coverage_history = []
    _state.current_obs = None

    loop = asyncio.get_event_loop()

    def _do_load():
        ckpt_path = _resolve_checkpoint(req.env_id, req.step)
        agent = _build_agent(ckpt_path, _device)
        env = _make_env(req.env_id, render=True)
        obs, _ = env.reset()
        return agent, env, _img(obs)

    try:
        agent, env, first_obs = await loop.run_in_executor(None, _do_load)
    except Exception as exc:
        return JSONResponse(
            {"status": "error", "detail": str(exc)}, status_code=500
        )

    _agent = agent
    _env = env
    _state.current_obs = first_obs
    _state.running = was_running

    return JSONResponse({
        "status": "loaded",
        "env_id": req.env_id,
        "step": req.step,
        "checkpoint_found": _resolve_checkpoint(req.env_id, req.step) is not None,
    })


class ControlRequest(BaseModel):
    action: str  # "play" | "pause" | "step" | "reset"


@app.post("/control")
async def control(req: ControlRequest):
    """Control inference playback."""
    global _agent, _env, _state

    if req.action == "play":
        _state.running = True
        return JSONResponse({"status": "playing"})

    elif req.action == "pause":
        _state.running = False
        return JSONResponse({"status": "paused"})

    elif req.action == "step":
        # Execute a single step regardless of running flag
        if _agent is None or _env is None or _state.current_obs is None:
            return JSONResponse({"status": "error", "detail": "no agent loaded"}, status_code=400)

        was_running = _state.running
        _state.running = False
        await asyncio.sleep(0.01)  # let loop settle

        loop = asyncio.get_event_loop()
        obs_for_agent = _state.current_obs

        def _do_single_step():
            action = _agent.step(obs_for_agent)
            result = _agent.causal_agent.pipeline.last_cycle_result
            fsm = "neutral"
            if result is not None and result.configurator_action is not None:
                fsm = getattr(result.configurator_action, "mode", "neutral")
            return action, fsm

        try:
            action, fsm = await loop.run_in_executor(None, _do_single_step)
            obs_next, _, term, trunc, info = _env.step(action)
            obs_next_img = _img(obs_next)
            await loop.run_in_executor(None, _agent.observe_result, obs_next_img)

            frame_rgb = _env.render()
            frame_b64 = _encode_frame(frame_rgb)

            pos = info.get("agent_pos") if info else None
            if pos is None:
                try:
                    pos = getattr(_env.unwrapped, "agent_pos", None)
                except Exception:
                    pos = None
            if pos is not None:
                _state.visited.add(tuple(pos))
            total_cells = _env.unwrapped.width * _env.unwrapped.height
            coverage = len(_state.visited) / max(total_cells, 1)
            _state.coverage_history.append(coverage)
            _state.step += 1

            sks_count = 0
            try:
                result = _agent.causal_agent.pipeline.last_cycle_result
                if result is not None:
                    sks_count = len(result.sks_clusters)
            except Exception:
                pass

            msg = {
                "type": "step",
                "frame": frame_b64,
                "coverage": round(coverage, 4),
                "fsm": str(fsm).upper(),
                "sks_count": sks_count,
                "step": _state.step,
            }
            await _broadcast(msg)

            if term or trunc:
                _agent.end_episode()
                obs_reset, _ = _env.reset()
                _state.current_obs = _img(obs_reset)
                _state.visited.clear()
            else:
                _state.current_obs = obs_next_img

            _state.running = was_running
            return JSONResponse({"status": "stepped", "step": _state.step})

        except Exception as exc:
            _state.running = was_running
            return JSONResponse({"status": "error", "detail": str(exc)}, status_code=500)

    elif req.action == "reset":
        if _env is None:
            return JSONResponse({"status": "error", "detail": "no env loaded"}, status_code=400)
        was_running = _state.running
        _state.running = False
        try:
            if _agent is not None:
                _agent.end_episode()
        except Exception:
            pass
        obs, _ = _env.reset()
        _state.current_obs = _img(obs)
        _state.step = 0
        _state.visited = set()
        _state.coverage_history = []
        _state.running = was_running
        return JSONResponse({"status": "reset"})

    return JSONResponse({"status": "error", "detail": f"unknown action: {req.action}"}, status_code=400)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint: receive live step updates."""
    await websocket.accept()
    _ws_clients.append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in _ws_clients:
            _ws_clients.remove(websocket)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="SNKS Inference Viewer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Path to checkpoint directory (e.g. checkpoints/exp41/"
            "MiniGrid-FourRooms-v0/final). Relative paths resolved from project root."
        ),
    )
    parser.add_argument(
        "--env",
        default="MiniGrid-FourRooms-v0",
        dest="env_id",
        help="Gymnasium env ID to load.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="PyTorch device: cuda, cpu, rocm.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="HTTP/WS port to listen on.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    global _agent, _env, _state, _device

    args = _parse_args(argv)
    _device = args.device

    # Pre-load agent + env if checkpoint provided
    if args.checkpoint is not None:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.is_absolute():
            project_root = Path(__file__).resolve().parents[3]
            ckpt_path = project_root / ckpt_path

        print(f"[inference_viewer] Loading env={args.env_id} device={args.device}", flush=True)
        try:
            _agent = _build_agent(ckpt_path if ckpt_path.exists() else None, args.device)
            _env = _make_env(args.env_id, render=True)
            obs, _ = _env.reset()
            _state.current_obs = _img(obs)
            _state.env_id = args.env_id
            print("[inference_viewer] Agent and env ready.", flush=True)
        except Exception as exc:
            print(f"[inference_viewer] Pre-load failed: {exc}", flush=True)
            print("[inference_viewer] Starting without agent — use POST /load.", flush=True)
    else:
        print("[inference_viewer] No --checkpoint provided. Use POST /load to start.", flush=True)

    print(f"[inference_viewer] Serving on http://0.0.0.0:{args.port}", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
