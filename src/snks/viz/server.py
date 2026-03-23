"""FastAPI + WebSocket server for real-time SNKS dashboard.

Endpoints:
    GET  /          → serves static/index.html
    GET  /status    → current pipeline state (JSON)
    POST /start     → start continuous pipeline loop
    POST /pause     → pause pipeline loop
    POST /step      → run one perception cycle
    WS   /ws        → real-time cycle updates
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from snks.daf.types import PipelineConfig, DcamConfig
from snks.dcam.world_model import DcamWorldModel
from snks.pipeline.config import load_config
from snks.pipeline.runner import Pipeline, CycleResult

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class ServerState:
    running: bool = False
    cycle_count: int = 0
    fps: float = 0.0
    last_result: dict = field(default_factory=dict)
    dcam_episodes: int = 0


# ---------------------------------------------------------------------------
# Globals (initialised lazily via _ensure_pipeline)
# ---------------------------------------------------------------------------

_pipeline: Pipeline | None = None
_dcam: DcamWorldModel | None = None
_state = ServerState()
_ws_clients: list[WebSocket] = []
_loop_task: asyncio.Task | None = None
_config_override: PipelineConfig | None = None  # for testing

STATIC_DIR = Path(__file__).parent / "static"
CONFIGS_DIR = Path(__file__).resolve().parents[3] / "configs"


def _build_pipeline() -> tuple[Pipeline, DcamWorldModel]:
    """Create Pipeline + DcamWorldModel from config."""
    if _config_override is not None:
        config = _config_override
    else:
        config_name = os.environ.get("SNKS_CONFIG", "small")
        config_path = CONFIGS_DIR / f"{config_name}.yaml"
        if config_path.exists():
            config = load_config(config_path)
        else:
            config = PipelineConfig(steps_per_cycle=50)

    pipeline = Pipeline(config)
    dcam = DcamWorldModel(config.dcam, device=torch.device("cpu"))
    return pipeline, dcam


def _ensure_pipeline() -> tuple[Pipeline, DcamWorldModel]:
    global _pipeline, _dcam
    if _pipeline is None:
        _pipeline, _dcam = _build_pipeline()
    return _pipeline, _dcam


def _generate_stimulus(size: int = 64) -> torch.Tensor:
    """Generate a random grating-like stimulus."""
    x = torch.linspace(0, 4 * 3.14159, size)
    freq = torch.rand(1).item() * 3 + 1
    angle = torch.rand(1).item() * 3.14159
    grating = torch.sin(freq * (x.unsqueeze(0) * torch.cos(torch.tensor(angle))
                                + x.unsqueeze(1) * torch.sin(torch.tensor(angle))))
    # Normalise to [0, 1]
    grating = (grating - grating.min()) / (grating.max() - grating.min() + 1e-8)
    return grating


def _cycle_to_json(result: CycleResult, state: ServerState) -> dict:
    """Convert CycleResult to JSON-serialisable dict."""
    # Top-K active nodes from SKS clusters
    top_nodes = []
    for cid, nodes in result.sks_clusters.items():
        for nid in sorted(nodes)[:10]:  # top 10 per cluster
            top_nodes.append({"id": nid, "cluster": cid})
    top_nodes = top_nodes[:50]  # overall cap

    sks_dict = {
        str(cid): sorted(list(nodes))[:50]
        for cid, nodes in result.sks_clusters.items()
    }

    return {
        "type": "cycle",
        "cycle": state.cycle_count,
        "top_nodes": top_nodes,
        "sks_clusters": sks_dict,
        "metrics": {
            "n_sks": result.n_sks,
            "mean_pe": round(result.mean_prediction_error, 4),
            "n_spikes": result.n_spikes,
            "cycle_time_ms": round(result.cycle_time_ms, 2),
            "fps": round(state.fps, 1),
            "dcam_episodes": state.dcam_episodes,
        },
    }


async def _broadcast(data: dict) -> None:
    """Send JSON to all connected WebSocket clients."""
    dead: list[WebSocket] = []
    for ws in _ws_clients:
        try:
            await ws.send_json(data)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _ws_clients.remove(ws)


def _run_one_cycle() -> dict:
    """Execute one perception cycle and return JSON payload."""
    pipeline, dcam = _ensure_pipeline()
    image = _generate_stimulus()
    result = pipeline.perception_cycle(image)

    # Store in DCAM
    active_nodes = {}
    for cid, nodes in result.sks_clusters.items():
        for nid in nodes:
            active_nodes[nid] = (0.0, 1.0)
    context = torch.randn(dcam.config.hac_dim)
    context = context / (context.norm() + 1e-8)
    dcam.store_episode(active_nodes, context, importance=0.5)
    _state.dcam_episodes = dcam._cycle_count

    _state.cycle_count += 1
    payload = _cycle_to_json(result, _state)
    _state.last_result = payload
    return payload


async def _pipeline_loop() -> None:
    """Continuous pipeline execution loop."""
    loop = asyncio.get_event_loop()
    fps_window: list[float] = []
    while _state.running:
        t0 = time.perf_counter()
        payload = await loop.run_in_executor(None, _run_one_cycle)
        elapsed = time.perf_counter() - t0

        fps_window.append(elapsed)
        if len(fps_window) > 20:
            fps_window.pop(0)
        _state.fps = len(fps_window) / sum(fps_window) if fps_window else 0.0
        payload["metrics"]["fps"] = round(_state.fps, 1)

        await _broadcast(payload)
        await asyncio.sleep(0)  # yield to event loop


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="SNKS Dashboard")

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index():
    """Serve the dashboard."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "SNKS Dashboard — no index.html yet"}


@app.get("/status")
async def status():
    """Return current server state."""
    return {
        "running": _state.running,
        "cycle_count": _state.cycle_count,
        "fps": round(_state.fps, 1),
        "dcam_episodes": _state.dcam_episodes,
        "last_result": _state.last_result,
    }


@app.post("/start")
async def start():
    """Start continuous pipeline execution."""
    global _loop_task
    if _state.running:
        return {"status": "already_running"}
    _state.running = True
    _loop_task = asyncio.create_task(_pipeline_loop())
    return {"status": "started"}


@app.post("/pause")
async def pause():
    """Pause pipeline execution."""
    global _loop_task
    _state.running = False
    if _loop_task is not None:
        _loop_task.cancel()
        try:
            await _loop_task
        except asyncio.CancelledError:
            pass
        _loop_task = None
    return {"status": "paused"}


@app.post("/step")
async def step():
    """Run one perception cycle."""
    loop = asyncio.get_event_loop()
    payload = await loop.run_in_executor(None, _run_one_cycle)
    await _broadcast(payload)
    return payload


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await websocket.accept()
    _ws_clients.append(websocket)
    try:
        while True:
            # Keep connection alive; ignore incoming messages
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in _ws_clients:
            _ws_clients.remove(websocket)


# ---------------------------------------------------------------------------
# Reset (for testing)
# ---------------------------------------------------------------------------

def set_config(config: PipelineConfig) -> None:
    """Override pipeline config — used by tests."""
    global _config_override
    _config_override = config


def reset_state() -> None:
    """Reset all server state — used by tests."""
    global _pipeline, _dcam, _loop_task, _config_override
    _state.running = False
    _state.cycle_count = 0
    _state.fps = 0.0
    _state.last_result = {}
    _state.dcam_episodes = 0
    _pipeline = None
    _dcam = None
    _config_override = None
    _ws_clients.clear()
    if _loop_task is not None:
        _loop_task.cancel()
        _loop_task = None
