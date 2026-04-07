"""FastAPI server for Crafter Survival Demo.

Usage:
    cd /home/yorick/agi
    .venv/bin/python -m demos.crafter_demo.server

Opens http://localhost:8421
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from demos.crafter_demo.engine import DemoEngine

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
CKPT_DIR = Path(__file__).parent.parent / "checkpoints" / "exp128"

engine: DemoEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    ckpt = CKPT_DIR if CKPT_DIR.exists() else None
    engine = DemoEngine(ckpt_path=ckpt)
    engine.start()
    logger.info("DemoEngine started (has_model=%s)", engine.has_model)
    yield
    engine.stop()
    logger.info("DemoEngine stopped")


app = FastAPI(title="Crafter Survival Demo", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/status")
async def api_status():
    if engine is None:
        return JSONResponse({"error": "not started"}, status_code=503)
    return {
        "mode": engine.mode,
        "state": engine.state,
        "episode": engine.episode_count,
        "step": engine.step_count,
        "has_model": engine.has_model,
        "target_fps": engine.target_fps,
    }


@app.get("/api/goals")
async def api_goals():
    if engine is None:
        return JSONResponse({"error": "not started"}, status_code=503)
    with engine.model_lock:
        if engine.chain_gen:
            goals = engine.chain_gen.available_goals()
        else:
            goals = []
    return {"goals": goals}


@app.post("/api/train")
async def api_train(epochs: int = 150):
    if engine is None:
        return JSONResponse({"error": "not started"}, status_code=503)
    engine.start_training(epochs=epochs)
    return JSONResponse({"status": "started"}, status_code=202)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    if engine is None:
        await ws.close(code=1011, reason="engine not started")
        return

    async def send_loop():
        """Send snapshots at target FPS."""
        last_step = -1
        while True:
            with engine.snapshot_lock:
                snap = engine.snapshot
            # Only send if new data
            if snap.step != last_step or engine.state != "paused":
                try:
                    await ws.send_json(snap.to_dict())
                except Exception:
                    break
                last_step = snap.step

            # Send training progress if active
            with engine.train_lock:
                tp = engine.train_progress
                if tp.phase and not tp.done:
                    try:
                        await ws.send_json(tp.to_dict())
                    except Exception:
                        break
                elif tp.done and tp.phase:
                    try:
                        await ws.send_json(tp.to_dict())
                    except Exception:
                        break
                    with engine.train_lock:
                        engine.train_progress.phase = ""  # clear after sending

            await asyncio.sleep(1.0 / max(1, engine.target_fps))

    async def recv_loop():
        """Receive commands from client."""
        while True:
            try:
                data = await ws.receive_text()
                cmd = json.loads(data)
                if cmd.get("cmd") == "train":
                    engine.start_training(epochs=cmd.get("epochs", 150))
                else:
                    engine.send_cmd(cmd)
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.warning("WS recv error: %s", e)
                break

    send_task = asyncio.create_task(send_loop())
    recv_task = asyncio.create_task(recv_loop())

    try:
        done, pending = await asyncio.wait(
            [send_task, recv_task], return_when=asyncio.FIRST_COMPLETED
        )
    finally:
        send_task.cancel()
        recv_task.cancel()


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(
        "demos.crafter_demo.server:app",
        host="0.0.0.0",
        port=8421,
        reload=False,
    )
