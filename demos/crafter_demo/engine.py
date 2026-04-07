"""DemoEngine — shared state for Crafter Survival Demo.

Manages env, model, ConceptStore, and provides GameSnapshot for the UI.
Thread-safe: snapshot_lock for frame data, model_lock for encoder/detector swap.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from snks.agent.concept_store import ConceptStore, CausalLink
from snks.agent.chain_generator import ChainGenerator
from snks.agent.crafter_pixel_env import (
    CrafterPixelEnv,
    ACTION_NAMES,
    SEMANTIC_NAMES,
)
from snks.agent.crafter_spatial_map import CrafterSpatialMap
from snks.agent.crafter_textbook import CrafterTextbook
from snks.agent.reactive_check import ReactiveCheck
from snks.encoder.cnn_encoder import CNNEncoder
from snks.encoder.near_detector import NearDetector

logger = logging.getLogger(__name__)

# Survival keys live inside Crafter's inventory dict
SURVIVAL_KEYS = {"health", "food", "drink", "energy"}

# Minimap color palette (semantic ID → RGB)
MINIMAP_COLORS: dict[int, tuple[int, int, int]] = {
    0: (40, 40, 40),       # unknown
    1: (52, 152, 219),     # water — blue
    2: (46, 204, 113),     # grass — green
    3: (149, 165, 166),    # stone — gray
    4: (189, 195, 199),    # path — light gray
    5: (241, 196, 15),     # sand — yellow
    6: (39, 174, 96),      # tree — dark green
    7: (231, 76, 60),      # lava — red
    8: (44, 62, 80),       # coal — dark
    9: (160, 106, 58),     # iron — brown
    10: (155, 89, 182),    # diamond — purple
    11: (211, 84, 0),      # table — orange
    12: (192, 57, 43),     # furnace — dark red
    13: (255, 255, 255),   # player — white
    14: (241, 196, 15),    # cow — yellow
    15: (192, 57, 43),     # zombie — red
    16: (236, 240, 241),   # skeleton — pale
}


@dataclass
class GameSnapshot:
    """Immutable snapshot of game state for WS streaming."""

    frame_b64: str = ""
    step: int = 0
    episode: int = 0
    mode: str = "survival"
    state: str = "paused"
    agent_action: str = "noop"
    agent_near: str = "empty"
    agent_reason: str = "idle"
    plan_step: int = 0
    plan_total: int = 0
    survival: dict[str, int] = field(default_factory=dict)
    inventory: dict[str, int] = field(default_factory=dict)
    plan: list[dict[str, str]] = field(default_factory=list)
    reactive: dict[str, Any] | None = None
    confidence: dict[str, float] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    minimap_b64: str = ""
    log_lines: list[str] = field(default_factory=list)
    drives: dict[str, float] = field(default_factory=dict)
    perception_sim: float = 0.0
    grounding_events: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "type": "frame",
            "frame": self.frame_b64,
            "step": self.step,
            "episode": self.episode,
            "mode": self.mode,
            "state": self.state,
            "agent": {
                "action": self.agent_action,
                "near": self.agent_near,
                "reason": self.agent_reason,
                "plan_step": self.plan_step,
                "plan_total": self.plan_total,
            },
            "survival": self.survival,
            "inventory": self.inventory,
            "plan": self.plan,
            "reactive": self.reactive,
            "confidence": self.confidence,
            "metrics": self.metrics,
            "minimap": self.minimap_b64,
            "log": self.log_lines,
            "drives": self.drives,
            "perception_sim": self.perception_sim,
            "grounding_events": self.grounding_events,
        }


@dataclass
class TrainProgress:
    """Training progress for WS streaming."""

    phase: str = ""
    epoch: int = 0
    total_epochs: int = 0
    loss: float = 0.0
    done: bool = False
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "type": "train_progress",
            "phase": self.phase,
            "epoch": self.epoch,
            "total_epochs": self.total_epochs,
            "loss": self.loss,
            "done": self.done,
            "error": self.error,
        }


@dataclass
class EpisodeMetrics:
    """Per-episode metrics tracker."""

    episode_lengths: list[int] = field(default_factory=list)
    resources_collected: list[int] = field(default_factory=list)
    zombie_encounters: list[int] = field(default_factory=list)
    zombie_survived: list[int] = field(default_factory=list)

    # Current episode accumulators
    cur_resources: int = 0
    cur_encounters: int = 0
    cur_survived: int = 0

    # Lifetime item stats (persist across episodes)
    collected: dict[str, int] = field(default_factory=dict)  # gathered items
    crafted: dict[str, int] = field(default_factory=dict)    # crafted/placed items

    def record_collected(self, item: str) -> None:
        self.collected[item] = self.collected.get(item, 0) + 1
        self.cur_resources += 1

    def record_crafted(self, item: str) -> None:
        self.crafted[item] = self.crafted.get(item, 0) + 1

    def finish_episode(self, length: int) -> None:
        self.episode_lengths.append(length)
        self.resources_collected.append(self.cur_resources)
        self.zombie_encounters.append(self.cur_encounters)
        self.zombie_survived.append(self.cur_survived)
        self.cur_resources = 0
        self.cur_encounters = 0
        self.cur_survived = 0

    def to_dict(self) -> dict:
        return {
            "episode_length": self.episode_lengths[-1] if self.episode_lengths else 0,
            "resources_collected": self.cur_resources,
            "zombie_encounters": self.cur_encounters,
            "zombie_survived": self.cur_survived,
            "collected": dict(self.collected),
            "crafted": dict(self.crafted),
            "history": {
                "lengths": self.episode_lengths[-20:],
                "resources": self.resources_collected[-20:],
                "encounters": self.zombie_encounters[-20:],
                "survived": self.zombie_survived[-20:],
            },
        }


def pixels_to_b64png(pixels: np.ndarray) -> str:
    """Convert (3, 64, 64) float32 [0,1] to base64 PNG string."""
    from PIL import Image

    if pixels.shape[0] == 3:
        img_array = (pixels.transpose(1, 2, 0) * 255).astype(np.uint8)
    else:
        img_array = (pixels * 255).astype(np.uint8)
    img = Image.fromarray(img_array)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def minimap_to_b64png(semantic: np.ndarray, player_pos: np.ndarray, radius: int = 4) -> str:
    """Crop 9x9 from semantic map around player, render to colored PNG."""
    from PIL import Image

    px, py = int(player_pos[0]), int(player_pos[1])
    h, w = semantic.shape

    img = np.zeros((radius * 2 + 1, radius * 2 + 1, 3), dtype=np.uint8)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            sx, sy = px + dx, py + dy
            if 0 <= sx < h and 0 <= sy < w:
                sid = int(semantic[sx, sy])
            else:
                sid = 0
            color = MINIMAP_COLORS.get(sid, (40, 40, 40))
            img[dy + radius, dx + radius] = color

    # Mark player center
    img[radius, radius] = (255, 255, 255)

    pil_img = Image.fromarray(img)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def flatten_confidence(store: ConceptStore) -> dict[str, float]:
    """Flatten ConceptStore causal link confidences to {key: float}."""
    result = {}
    for cid, concept in store.concepts.items():
        for link in concept.causal_links:
            key = f"{cid}_{link.action}_{link.result}"
            result[key] = round(link.confidence, 3)
    return result


class DemoEngine:
    """Central engine — shared between env thread, train thread, and FastAPI."""

    def __init__(self, ckpt_path: Path | None = None) -> None:
        # Models (protected by model_lock)
        self.encoder: CNNEncoder | None = None
        self.detector: NearDetector | None = None
        self.store: ConceptStore = ConceptStore()
        self.chain_gen: ChainGenerator | None = None
        self.reactive: ReactiveCheck | None = None
        self.spatial_map: CrafterSpatialMap = CrafterSpatialMap()
        self.model_lock = threading.Lock()

        # Env
        self.env: CrafterPixelEnv | None = None
        self.last_pixels: np.ndarray | None = None
        self.last_info: dict = {}

        # State
        self.mode: str = "survival"  # survival | interactive
        self.state: str = "paused"   # playing | paused | stepping
        self.target_fps: int = 10
        self.step_count: int = 0
        self.episode_count: int = 0
        self.has_model: bool = False

        # Snapshot (protected by snapshot_lock)
        self.snapshot = GameSnapshot()
        self.snapshot_lock = threading.Lock()

        # Metrics
        self.metrics = EpisodeMetrics()

        # Event log
        self.event_log: deque[str] = deque(maxlen=200)
        self._new_log_lines: list[str] = []

        # Command queue for env thread
        self.cmd_queue: queue.Queue[dict] = queue.Queue()
        self._pending_goal: str = ""

        # Training progress
        self.train_progress = TrainProgress()
        self.train_lock = threading.Lock()

        # Threads
        self._env_thread: threading.Thread | None = None
        self._train_thread: threading.Thread | None = None
        self._running = False

        # Load checkpoint if available
        if ckpt_path:
            self._load_checkpoint(ckpt_path)
        else:
            self._load_textbook_only()

    def _load_checkpoint(self, base: Path) -> None:
        """Load checkpoint with fallback chain."""
        candidates = [
            base / "final",
            base / "phase3",
            base / "phase1",
        ]

        loaded = False
        for d in candidates:
            if (d / "encoder.pt").exists():
                logger.info("Loading checkpoint from %s", d)
                self._load_from_dir(d)
                loaded = True
                break

        if not loaded:
            # Try legacy stage66.pt
            legacy = base.parent / "stage66.pt"
            if legacy.exists():
                logger.info("Loading legacy checkpoint %s", legacy)
                self._load_legacy(legacy)
                loaded = True

        if not loaded:
            logger.warning("No checkpoint found, starting without model")
            self._load_textbook_only()

    def _load_from_dir(self, d: Path) -> None:
        """Load encoder + detector + ConceptStore from checkpoint dir."""
        encoder = CNNEncoder()
        encoder.load_state_dict(torch.load(d / "encoder.pt", weights_only=True))
        encoder.eval()

        detector = None
        det_path = d / "detector.pt"
        if det_path.exists():
            det_data = torch.load(det_path, weights_only=True)
            det_encoder = CNNEncoder()
            det_encoder.load_state_dict(det_data["encoder"])
            det_encoder.eval()
            detector = NearDetector(det_encoder)
        else:
            detector = NearDetector(encoder)

        store = ConceptStore()
        store_path = d / "concept_store"
        if store_path.exists():
            store.load(str(store_path))
        else:
            self._init_textbook(store)

        with self.model_lock:
            self.encoder = encoder
            self.detector = detector
            self.store = store
            self.chain_gen = ChainGenerator(store, use_semantic_nav=False)
            self.reactive = ReactiveCheck(store)
            self.has_model = True

    def _load_legacy(self, path: Path) -> None:
        """Load legacy stage66.pt checkpoint."""
        data = torch.load(path, weights_only=True)
        encoder = CNNEncoder()
        if "encoder" in data:
            encoder.load_state_dict(data["encoder"])
        else:
            encoder.load_state_dict(data)
        encoder.eval()
        detector = NearDetector(encoder)

        store = ConceptStore()
        self._init_textbook(store)

        with self.model_lock:
            self.encoder = encoder
            self.detector = detector
            self.store = store
            self.chain_gen = ChainGenerator(store, use_semantic_nav=False)
            self.reactive = ReactiveCheck(store)
            self.has_model = True

    def _load_textbook_only(self) -> None:
        """Initialize ConceptStore from textbook without any model."""
        store = ConceptStore()
        self._init_textbook(store)
        with self.model_lock:
            self.store = store
            self.chain_gen = ChainGenerator(store, use_semantic_nav=False)
            self.reactive = ReactiveCheck(store)
            self.has_model = False

    @staticmethod
    def _init_textbook(store: ConceptStore) -> None:
        tb_path = Path(__file__).parent.parent.parent / "configs" / "crafter_textbook.yaml"
        if tb_path.exists():
            tb = CrafterTextbook(str(tb_path))
            tb.load_into(store)

    def reset_env(self) -> None:
        """Reset or create env."""
        self.env = CrafterPixelEnv()
        pixels, info = self.env.reset()
        self.last_pixels = pixels
        self.last_info = info
        self.step_count = 0
        self.episode_count += 1
        self.spatial_map.reset()

    def log_event(self, msg: str) -> None:
        """Add event to log."""
        entry = f"[{self.step_count}] {msg}"
        self.event_log.append(entry)
        self._new_log_lines.append(entry)

    def drain_new_logs(self) -> list[str]:
        """Return and clear new log lines since last drain."""
        lines = self._new_log_lines[-5:]  # last 5 new
        self._new_log_lines = []
        return lines

    def build_snapshot(
        self,
        agent_action: str = "noop",
        agent_near: str = "empty",
        agent_reason: str = "idle",
        plan_data: list[dict] | None = None,
        plan_step: int = 0,
        plan_total: int = 0,
        reactive_data: dict | None = None,
        drives: dict[str, float] | None = None,
        perception_sim: float = 0.0,
        grounding_events: list[str] | None = None,
    ) -> GameSnapshot:
        """Build GameSnapshot from current state."""
        inv = dict(self.last_info.get("inventory", {}))
        survival = {k: inv.pop(k, 9) for k in list(SURVIVAL_KEYS)}
        items = {k: v for k, v in inv.items() if v > 0}

        frame_b64 = ""
        if self.last_pixels is not None:
            frame_b64 = pixels_to_b64png(self.last_pixels)

        minimap_b64 = ""
        semantic = self.last_info.get("semantic")
        player_pos = self.last_info.get("player_pos")
        if semantic is not None and player_pos is not None:
            minimap_b64 = minimap_to_b64png(semantic, player_pos)

        with self.model_lock:
            confidence = flatten_confidence(self.store)

        return GameSnapshot(
            frame_b64=frame_b64,
            step=self.step_count,
            episode=self.episode_count,
            mode=self.mode,
            state=self.state,
            agent_action=agent_action,
            agent_near=agent_near,
            agent_reason=agent_reason,
            plan_step=plan_step,
            plan_total=plan_total,
            survival=survival,
            inventory=items,
            plan=plan_data or [],
            reactive=reactive_data,
            confidence=confidence,
            metrics=self.metrics.to_dict(),
            minimap_b64=minimap_b64,
            log_lines=self.drain_new_logs(),
            drives=drives or {},
            perception_sim=perception_sim,
            grounding_events=grounding_events or [],
        )

    def send_cmd(self, cmd: dict) -> None:
        """Send command to env thread."""
        self.cmd_queue.put(cmd)

    def start(self) -> None:
        """Start env thread."""
        if self._running:
            return
        self._running = True
        self.reset_env()

        from demos.crafter_demo.agent_loop import env_thread_loop
        self._env_thread = threading.Thread(
            target=env_thread_loop, args=(self,), daemon=True
        )
        self._env_thread.start()

    def stop(self) -> None:
        """Stop env thread."""
        self._running = False
        if self._env_thread:
            self._env_thread.join(timeout=5)

    def start_training(self, epochs: int = 150) -> None:
        """Start training in background thread."""
        if self._train_thread and self._train_thread.is_alive():
            logger.warning("Training already running")
            return

        self._train_thread = threading.Thread(
            target=self._train_worker, args=(epochs,), daemon=True
        )
        self._train_thread.start()

    def _train_worker(self, epochs: int) -> None:
        """Training worker — runs in TrainThread."""
        try:
            with self.train_lock:
                self.train_progress = TrainProgress(phase="collecting", total_epochs=epochs)

            from demos.crafter_demo.train import run_training
            encoder, detector, store = run_training(
                epochs=epochs,
                existing_store=self.store,
                progress_cb=self._train_progress_cb,
            )

            with self.model_lock:
                self.encoder = encoder
                self.detector = detector
                self.store = store
                self.chain_gen = ChainGenerator(store, use_semantic_nav=False)
                self.reactive = ReactiveCheck(store)
                self.has_model = True

            with self.train_lock:
                self.train_progress.done = True
                self.train_progress.phase = "done"

            self.log_event("Training complete — model hot-swapped")

        except Exception as e:
            logger.exception("Training failed")
            with self.train_lock:
                self.train_progress.error = str(e)
                self.train_progress.done = True

    def _train_progress_cb(self, phase: str, epoch: int, total: int, loss: float) -> None:
        with self.train_lock:
            self.train_progress.phase = phase
            self.train_progress.epoch = epoch
            self.train_progress.total_epochs = total
            self.train_progress.loss = loss
