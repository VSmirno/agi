#!/usr/bin/env /home/yorick/agi/.venv/bin/python
"""Stage 66 Interactive Demo — Pixel Crafter + Prototype Memory World Model.

Usage:
    # Quick train from scratch (20 trajs × 100 steps, 30 epochs), then demo:
    .venv/bin/python demos/stage66_demo.py --mode scratch

    # Load saved checkpoint, skip training:
    .venv/bin/python demos/stage66_demo.py --mode checkpoint --ckpt demos/checkpoints/stage66.pt

Run from project root:
    cd /home/yorick/agi
"""

from __future__ import annotations

import sys
import os
import argparse
import time
import math
import random
from typing import Optional

# Must be before any snks imports
sys.path.insert(0, "/home/yorick/agi/src")

import numpy as np
import torch
import torch.nn.functional as F

# ── Imports ──────────────────────────────────────────────────────────────────

try:
    import pygame
    import pygame.font
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False
    print("ERROR: pygame not installed. Run: pip install pygame")
    sys.exit(1)

try:
    import crafter
    HAS_CRAFTER = True
except ImportError:
    HAS_CRAFTER = False
    print("ERROR: crafter not installed. Run: pip install crafter")
    sys.exit(1)

from snks.encoder.cnn_encoder import CNNEncoder
from snks.encoder.predictive_trainer import JEPAPredictor, PredictiveTrainer
from snks.agent.prototype_memory import PrototypeMemory
from snks.agent.cls_world_model import CLSWorldModel
from snks.agent.crafter_pixel_env import (
    CrafterPixelEnv, ACTION_TO_IDX, ACTION_NAMES, SEMANTIC_NAMES, INVENTORY_ITEMS,
)
from snks.agent.crafter_trainer import (
    CRAFTER_RULES, generate_taught_transitions,
)
from snks.agent.decode_head import NEAR_CLASSES

# ── Constants ─────────────────────────────────────────────────────────────────

WINDOW_W, WINDOW_H = 1200, 700
FRAME_SIZE = 512  # left panel: pixel frame scaled to this

# Colors
BG       = (26, 26, 46)
PANEL_BG = (20, 20, 38)
TEXT_CLR = (234, 234, 234)
DIM_CLR  = (140, 140, 170)
GREEN    = (46, 204, 113)
YELLOW   = (243, 156, 18)
RED      = (231, 76, 60)
BLUE     = (52, 152, 219)
BORDER   = (60, 60, 90)

# Semantic map object IDs
OBJ_ID = {
    "water": 1, "grass": 2, "stone": 3, "path": 4, "sand": 5,
    "tree": 6, "lava": 7, "coal": 8, "iron": 9, "diamond": 10,
    "table": 11, "furnace": 12, "player": 13, "cow": 14,
    "zombie": 15, "skeleton": 16,
}

# Action index shortcuts
ACT = ACTION_TO_IDX  # e.g. ACT["do"] = 5

# Movement actions
MOVE_ACTIONS = {
    "left":  ACT["move_left"],
    "right": ACT["move_right"],
    "up":    ACT["move_up"],
    "down":  ACT["move_down"],
}

# ── Goals ─────────────────────────────────────────────────────────────────────

# Each goal: (display_name, target_object_name, final_action_str, requires_items)
GOALS = [
    ("collect_wood",        "tree",    "do",                  {}),
    ("collect_stone",       "stone",   "do",                  {"wood_pickaxe": 1}),
    ("collect_coal",        "coal",    "do",                  {"wood_pickaxe": 1}),
    ("collect_iron",        "iron",    "do",                  {"stone_pickaxe": 1}),
    ("make_wood_pickaxe",   "table",   "make_wood_pickaxe",   {"wood": 1}),
    ("make_stone_pickaxe",  "table",   "make_stone_pickaxe",  {"wood": 1, "stone": 1}),
    ("make_wood_sword",     "table",   "make_wood_sword",     {"wood": 1}),
    ("place_table",         None,      "place_table",         {"wood": 2}),
    ("free_explore",        None,      None,                  {}),
]

# ── Planner ───────────────────────────────────────────────────────────────────

def compute_plan(goal_idx: int, inventory: dict[str, int]) -> list[dict]:
    """Compute dependency chain for a goal given current inventory.

    Returns list of step dicts:
        {"label": str, "target": str|None, "action": str|None, "done": bool}
    """
    goal_name, target_obj, final_action, requires = GOALS[goal_idx]

    if goal_name == "free_explore":
        return [{"label": "random walk", "target": None, "action": None, "done": False}]

    steps = []

    # Check dependencies
    if goal_name == "collect_stone" or goal_name == "collect_coal":
        # Need wood_pickaxe → need wood + table
        if not inventory.get("wood_pickaxe", 0):
            if not inventory.get("wood", 0):
                steps.append({
                    "label": "collect_wood (for pickaxe)",
                    "target": "tree", "action": "do", "done": False,
                })
            if not _has_nearby_table_in_plan():
                steps.append({
                    "label": "place_table",
                    "target": None, "action": "place_table", "done": False,
                })
            steps.append({
                "label": "make_wood_pickaxe",
                "target": "table", "action": "make_wood_pickaxe", "done": False,
            })

    elif goal_name == "collect_iron":
        # Need stone_pickaxe → need wood + stone + table
        if not inventory.get("stone_pickaxe", 0):
            if not inventory.get("wood_pickaxe", 0):
                if not inventory.get("wood", 0):
                    steps.append({
                        "label": "collect_wood",
                        "target": "tree", "action": "do", "done": False,
                    })
                steps.append({
                    "label": "place_table",
                    "target": None, "action": "place_table", "done": False,
                })
                steps.append({
                    "label": "make_wood_pickaxe",
                    "target": "table", "action": "make_wood_pickaxe", "done": False,
                })
            if not inventory.get("stone", 0):
                steps.append({
                    "label": "collect_stone",
                    "target": "stone", "action": "do", "done": False,
                })
            steps.append({
                "label": "make_stone_pickaxe",
                "target": "table", "action": "make_stone_pickaxe", "done": False,
            })

    elif goal_name in ("make_wood_pickaxe", "make_stone_pickaxe",
                       "make_wood_sword"):
        # Need wood + possibly stone
        if not inventory.get("wood", 0):
            steps.append({
                "label": "collect_wood",
                "target": "tree", "action": "do", "done": False,
            })
        if "stone" in requires and not inventory.get("stone", 0):
            steps.append({
                "label": "collect_stone",
                "target": "stone", "action": "do", "done": False,
            })
        # Need a table
        steps.append({
            "label": f"navigate to table for {goal_name}",
            "target": "table", "action": None, "done": False,
        })

    elif goal_name == "place_table":
        if inventory.get("wood", 0) < 2:
            steps.append({
                "label": "collect_wood (need 2)",
                "target": "tree", "action": "do", "done": False,
            })

    # Final step
    if target_obj is not None:
        steps.append({
            "label": f"navigate to {target_obj}",
            "target": target_obj, "action": None, "done": False,
        })
    steps.append({
        "label": goal_name,
        "target": target_obj,
        "action": final_action,
        "done": False,
    })

    return steps


def _has_nearby_table_in_plan() -> bool:
    """Placeholder — always assume we need a table unless inventory says we have one."""
    return False


# ── Navigator ─────────────────────────────────────────────────────────────────

def find_nearest_object(semantic: np.ndarray, player_pos: np.ndarray,
                        obj_name: str) -> Optional[tuple[int, int]]:
    """Find nearest cell with given object type in semantic map.

    Returns (row, col) or None.
    """
    obj_id = OBJ_ID.get(obj_name)
    if obj_id is None:
        return None

    py, px = int(player_pos[0]), int(player_pos[1])
    h, w = semantic.shape

    best_pos = None
    best_dist = float("inf")

    for r in range(h):
        for c in range(w):
            if int(semantic[r, c]) == obj_id:
                dist = abs(r - py) + abs(c - px)
                if dist < best_dist:
                    best_dist = dist
                    best_pos = (r, c)

    return best_pos


def navigate_toward(player_pos: np.ndarray,
                    target_pos: tuple[int, int]) -> int:
    """Return a move action to get closer to target.

    Returns action index.
    """
    py, px = int(player_pos[0]), int(player_pos[1])
    ty, tx = target_pos

    dy = ty - py
    dx = tx - px

    if abs(dy) >= abs(dx):
        return ACT["move_down"] if dy > 0 else ACT["move_up"]
    else:
        return ACT["move_right"] if dx > 0 else ACT["move_left"]


def is_adjacent(player_pos: np.ndarray, target_pos: tuple[int, int],
                threshold: int = 2) -> bool:
    """True if player is within threshold cells of target."""
    py, px = int(player_pos[0]), int(player_pos[1])
    ty, tx = target_pos
    return abs(py - ty) <= threshold and abs(px - tx) <= threshold


# ── Training ──────────────────────────────────────────────────────────────────

def make_situation_label(sym_obs: dict) -> str:
    near = sym_obs.get("near", "empty")
    inv_parts = sorted(k for k in sym_obs if k.startswith("has_"))
    inv_key = "_".join(inv_parts) if inv_parts else "noinv"
    return f"{near}_{inv_key}"


def collect_trajectories(n_trajectories: int = 20, steps_per_traj: int = 100,
                         seed: int = 42,
                         screen=None, font=None,
                         progress_callback=None) -> dict:
    """Collect pixel transitions for encoder training."""
    all_pt, all_pt1, all_actions, all_sit_labels = [], [], [], []
    label_to_idx: dict[str, int] = {}

    for traj in range(n_trajectories):
        env = CrafterPixelEnv(seed=seed + traj * 7)
        pixels, sym = env.reset()

        rng = np.random.RandomState(seed + traj * 1000)
        for step in range(steps_per_traj):
            action_idx = rng.randint(0, 17)
            next_pixels, next_sym, _r, done = env.step(action_idx)

            all_pt.append(torch.from_numpy(pixels))
            all_pt1.append(torch.from_numpy(next_pixels))
            all_actions.append(action_idx)

            sit_label = make_situation_label(sym)
            if sit_label not in label_to_idx:
                label_to_idx[sit_label] = len(label_to_idx)
            all_sit_labels.append(label_to_idx[sit_label])

            pixels = next_pixels
            sym = next_sym
            if done:
                pixels, sym = env.reset()

        if progress_callback:
            progress_callback(f"Collecting trajectories... {traj+1}/{n_trajectories}")
        _pump_events()

    return {
        "pixels_t":          torch.stack(all_pt),
        "pixels_t1":         torch.stack(all_pt1),
        "actions":           torch.tensor(all_actions),
        "situation_labels":  torch.tensor(all_sit_labels),
        "label_to_idx":      label_to_idx,
    }


def train_encoder(dataset: dict, epochs: int = 30, batch_size: int = 128,
                  progress_callback=None) -> tuple[CNNEncoder, JEPAPredictor]:
    """Train CNN encoder (JEPA + SupCon)."""
    encoder = CNNEncoder(n_near_classes=len(NEAR_CLASSES))
    predictor = JEPAPredictor()
    trainer = PredictiveTrainer(encoder, predictor,
                                contrastive_weight=0.5, device="cpu")

    # Minimal train_full replacement that reports per-epoch to callback
    from torch.utils.data import DataLoader, TensorDataset
    from snks.encoder.predictive_trainer import supcon_loss

    pixels_t  = dataset["pixels_t"]
    pixels_t1 = dataset["pixels_t1"]
    actions   = dataset["actions"]
    sit_labels = dataset["situation_labels"]

    ds = TensorDataset(pixels_t, pixels_t1, actions, sit_labels)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    opt = torch.optim.Adam(
        list(encoder.parameters()) + list(predictor.parameters()), lr=1e-3
    )

    encoder.train()
    predictor.train()

    for epoch in range(epochs):
        total_loss = 0.0
        for pt, pt1, acts, labels in loader:
            out_t  = encoder(pt)
            out_t1 = encoder(pt1)
            z_t  = out_t.z_real
            z_t1 = out_t1.z_real

            # JEPA prediction loss
            z_pred = predictor(z_t, acts)
            pred_loss = F.mse_loss(z_pred, z_t1.detach())

            # SupCon
            con_loss = supcon_loss(z_t, labels)

            loss = pred_loss + 0.5 * con_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

        if progress_callback:
            progress_callback(
                f"Training encoder... epoch {epoch+1}/{epochs}  "
                f"loss={total_loss/max(len(loader),1):.4f}"
            )
        _pump_events()

    encoder.eval()
    return encoder, predictor


def collect_prototypes(encoder: CNNEncoder, cls: CLSWorldModel,
                       n_seeds: int = 20,
                       progress_callback=None) -> None:
    """Collect per-rule pixel prototypes."""
    encoder.eval()
    n_rules = len(CRAFTER_RULES)
    for ri, rule in enumerate(CRAFTER_RULES):
        rule_added = 0
        for seed_idx in range(n_seeds):
            seed = 2000 + seed_idx * 13 + ri * 100
            env = CrafterPixelEnv(seed=seed)
            pixels, sym = env.reset()

            rng = np.random.RandomState(seed)
            found = False
            for _ in range(200):
                act_name = rng.choice(
                    ["move_left", "move_right", "move_up", "move_down"]
                )
                pixels, sym, _, done = env.step(act_name)
                if sym.get("near") == rule["near"]:
                    found = True
                    break
                if done:
                    pixels, sym = env.reset()

            if not found:
                continue

            with torch.no_grad():
                out = encoder(torch.from_numpy(pixels))
            outcome = {"result": rule["result"], "gives": rule.get("gives", "")}
            cls.prototype_memory.add(out.z_real, rule["action"], outcome)
            rule_added += 1

        if progress_callback:
            progress_callback(
                f"Collecting prototypes... rule {ri+1}/{n_rules}: "
                f"{rule['action']} near {rule['near']} ({rule_added}/{n_seeds})"
            )
        _pump_events()

    # Load symbolic rules into neocortex
    cls.train(generate_taught_transitions())


def save_checkpoint(path: str, encoder: CNNEncoder, cls: CLSWorldModel) -> None:
    state = {
        "encoder_state":       encoder.state_dict(),
        "prototype_z":         [z.cpu() for z in cls.prototype_memory.z_store],
        "prototype_actions":   cls.prototype_memory.actions,
        "prototype_outcomes":  cls.prototype_memory.outcomes,
    }
    torch.save(state, path)
    print(f"Checkpoint saved → {path}")


def load_checkpoint(path: str) -> tuple[CNNEncoder, CLSWorldModel]:
    state = torch.load(path, map_location="cpu")
    encoder = CNNEncoder(n_near_classes=len(NEAR_CLASSES))
    encoder.load_state_dict(state["encoder_state"])
    encoder.eval()

    cls = CLSWorldModel(dim=2048, device="cpu")
    mem = cls.prototype_memory
    mem.z_store   = state["prototype_z"]
    mem.actions   = state["prototype_actions"]
    mem.outcomes  = state["prototype_outcomes"]

    # Re-load symbolic rules
    cls.train(generate_taught_transitions())

    print(f"Checkpoint loaded from {path}  ({len(mem)} prototypes)")
    return encoder, cls


# ── Pygame helpers ────────────────────────────────────────────────────────────

def _pump_events():
    """Drain pygame event queue (prevents OS 'not responding' during training)."""
    if HAS_PYGAME and pygame.get_init():
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)


def draw_training_screen(screen, font_lg, font_sm, message: str, sub: str = ""):
    screen.fill(BG)
    # Title
    surf = font_lg.render("Stage 66 — Training", True, TEXT_CLR)
    screen.blit(surf, (WINDOW_W // 2 - surf.get_width() // 2, 200))
    # Main message
    surf2 = font_sm.render(message, True, YELLOW)
    screen.blit(surf2, (WINDOW_W // 2 - surf2.get_width() // 2, 300))
    if sub:
        surf3 = font_sm.render(sub, True, DIM_CLR)
        screen.blit(surf3, (WINDOW_W // 2 - surf3.get_width() // 2, 340))
    pygame.display.flip()


# ── Main Demo ─────────────────────────────────────────────────────────────────

class DemoState:
    """All mutable state for one demo run."""

    def __init__(self):
        # Crafter env (raw, for info dict access)
        self.env = crafter.Env(seed=random.randint(0, 9999))
        self.obs = self.env.reset()          # (64,64,3) uint8
        self.obs, _r, _d, self.info = self.env.step(0)  # noop → get info

        # Agent state
        self.goal_idx: int = 0
        self.plan: list[dict] = []
        self.plan_step: int = 0
        self.status: str = "idle"            # idle / navigating / acting / done / failed
        self.thought: str = ""               # what the agent is thinking this step
        self.wm_pred: str = ""              # world model prediction text
        self.paused: bool = False
        self.step_count: int = 0
        self.last_step_time: float = 0.0

        # Navigation state
        self.nav_stuck_count: int = 0
        self.last_player_pos: Optional[np.ndarray] = None

        # World model display
        self.wm_action: str = ""
        self.wm_outcome: str = ""
        self.wm_conf: float = 0.0
        self.wm_source: str = ""

        self._rebuild_plan()

    def get_inventory(self) -> dict[str, int]:
        inv = self.info.get("inventory", {})
        return {k: int(v) for k, v in inv.items() if v > 0}

    def get_semantic(self) -> np.ndarray:
        return self.info.get("semantic", np.zeros((9, 9), dtype=int))

    def get_player_pos(self) -> np.ndarray:
        return self.info.get("player_pos", np.array([4, 4]))

    def get_pixels(self) -> np.ndarray:
        """(3,64,64) float32 [0,1]."""
        return self.obs.astype(np.float32).transpose(2, 0, 1) / 255.0

    def _rebuild_plan(self):
        inv = self.get_inventory()
        self.plan = compute_plan(self.goal_idx, inv)
        self.plan_step = 0
        self.status = "navigating" if self.plan else "idle"
        self.nav_stuck_count = 0

    def set_goal(self, goal_idx: int):
        self.goal_idx = goal_idx % len(GOALS)
        self._rebuild_plan()
        self.thought = f"New goal: {GOALS[self.goal_idx][0]}"

    def reset_env(self):
        self.env = crafter.Env(seed=random.randint(0, 9999))
        self.obs = self.env.reset()
        self.obs, _r, _d, self.info = self.env.step(0)
        self.step_count = 0
        self.nav_stuck_count = 0
        self.last_player_pos = None
        self._rebuild_plan()
        self.thought = "Environment reset"
        self.wm_pred = ""


def agent_step(state: DemoState, encoder: CNNEncoder, cls: CLSWorldModel):
    """Execute one agent step: navigate or act."""
    if state.paused or state.status in ("done", "failed", "idle"):
        return

    if not state.plan or state.plan_step >= len(state.plan):
        state.status = "done"
        state.thought = "Plan complete!"
        return

    step = state.plan[state.plan_step]
    goal_name, _, _, _ = GOALS[state.goal_idx]
    semantic = state.get_semantic()
    player_pos = state.get_player_pos()

    # Free explore: random walk
    if goal_name == "free_explore":
        action_idx = random.choice([ACT["move_left"], ACT["move_right"],
                                    ACT["move_up"], ACT["move_down"]])
        state.thought = "Free exploring (random walk)"
        _env_step(state, action_idx, encoder, cls, step)
        return

    target_obj = step.get("target")
    action_str = step.get("action")

    # If no target and no action, skip step
    if target_obj is None and action_str is None:
        state.plan[state.plan_step]["done"] = True
        state.plan_step += 1
        return

    # Handle "place_table" — just do it in place (no specific target needed)
    if action_str == "place_table" and target_obj is None:
        state.status = "acting"
        state.thought = "Placing table..."
        _wm_query(state, encoder, cls, "place_table")
        action_idx = ACT["place_table"]
        obs, r, done, info = state.env.step(action_idx)
        state.obs = obs
        state.info = info
        state.step_count += 1
        if done:
            state.reset_env()
            return
        # Check if table is now in semantic map
        sem = state.get_semantic()
        has_table = (sem == OBJ_ID["table"]).any()
        if has_table or r > 0:
            state.plan[state.plan_step]["done"] = True
            state.plan_step += 1
            state.status = "navigating"
            # Rebuild plan with new inventory
            old_idx = state.goal_idx
            state.goal_idx = old_idx
            state._rebuild_plan()
        return

    # Navigate to target
    if target_obj is not None:
        target_pos = find_nearest_object(semantic, player_pos, target_obj)

        if target_pos is None:
            # Target not visible — random walk to explore
            state.status = "navigating"
            state.thought = f"Searching for {target_obj}..."
            action_idx = random.choice([ACT["move_left"], ACT["move_right"],
                                        ACT["move_up"], ACT["move_down"]])
            _env_step(state, action_idx, encoder, cls, step)
            return

        if not is_adjacent(player_pos, target_pos, threshold=2):
            # Move toward target
            state.status = "navigating"
            state.thought = f"Moving toward {target_obj}..."
            move_idx = navigate_toward(player_pos, target_pos)

            # Stuck detection
            pp = tuple(player_pos.tolist())
            if state.last_player_pos is not None:
                lp = tuple(state.last_player_pos.tolist())
                if pp == lp:
                    state.nav_stuck_count += 1
                else:
                    state.nav_stuck_count = 0
            state.last_player_pos = player_pos.copy()

            if state.nav_stuck_count > 8:
                # Unstick: random move
                move_idx = random.choice([ACT["move_left"], ACT["move_right"],
                                          ACT["move_up"], ACT["move_down"]])
                state.nav_stuck_count = 0

            _env_step(state, move_idx, encoder, cls, step)
            return

    # Adjacent to target (or no navigation needed) — execute action
    if action_str is not None:
        state.status = "acting"
        state.thought = f"Executing: {action_str}"
        _wm_query(state, encoder, cls, action_str)
        action_idx = ACT.get(action_str, ACT["do"])
        _env_step(state, action_idx, encoder, cls, step)
        # Mark step as done (optimistically — check next iteration)
        inv = state.get_inventory()
        # Simple check: did something change / reward signal?
        state.plan[state.plan_step]["done"] = True
        state.plan_step += 1
        state.status = "navigating"
        if state.plan_step >= len(state.plan):
            state.status = "done"
            state.thought = f"Goal '{goal_name}' complete!"


def _wm_query(state: DemoState, encoder: CNNEncoder,
              cls: CLSWorldModel, action_str: str):
    """Query world model from pixels, update state display fields."""
    try:
        with torch.no_grad():
            pixels = torch.from_numpy(state.get_pixels())
            outcome, conf, source = cls.query_from_pixels(pixels, action_str, encoder)
        result = outcome.get("result", "unknown")
        gives  = outcome.get("gives", "")
        state.wm_action  = action_str
        state.wm_outcome = f"{result}" + (f" → {gives}" if gives else "")
        state.wm_conf    = conf
        state.wm_source  = source
        state.wm_pred = (
            f'"{action_str}" → {result}'
            + (f" ({gives})" if gives else "")
            + f"  conf={conf:.2f}  src={source}"
        )
    except Exception as e:
        state.wm_pred = f"WM error: {e}"


def _env_step(state: DemoState, action_idx: int,
              encoder: CNNEncoder, cls: CLSWorldModel, step: dict):
    """Execute action in env, update state."""
    obs, r, done, info = state.env.step(action_idx)
    state.obs = obs
    state.info = info
    state.step_count += 1
    if done:
        state.thought = "Episode done — resetting..."
        state.reset_env()


# ── Rendering ──────────────────────────────────────────────────────────────────

def render(screen, state: DemoState, fonts: dict):
    screen.fill(BG)
    font_lg  = fonts["lg"]
    font_md  = fonts["md"]
    font_sm  = fonts["sm"]
    font_xs  = fonts["xs"]

    # ── Left Panel: Crafter Frame ──────────────────────────────────────────
    # Draw border
    pygame.draw.rect(screen, BORDER, (0, 0, FRAME_SIZE + 4, FRAME_SIZE + 4))

    # Scale pixel obs 64×64 → 512×512
    obs_hwc = state.obs  # (64,64,3) uint8
    surf_small = pygame.surfarray.make_surface(obs_hwc.transpose(1, 0, 2))
    surf_big   = pygame.transform.scale(surf_small, (FRAME_SIZE, FRAME_SIZE))
    screen.blit(surf_big, (2, 2))

    # Step counter overlay
    step_text = font_xs.render(f"step {state.step_count}", True, DIM_CLR)
    screen.blit(step_text, (8, FRAME_SIZE - 22))

    # ── Right Panel ────────────────────────────────────────────────────────
    rx = FRAME_SIZE + 20  # right panel x start
    ry = 12

    def rtext(txt, color=TEXT_CLR, font=font_sm, bold=False):
        nonlocal ry
        if bold:
            surf = font_md.render(txt, True, color)
        else:
            surf = font.render(txt, True, color)
        screen.blit(surf, (rx, ry))
        ry += surf.get_height() + 3

    def rsep(h=8):
        nonlocal ry
        ry += h

    # Title
    rtext("Stage 66 — Prototype Memory Demo", TEXT_CLR, font_lg)
    rsep()

    # Goal
    goal_name = GOALS[state.goal_idx][0]
    rtext(f"Goal:  {goal_name}", YELLOW, font_md)
    rsep(4)

    # Inventory
    rtext("Inventory:", DIM_CLR, font_sm)
    inv = state.get_inventory()
    display_items = [
        "wood", "stone", "coal", "iron", "diamond",
        "wood_pickaxe", "stone_pickaxe", "iron_pickaxe",
        "wood_sword", "stone_sword",
    ]
    shown = 0
    for item in display_items:
        cnt = inv.get(item, 0)
        if cnt > 0:
            rtext(f"  {item}: {cnt}", TEXT_CLR, font_sm)
            shown += 1
    if shown == 0:
        rtext("  (empty)", DIM_CLR, font_sm)
    rsep()

    # Plan
    rtext("Agent plan:", DIM_CLR, font_sm)
    for i, step in enumerate(state.plan):
        if step.get("done"):
            mark  = "[x]"
            color = GREEN
        elif i == state.plan_step:
            mark  = ">>>"
            color = YELLOW
        else:
            mark  = "[ ]"
            color = DIM_CLR
        label = step.get("label", "?")
        rtext(f"  {mark} {label}", color, font_sm)
    rsep()

    # World model reasoning
    rtext("World model reasoning:", DIM_CLR, font_sm)
    if state.wm_pred:
        # Wrap long text
        pred = state.wm_pred
        if len(pred) > 55:
            pred = pred[:55] + "…"
        rtext(f"  {pred}", BLUE, font_sm)
        rtext(f"  source: {state.wm_source}", DIM_CLR, font_xs)
    else:
        rtext("  (no query yet)", DIM_CLR, font_sm)
    rsep()

    # Near object
    near = state.info.get("semantic") is not None
    if near:
        from snks.agent.crafter_pixel_env import SEMANTIC_NAMES
        sem = state.get_semantic()
        pos = state.get_player_pos()
        # Detect nearest non-grass object
        py, px = int(pos[0]), int(pos[1])
        best = "empty"
        best_d = 999
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                ny, nx = py + dy, px + dx
                if 0 <= ny < sem.shape[0] and 0 <= nx < sem.shape[1]:
                    sid = int(sem[ny, nx])
                    nm = SEMANTIC_NAMES.get(sid, "unknown")
                    if nm not in ("grass", "path", "sand", "unknown", "player"):
                        d = abs(dy) + abs(dx)
                        if d < best_d:
                            best_d = d
                            best = nm
        rtext(f"Near:  {best}", TEXT_CLR, font_sm)

    # Thought
    rtext(f"Thought:  {state.thought[:60]}", YELLOW, font_xs)
    rsep()

    # Status
    status_color = {
        "navigating": BLUE,
        "acting":     YELLOW,
        "done":       GREEN,
        "failed":     RED,
        "idle":       DIM_CLR,
    }.get(state.status, TEXT_CLR)
    rtext(f"Status:  {state.status}", status_color, font_md)
    rsep(12)

    # Keys help
    rtext("Keys:", DIM_CLR, font_xs)
    key_lines = [
        "1-9: set goal",
        "SPACE: pause/resume",
        "R: reset env",
        "Q / ESC: quit",
    ]
    for kl in key_lines:
        rtext(f"  {kl}", DIM_CLR, font_xs)

    # Pause overlay
    if state.paused:
        surf = font_lg.render("PAUSED", True, YELLOW)
        screen.blit(surf, (FRAME_SIZE // 2 - surf.get_width() // 2,
                            FRAME_SIZE // 2 - surf.get_height() // 2))

    pygame.display.flip()


# ── Entry Point ────────────────────────────────────────────────────────────────

def run_training_phase(screen, fonts, ckpt_path: str
                       ) -> tuple[CNNEncoder, CLSWorldModel]:
    """Run training phases with progress display."""
    font_lg = fonts["lg"]
    font_sm = fonts["sm"]

    msg_holder = ["Initializing..."]

    def progress(msg: str):
        msg_holder[0] = msg
        draw_training_screen(screen, font_lg, font_sm, msg)

    # Phase 1: Collect trajectories
    progress("Phase 1: Collecting trajectories...")
    dataset = collect_trajectories(
        n_trajectories=20, steps_per_traj=100, seed=42,
        progress_callback=progress,
    )

    # Phase 2: Train encoder
    progress("Phase 2: Training encoder (30 epochs)...")
    encoder, _predictor = train_encoder(
        dataset, epochs=30, batch_size=128,
        progress_callback=progress,
    )

    # Phase 3: Collect prototypes
    cls = CLSWorldModel(dim=2048, device="cpu")
    progress("Phase 3: Collecting prototypes...")
    collect_prototypes(
        encoder, cls, n_seeds=20,
        progress_callback=progress,
    )

    # Save checkpoint
    progress(f"Saving checkpoint → {ckpt_path}")
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    save_checkpoint(ckpt_path, encoder, cls)

    draw_training_screen(screen, font_lg, font_sm,
                         "Training complete!",
                         f"Prototypes: {len(cls.prototype_memory)}  "
                         f"Neocortex: {len(cls.neocortex)} rules")
    pygame.time.wait(1500)
    return encoder, cls


def run_interactive_demo(screen, fonts, encoder: CNNEncoder, cls: CLSWorldModel):
    """Main interactive demo loop."""
    clock = pygame.time.Clock()
    state = DemoState()
    STEP_DELAY_MS = 100  # ~10 FPS agent steps

    last_agent_step = pygame.time.get_ticks()

    running = True
    while running:
        now = pygame.time.get_ticks()

        # Events
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                key = ev.key
                if key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif key == pygame.K_SPACE:
                    state.paused = not state.paused
                elif key == pygame.K_r:
                    state.reset_env()
                elif pygame.K_1 <= key <= pygame.K_9:
                    idx = key - pygame.K_1  # 0-indexed
                    state.set_goal(idx)

        # Agent step (every STEP_DELAY_MS ms, unless paused)
        if not state.paused and (now - last_agent_step) >= STEP_DELAY_MS:
            agent_step(state, encoder, cls)
            last_agent_step = now

            # Auto-reset after "done" status for a bit
            if state.status == "done":
                pygame.time.wait(800)
                state.reset_env()

        render(screen, state, fonts)
        clock.tick(30)  # render up to 30 FPS

    pygame.quit()


def main():
    parser = argparse.ArgumentParser(description="Stage 66 Interactive Demo")
    parser.add_argument("--mode", choices=["scratch", "checkpoint"],
                        default="scratch",
                        help="scratch=train from scratch; checkpoint=load saved ckpt")
    parser.add_argument("--ckpt", default="demos/checkpoints/stage66.pt",
                        help="Path to checkpoint file")
    args = parser.parse_args()

    # ── Pygame init ──
    pygame.init()
    pygame.font.init()

    try:
        screen = pygame.display.set_mode(
            (WINDOW_W, WINDOW_H), pygame.RESIZABLE
        )
    except Exception:
        screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))

    pygame.display.set_caption("Stage 66 — Pixel Crafter + Prototype Memory")

    # Fonts
    def load_font(size):
        for name in ["DejaVuSansMono", "FreeMono", "Courier New", None]:
            try:
                if name is None:
                    return pygame.font.SysFont("monospace", size)
                return pygame.font.SysFont(name, size)
            except Exception:
                continue
        return pygame.font.Font(None, size)

    fonts = {
        "lg": load_font(28),
        "md": load_font(20),
        "sm": load_font(16),
        "xs": load_font(13),
    }

    # ── Load or train ──
    ckpt_path = args.ckpt

    if args.mode == "checkpoint":
        if not os.path.exists(ckpt_path):
            draw_training_screen(screen, fonts["lg"], fonts["sm"],
                                 f"Checkpoint not found: {ckpt_path}",
                                 "Use --mode scratch to train first.")
            pygame.time.wait(3000)
            pygame.quit()
            sys.exit(1)
        draw_training_screen(screen, fonts["lg"], fonts["sm"],
                             f"Loading checkpoint: {ckpt_path}")
        pygame.display.flip()
        encoder, cls = load_checkpoint(ckpt_path)
        pygame.time.wait(500)

    else:  # scratch
        encoder, cls = run_training_phase(screen, fonts, ckpt_path)

    # ── Interactive demo ──
    run_interactive_demo(screen, fonts, encoder, cls)


if __name__ == "__main__":
    main()
