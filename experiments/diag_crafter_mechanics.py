"""Diagnostic: observe real Crafter env mechanics step-by-step.

Runs CrafterPixelEnv with trivial actions and logs body state every tick
to reveal actual decay/damage rates. This evidence is for systematic
debugging of the Stage 77a survival wall (~170 steps, cause=health,
but with food/drink NOT at zero — something else is killing the agent).

Runs multiple scenarios:
  1. NO-OP (stand still) — pure decay, no action effects
  2. RANDOM walk — exploration with no intent
  3. MOVE_RIGHT — constant direction
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from snks.agent.crafter_pixel_env import CrafterPixelEnv


def run_scenario(
    name: str,
    action_fn,
    max_steps: int = 500,
    seed: int = 42,
    disable_enemies: bool = False,
) -> None:
    """Run one diagnostic scenario and log body trajectory."""
    print(f"\n{'='*60}\n{name} (seed={seed}, disable_enemies={disable_enemies})\n{'='*60}")
    env = CrafterPixelEnv(seed=seed)
    if disable_enemies:
        try:
            env._env._balance_chunk = lambda *a, **kw: None
        except Exception:
            pass

    pixels, info = env.reset()
    rng = np.random.RandomState(seed)
    prev_inv = dict(info.get("inventory", {}))

    # Print header
    body_keys = ["health", "food", "drink", "energy"]
    print(f"{'step':>5} {'action':<15} {'H':>3} {'F':>3} {'D':>3} {'E':>3} {'wood':>5} {'cause':>20}")

    last_print_step = 0
    for step in range(max_steps):
        action = action_fn(step, rng)
        pixels, reward, done, info = env.step(action)
        inv = dict(info.get("inventory", {}))

        # Print on body change or every 10 steps
        body_changed = any(inv.get(k, 0) != prev_inv.get(k, 0) for k in body_keys)
        if body_changed or step - last_print_step >= 20 or done:
            delta = {k: inv.get(k, 0) - prev_inv.get(k, 0) for k in body_keys}
            delta_str = " ".join(f"{k[0].upper()}{'+' if v > 0 else ''}{v}" for k, v in delta.items() if v != 0)
            cause = ""
            if done:
                zeroed = [k for k in body_keys if inv.get(k, 0) <= 0]
                cause = f"DONE: {','.join(zeroed) or 'other'}"
            print(
                f"{step:>5} {action:<15} "
                f"{inv.get('health', 0):>3} {inv.get('food', 0):>3} "
                f"{inv.get('drink', 0):>3} {inv.get('energy', 0):>3} "
                f"{inv.get('wood', 0):>5} {delta_str:>20} {cause}"
            )
            last_print_step = step

        prev_inv = inv
        if done:
            print(f"EPISODE END at step {step + 1}")
            break


def action_noop(step: int, rng: np.random.RandomState) -> str:
    """Always do nothing (no-op). Pure decay observation."""
    return "do"  # "do" on empty tile is effectively no-op


def action_standstill(step: int, rng: np.random.RandomState) -> str:
    """Move into a wall — player doesn't move but steps tick."""
    return "move_up"  # repeating one direction, might hit boundary


def action_random(step: int, rng: np.random.RandomState) -> str:
    """Random move each step."""
    return rng.choice(["move_up", "move_down", "move_left", "move_right"])


def action_move_right(step: int, rng: np.random.RandomState) -> str:
    return "move_right"


def main():
    # Scenario 1: stand still, disable enemies (what warmup-safe should look like)
    run_scenario(
        "NO-OP with enemies disabled",
        action_noop,
        max_steps=400,
        seed=42,
        disable_enemies=True,
    )

    # Scenario 2: stand still, enemies enabled (to see natural deaths)
    run_scenario(
        "NO-OP with enemies enabled",
        action_noop,
        max_steps=400,
        seed=42,
        disable_enemies=False,
    )

    # Scenario 3: random walk, enemies disabled
    run_scenario(
        "Random walk, enemies disabled",
        action_random,
        max_steps=400,
        seed=42,
        disable_enemies=True,
    )

    # Scenario 4: cycle directions (our MPC default)
    def action_cycle(step, rng):
        return ["move_up", "move_right", "move_down", "move_left"][step % 4]

    run_scenario(
        "Direction cycling, enemies disabled",
        action_cycle,
        max_steps=400,
        seed=42,
        disable_enemies=True,
    )


if __name__ == "__main__":
    main()
