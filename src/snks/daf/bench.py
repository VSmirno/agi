"""Benchmark utility for DAF Engine step throughput."""

from __future__ import annotations

import time

import torch

from snks.daf.engine import DafEngine
from snks.daf.types import DafConfig
from snks.device import get_device


def run_bench(
    config: DafConfig | None = None,
    n_steps: int = 200,
    n_cycles: int = 5,
    enable_learning: bool = True,
) -> dict:
    """Benchmark DafEngine.step() throughput.

    Measures CUDA Graph replay performance (not capture).

    Args:
        config: DafConfig (default: 50K nodes, avg_degree 50).
        n_steps: integration steps per cycle.
        n_cycles: number of cycles to measure (average).
        enable_learning: include STDP + homeostasis.

    Returns:
        dict with keys: steps_sec, total_sec, n_steps, num_nodes, num_edges, device
    """
    if config is None:
        config = DafConfig(num_nodes=50_000, avg_degree=50)

    engine = DafEngine(config, enable_learning=enable_learning)

    # Inject stimulus
    stim = torch.zeros(config.num_nodes, device=engine.device)
    stim[:config.num_nodes // 10] = 1.0
    engine.set_input(stim)

    # Warmup: first call captures CUDA graph, second verifies replay
    engine.step(n_steps)
    engine.step(n_steps)

    # Sync before timing
    if engine.device.type == "cuda":
        torch.cuda.synchronize(engine.device)

    t0 = time.perf_counter()
    for _ in range(n_cycles):
        engine.step(n_steps)
    if engine.device.type == "cuda":
        torch.cuda.synchronize(engine.device)
    elapsed = time.perf_counter() - t0

    total_steps = n_steps * n_cycles
    steps_sec = total_steps / elapsed

    return {
        "steps_sec": steps_sec,
        "total_sec": elapsed,
        "n_steps": n_steps,
        "n_cycles": n_cycles,
        "total_steps": total_steps,
        "num_nodes": config.num_nodes,
        "num_edges": engine.graph.num_edges,
        "device": str(engine.device),
    }


def main() -> None:
    """Run benchmarks for small and default configurations."""
    configs = {
        "small (10K, deg=30)": DafConfig(num_nodes=10_000, avg_degree=30),
        "default (50K, deg=50)": DafConfig(num_nodes=50_000, avg_degree=50),
    }

    device = get_device("auto")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
        props = torch.cuda.get_device_properties(device)
        print(f"VRAM: {props.total_memory / 1024**3:.1f} GB")
    print()

    for name, cfg in configs.items():
        cfg.device = str(device)
        result = run_bench(cfg, n_steps=200, n_cycles=5)
        print(f"{name}:")
        print(f"  {result['num_nodes']:,} nodes, {result['num_edges']:,} edges")
        print(f"  {result['steps_sec']:,.0f} steps/sec "
              f"({result['total_sec']:.3f}s for {result['total_steps']} steps, "
              f"{result['n_cycles']} cycles)")
        gate = "PASS" if result["steps_sec"] >= 10_000 else "FAIL"
        print(f"  Gate (>=10K steps/sec): {gate}")
        print()


if __name__ == "__main__":
    main()
