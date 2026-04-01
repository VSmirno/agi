"""Smoke test: verify torch.compile works on GPU without hanging."""

import time
import sys
sys.stdout.reconfigure(encoding="utf-8")

from snks.agent.pure_daf_agent import PureDafAgent, PureDafConfig
from snks.env.adapter import MiniGridAdapter


def main():
    cfg = PureDafConfig()
    cfg.n_actions = 5
    cfg.max_episode_steps = 10
    cfg.causal.pipeline.daf.device = "auto"
    cfg.causal.pipeline.daf.disable_csr = True
    cfg.causal.pipeline.daf.dt = 0.002
    cfg.causal.pipeline.steps_per_cycle = 200
    cfg.causal.pipeline.sks.coherence_mode = "cofiring"
    cfg.causal.pipeline.sks.top_k = 2000
    cfg.causal.pipeline.daf.fhn_I_base = 0.3
    cfg.causal.pipeline.daf.coupling_strength = 0.05

    print("Creating agent...")
    t0 = time.time()
    agent = PureDafAgent(cfg)
    print("Agent created in %.1fs" % (time.time() - t0))

    adapter = MiniGridAdapter("MiniGrid-Empty-5x5-v0")
    print("Running 1 episode (10 steps max)...")
    t0 = time.time()
    result = agent.run_episode(adapter, max_steps=10)
    print("Episode done in %.1fs, steps=%d" % (time.time() - t0, result.steps))
    print("SMOKE TEST PASS")


if __name__ == "__main__":
    main()
