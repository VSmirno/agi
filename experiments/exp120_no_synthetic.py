#!/usr/bin/env python3
"""Exp120: Stage 64 gate — no synthetic transitions.

Gate:
- ≥80% Crafter QA
- ≤10 taught rules
- ≥10 self-discovered rules through curiosity
- 0 calls to generate_synthetic_transitions()
- MiniGrid regression ≥90%
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import json
import torch

from snks.agent.cls_world_model import CLSWorldModel
from snks.agent.crafter_trainer import (
    generate_taught_transitions,
    CRAFTER_TAUGHT,
    CRAFTER_RULES,
)
from snks.agent.crafter_env_symbolic import CrafterSymbolicEnv
from snks.agent.curiosity_explorer import CuriosityExplorer
from snks.agent.world_model_trainer import extract_demo_transitions


def main():
    print("=" * 60)
    print("EXP120: Stage 64 — No Synthetic, Demo + Curiosity")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Phase 1: Teacher demos only (NO synthetic!)
    taught = generate_taught_transitions()
    print(f"Taught rules: {len(taught)} (max 10)")

    # MiniGrid demos
    with open("_docs/demo_episodes_bosslevel.json") as f:
        demos = json.load(f)
    mg_demo = extract_demo_transitions(demos)
    print(f"MiniGrid demo transitions: {len(mg_demo)}")

    # Train initial WM
    model = CLSWorldModel(dim=2048, n_locations=5000, device=device)
    initial_stats = model.train(taught + mg_demo)
    initial_neocortex = len(model.neocortex)
    print(f"Initial training: {initial_stats}")
    print(f"Initial neocortex rules: {initial_neocortex}")

    # Phase 2: Curiosity exploration
    print(f"\n{'='*60}")
    print("CURIOSITY EXPLORATION")
    env = CrafterSymbolicEnv(seed=42)
    explorer = CuriosityExplorer(model, explore_threshold=0.3, seed=42)

    all_discovered = []
    for batch in range(5):  # 5 batches of 20 episodes
        discovered = explorer.explore(env, n_episodes=20, steps_per_episode=80)
        all_discovered.extend(discovered)
        print(f"  Batch {batch+1}: +{len(discovered)} discoveries, "
              f"neocortex={len(model.neocortex)}")

    # Rebuild abstraction after all exploration
    model.abstraction.discover_categories(model.neocortex)
    model.abstraction.build_abstract_sdm()

    final_neocortex = len(model.neocortex)
    new_rules = final_neocortex - initial_neocortex
    print(f"\nTotal discoveries: {len(all_discovered)}")
    print(f"Neocortex growth: {initial_neocortex} → {final_neocortex} (+{new_rules})")

    # Identify which Crafter rules were discovered (not taught)
    taught_keys = {(r["action"], r["near"]) for r in CRAFTER_TAUGHT}
    all_rule_keys = {(r["action"], r["near"]) for r in CRAFTER_RULES}
    discovered_crafter = set()
    for t in all_discovered:
        if t.situation.get("domain") == "crafter":
            key = (t.action, t.situation.get("near", ""))
            if key not in taught_keys and key in all_rule_keys:
                if t.outcome.get("result") not in ("nothing_happened",):
                    discovered_crafter.add(key)

    print(f"Self-discovered Crafter rules: {len(discovered_crafter)}")
    for ak in sorted(discovered_crafter):
        print(f"  {ak[0]} near {ak[1]}")

    # Phase 3: Crafter QA
    print(f"\n{'='*60}")
    print("CRAFTER QA")
    from exp119_abstraction import test_crafter_qa
    c_ok, c_total, c_fails = test_crafter_qa(model)
    for f in c_fails:
        print(f)
    print(f"Crafter QA: {c_ok}/{c_total} = {c_ok*100//c_total}%  [gate ≥80%]")

    # Phase 4: MiniGrid regression
    print(f"\n{'='*60}")
    print("MINIGRID QA (regression)")
    from exp118_world_model_qa import (
        level1_object_identity, level2_preconditions,
        level3_consequences, level4_planning,
    )
    c3, t3, f3 = level1_object_identity(model)
    c4, t4, f4 = level2_preconditions(model)
    c5, t5, f5 = level3_consequences(model)
    c6, t6, f6 = level4_planning(model)
    for f in f3 + f4 + f5 + f6:
        print(f)
    mg_total = c3 + c4 + c5 + c6
    mg_max = t3 + t4 + t5 + t6
    print(f"MiniGrid: {mg_total}/{mg_max} = {mg_total*100//mg_max}%")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print("=" * 60)
    crafter_pct = c_ok * 100 // c_total
    mg_pct = mg_total * 100 // mg_max
    print(f"Taught rules: {len(taught)} (gate ≤10)")
    print(f"Discovered Crafter rules: {len(discovered_crafter)} (gate ≥10)")
    print(f"Crafter QA: {c_ok}/{c_total} = {crafter_pct}%  (gate ≥80%)")
    print(f"MiniGrid QA: {mg_total}/{mg_max} = {mg_pct}%  (gate ≥90%)")
    print(f"Categories: {len(model.abstraction.categories)}")

    taught_pass = len(taught) <= 10
    discovered_pass = len(discovered_crafter) >= 10
    crafter_pass = crafter_pct >= 80
    mg_pass = mg_pct >= 90

    print(f"\nTaught ≤10: {'PASS' if taught_pass else 'FAIL'}")
    print(f"Discovered ≥10: {'PASS' if discovered_pass else 'FAIL'}")
    print(f"Crafter ≥80%: {'PASS' if crafter_pass else 'FAIL'}")
    print(f"MiniGrid ≥90%: {'PASS' if mg_pass else 'FAIL'}")
    print(f"ALL GATES: {'PASS' if all([taught_pass, discovered_pass, crafter_pass, mg_pass]) else 'FAIL'}")


if __name__ == "__main__":
    main()
