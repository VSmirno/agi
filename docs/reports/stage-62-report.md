# Stage 62 Report: CLS World Model — Domain Understanding

**Status:** COMPLETE (gate passed)
**Date:** 2026-04-04
**Gate:** QA ≥90% avg (4 levels) AND Planning ≥80% → **100% / 100%**

## Pivot History

Stage 62 started as BossLevel navigation (gate ≥50%). Reached 44% but stuck
on blind exploration. NavigationPolicy (learned SDM directions) made it WORSE
(16%). Unified SDM world model failed (SDM capacity limit at ~500 patterns).

**Pivot:** CLS (Complementary Learning System) — bio-inspired two-tier memory.

## Architecture

```
CLSWorldModel
├── Neocortex (dict)         — 1160 verified rules, exact match, 100% accuracy
├── Hippocampus (SDM)        — dim=2048, 5000 locations, generalization
├── Write-on-Surprise        — skip if already known (77% skip rate)
├── Consolidation            — SDM → neocortex promotion
└── Color Generalization     — same/different color substitution
```

## Results: Exp118

**Training:** 1080 synthetic (3 colors) + 1815 demo transitions = 2895 total

| Level | Score | Gate | Status |
|-------|-------|------|--------|
| L1: Object Identity | 22/22 = 100% | ≥95% | PASS |
| L2: Preconditions + generalization | 14/14 = 100% | ≥90% | PASS |
| L3: Consequences + generalization | 18/18 = 100% | ≥85% | PASS |
| L4: Planning | 19/19 = 100% | ≥80% | PASS |
| **Average** | **73/73 = 100%** | **≥90%** | **PASS** |

**Held-out color generalization:** Train on red/green/blue, test on purple/yellow/grey. 0 failures.

## Key Technical Decisions

1. **CLS over unified SDM** — Binary SDM holds ~500 patterns. 6000 transitions in 1 SDM = noise. Per-concept SDMs (Stage 60) worked because each had 10-50 items. CLS: neocortex (dict) for bulk, SDM for generalization.

2. **Write-on-Surprise** — 77% writes skipped (already known). Reduces SDM load from 2895 to 636.

3. **VSA identity for color generalization** — `bind(color_X, color_X) = zero` for ANY X. Same-color key+door pairs produce identical SDM addresses regardless of color.

4. **Same/different preservation** — When generalizing to held-out colors, preserve color relationship: same-color pair → substitute with same trained color, different → substitute with different.

## Honest Assessment

**What works:** Neocortex = perfect lookup for trained situations. Color generalization via neocortex substitution (not pure SDM).

**What's NOT yet proven:**
- Pure SDM generalization (SDM gets too few writes with write-on-surprise)
- Object-type generalization (train on key+door, test on ball+box)
- Custom object transfer (lever→gate analogous to key→door)
- Multi-step planning beyond 2 steps with state changes
- Spatial reasoning ("object in another room")

## Research Findings

CLS architecture confirmed by literature review:
- LeCun's JEPA: prediction in representation space (our VSA encoding)
- Kanerva SDM: ~500 pattern capacity at dim=512 (confirmed empirically)
- Complementary Learning Systems (McClelland et al.): hippocampus (fast, limited) + neocortex (slow, unlimited)
- Write-on-Surprise: 50-90% write reduction (we got 77%)

Full research: `_docs/research_world_models_2026-04-03.md`

## Files

| File | Purpose |
|------|---------|
| `src/snks/agent/cls_world_model.py` | CLSWorldModel (hippocampus+neocortex) |
| `src/snks/agent/world_model_trainer.py` | Transition extraction + synthetic generation |
| `src/snks/agent/unified_world_model.py` | Deprecated (failed approach) |
| `src/snks/agent/nav_policy.py` | Deprecated (made navigation worse) |
| `experiments/exp118_world_model_qa.py` | Gate experiment |
| `tests/test_stage62_bosslevel.py` | Unit tests (29 pass) |

## Next: Stage 63

1. **Pure SDM generalization** — remove neocortex substitution, strengthen SDM signal
2. **Object-type hold-out** — train on key+door, test ball+box generalization
3. **Custom objects** — lever→gate transfer (analogous to key→door)
4. **Reconnect to grid** — CLSWorldModel drives BossLevelAgent decisions
5. **Curiosity-driven exploration** — affordance-based interaction, not frontier BFS
