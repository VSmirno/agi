# Stage 64: Remove Synthetic Transitions → Demo + Curiosity

## Summary

Remove all hardcoded synthetic transition generators. Agent learns from:
1. Teacher demonstrations (≤10 Crafter rules, MiniGrid demo episodes)
2. Own exploration via curiosity (WM confidence → action selection)

Neocortex starts empty. Knowledge accumulates through experience.

**Key hypothesis:** Curiosity-driven exploration discovers rules not shown
in demos. Teacher shows wood→table→pickaxe; agent discovers stone_pickaxe
and iron_pickaxe on its own.

## Gate

- ≥80% Crafter QA
- ≤10 taught rules (teacher demos)
- ≥10 self-discovered rules through curiosity exploration (out of ~20+ unknown)
- 0 calls to generate_synthetic_transitions()
- MiniGrid regression ≥90% (from demo transitions only)

## Architecture

```
CLSWorldModel v3 (unchanged)
├── neocortex: dict[str, Rule]          # starts EMPTY
├── hippocampus: SDMMemory(2048)        # starts EMPTY
├── abstraction: AbstractionEngine      # discovers categories from experience
├── train(transitions)                  # existing — works with any transitions
└── consolidate()                       # existing — promotes verified → neocortex

CuriosityExplorer (NEW, separate class)
├── wm: CLSWorldModel                  # reference to world model
├── select_action(situation, actions)   # lowest confidence → explore
└── explore_episode(env, max_steps)     # run one episode, return discoveries
    ├── query WM → confidence
    ├── low confidence → try action
    └── write_on_surprise via wm.train()
```

## Data Sources

| Source | What | Count |
|--------|------|-------|
| MiniGrid demos (BossLevel) | pickup, toggle, forward, unlock | ~1815 transitions |
| Crafter taught demos | wood→table→wood_pickaxe chain | ≤10 rules |
| Curiosity exploration | stone_pickaxe, iron_pickaxe, swords, failures | discovered |

### Crafter Taught Demos

A subset of CRAFTER_RULES reframed as "teacher showed these":

```python
CRAFTER_TAUGHT = [
    # Basic resource collection
    {"action": "do", "near": "tree", "requires": {}, "gives": "wood", "result": "collected"},
    {"action": "do", "near": "water", "requires": {}, "gives": "drink", "result": "collected"},
    # Placing
    {"action": "place_table", "near": "empty", "requires": {"wood": 2}, "gives": "table", "result": "placed"},
    # Basic crafting
    {"action": "make_wood_pickaxe", "near": "table", "requires": {"wood": 1}, "gives": "wood_pickaxe", "result": "crafted"},
    # Tool-gated collection (one example)
    {"action": "do", "near": "stone", "requires": {"wood_pickaxe": 1}, "gives": "stone", "result": "collected"},
]
```

5 taught rules. Agent must discover:
- stone_pickaxe, iron_pickaxe (crafting with different materials)
- wood_sword, stone_sword, iron_sword (weapon crafting)
- coal, iron, diamond collection (tool-gated resources)
- place_furnace, place_plant, place_stone (placing variants)
- All failure cases (no tool, no resource, no station)

## Curiosity Explorer

### Core: confidence-based action selection

```python
class CuriosityExplorer:
    def __init__(self, wm: CLSWorldModel, explore_threshold: float = 0.3):
        self.wm = wm
        self.explore_threshold = explore_threshold

    def select_action(self, situation: dict, available_actions: list[str]) -> str:
        """Pick action with lowest WM confidence — explore the unknown.

        Random tiebreaking when multiple actions have equal (e.g. zero) confidence.
        """
        import random
        candidates = []
        worst_conf = 1.0
        for action in available_actions:
            outcome, conf, source = self.wm.query(situation, action)
            if conf < worst_conf:
                worst_conf = conf
                candidates = [action]
            elif conf == worst_conf:
                candidates.append(action)
        return random.choice(candidates)

    def explore_episode(self, env, max_steps: int = 50) -> list[Transition]:
        """One exploration episode. Returns discovered transitions."""
        discovered = []
        for step in range(max_steps):
            situation = env.observe()
            action = self.select_action(situation, env.available_actions())
            outcome, reward = env.step(action)
            t = Transition(situation=situation, action=action,
                          outcome=outcome, reward=reward)
            # Check if this was surprising (not already known)
            known_outcome, conf, _ = self.wm.query(situation, action)
            if conf < self.explore_threshold or known_outcome.get("result") != outcome.get("result"):
                discovered.append(t)
            # Train incrementally
            self.wm.train([t])
        return discovered
```

### Exploration strategy

Per episode:
1. Pick random nearby object (tree, stone, table, furnace, etc.)
2. Try all available actions with current inventory
3. Low-confidence outcomes → try them → learn
4. After each action, retrain WM incrementally

Over episodes:
- Episode 1-5: explore basic interactions (chop, place)
- Episode 6-15: explore crafting (have tools, try new recipes)
- Episode 16+: explore with advanced inventory (stone tools, iron tools)

Inventory builds up naturally: chop tree → wood → table → pickaxe → mine stone → ...

## Crafter Symbolic Environment

Full Crafter is a pixel-based game. For Stage 64, use a symbolic wrapper
that simulates the tech tree without rendering:

```python
class CrafterSymbolicEnv:
    """Symbolic Crafter tech tree for curiosity exploration."""

    # All nearby targets the agent can encounter
    ALL_NEARBY = ["tree", "stone", "coal", "iron", "diamond",
                  "water", "cow", "table", "furnace", "empty"]

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)
        self._all_rules = CRAFTER_RULES  # ground truth, hidden from agent
        self._all_failures = CRAFTER_FAILURES
        self.reset()

    def reset(self) -> dict[str, str]:
        """Reset inventory and pick random nearby target."""
        self.inventory: dict[str, int] = {}
        self._nearby_idx = 0
        self._rng.shuffle(self.ALL_NEARBY[:])  # randomize encounter order
        self._nearby_order = list(self.ALL_NEARBY)
        self._rng.shuffle(self._nearby_order)
        return self.observe()

    def next_target(self) -> None:
        """Move to next nearby target (called between actions or by explorer)."""
        self._nearby_idx = (self._nearby_idx + 1) % len(self._nearby_order)

    def observe(self) -> dict[str, str]:
        """Current situation as dict."""
        situation = {
            "domain": "crafter",
            "near": self._nearby_order[self._nearby_idx],
        }
        for item, count in self.inventory.items():
            situation[f"has_{item}"] = str(count)
        return situation

    def available_actions(self) -> list[str]:
        """All possible actions (agent doesn't know which will succeed)."""
        return ["do", "place_table", "place_furnace", "place_plant",
                "place_stone", "make_wood_pickaxe", "make_stone_pickaxe",
                "make_iron_pickaxe", "make_wood_sword", "make_stone_sword",
                "make_iron_sword"]

    def step(self, action: str) -> tuple[dict, float]:
        """Execute action, return (outcome, reward).

        On success: consume required items, add produced item to inventory.
        On failure: no inventory change.
        """
        situation = self.observe()
        # Check against ground truth rules
        for rule in self._all_rules:
            if self._matches(rule, action, situation):
                # Consume required resources
                for item, count in rule.get("requires", {}).items():
                    self.inventory[item] = self.inventory.get(item, 0) - count
                    if self.inventory[item] <= 0:
                        del self.inventory[item]
                # Add produced item
                gives = rule["gives"]
                self.inventory[gives] = self.inventory.get(gives, 0) + 1
                return {"result": rule["result"], "gives": gives}, 1.0
        # Check failure rules
        for fail in self._all_failures:
            if self._matches_failure(fail, action, situation):
                return {"result": fail["result"]}, -1.0
        return {"result": "nothing_happened"}, 0.0
```

Key: CRAFTER_RULES is the hidden ground truth of the environment.
The agent never sees these rules directly — only observes outcomes.

## Changes to CLSWorldModel

Minimal changes:
- No new methods needed — `train()` already works with any list of Transitions
- `explore()` is in CuriosityExplorer, not in WM itself
- Abstraction engine auto-discovers categories from whatever rules accumulate

The only change: exp120 does NOT call `generate_synthetic_transitions()`.

## Files

| File | Change |
|------|--------|
| `src/snks/agent/curiosity_explorer.py` | NEW: CuriosityExplorer class |
| `src/snks/agent/crafter_env_symbolic.py` | NEW: symbolic Crafter env |
| `src/snks/agent/crafter_trainer.py` | Add CRAFTER_TAUGHT (subset), keep full rules as env ground truth |
| `tests/test_stage64_curiosity.py` | Unit tests: curiosity discovers rules |
| `experiments/exp120_no_synthetic.py` | Gate experiment |

## What does NOT change

- `cls_world_model.py` — train/query/consolidate unchanged
- `abstraction_engine.py` — works on whatever neocortex contains
- `world_model_trainer.py` — extract_demo_transitions() unchanged
- `vsa_world_model.py` — SDM/VSA unchanged

## MiniGrid Regression

MiniGrid demos (BossLevel) contain ~1815 transitions covering:
- forward (moved, blocked)
- pickup (picked_up)
- toggle (door_opened)
- Some unlock transitions

This should be sufficient for ≥90% MiniGrid QA without synthetic.
Missing coverage: some edge cases (drop, locked door with wrong key)
may require abstract generalization from categories.
