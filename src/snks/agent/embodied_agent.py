"""EmbodiedAgent: thin orchestration layer integrating CausalAgent +
StochasticSimulator + Configurator FSM output (Stage 14).

Architecture:
    EmbodiedAgent
    ├── CausalAgent           # owns Pipeline (MetaEmbedder, HACPredictor,
    │                         #   IntrinsicCostModule, Configurator inside
    │                         #   perception_cycle())
    ├── StochasticSimulator   # Stage 11: N-sample Monte Carlo planning
    └── EmbodiedAgentConfig   # ablation flags + planner params

CausalAgent is NOT modified. EmbodiedAgent reads CycleResult.configurator_action
from Pipeline.last_cycle_result (cached after each perception_cycle) and
uses it to override action selection.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

from snks.agent.agent import CausalAgent, _perceptual_hash
from snks.agent.stochastic_simulator import StochasticSimulator
from snks.daf.types import CausalAgentConfig


@dataclass
class EmbodiedAgentConfig:
    """Configuration for EmbodiedAgent (Stage 14).

    Ablation is controlled by setting:
        causal.pipeline.configurator.enabled = False
        causal.pipeline.cost_module.enabled  = False
    before constructing the agent.
    """
    causal: CausalAgentConfig = field(default_factory=CausalAgentConfig)
    use_stochastic_planner: bool = True
    n_plan_samples: int = 8
    max_plan_depth: int = 5
    goal_cost_value: float = 1.0  # passed to icm.set_goal_cost() when GOAL_SEEKING


class EmbodiedAgent:
    """Thin orchestration wrapper over CausalAgent.

    step() cycle:
        1. causal_agent.step(obs) → runs ALL Stage 6-13 components, returns action
        2. Read configurator mode from pipeline.last_cycle_result
        3. Update ICM goal cost for the next cycle
        4. Override action by mode:
           - explore      → random action
           - goal_seeking → stochastic planner (if _goal_sks set) or CausalAgent default
           - else         → CausalAgent default

    observe_result() delegates directly to causal_agent.observe_result().
    """

    def __init__(self, config: EmbodiedAgentConfig) -> None:
        self.config = config
        self.causal_agent = CausalAgent(config.causal)
        self.simulator = StochasticSimulator(self.causal_agent.causal_model)
        self.n_actions: int = CausalAgent.N_ACTIONS
        # Goal SKS for stochastic planning. Set externally via set_goal_sks().
        self._goal_sks: set[int] | None = None

    def set_goal_sks(self, sks: set[int] | None) -> None:
        """Set goal state SKS for stochastic planning.

        Args:
            sks: SKS cluster IDs representing the goal state, or None to clear.
        """
        self._goal_sks = sks

    def step(self, obs: np.ndarray) -> int:
        """One agent step.

        Args:
            obs: RGB observation from environment (H, W, 3) uint8.

        Returns:
            action integer in [0, n_actions).
        """
        # 1. CausalAgent runs ALL Stage 6-13 components via Pipeline.perception_cycle()
        action = self.causal_agent.step(obs)
        result = self.causal_agent.pipeline.last_cycle_result

        if result is None:
            return action

        sks = set(result.sks_clusters.keys())
        # Augment with perceptual hash for stable goal matching (mirrors CausalAgent)
        image = self.causal_agent.obs_adapter.convert(obs)
        sks |= _perceptual_hash(image)
        conf_action = result.configurator_action
        mode = conf_action.mode if conf_action is not None else "neutral"

        # 2. Update goal cost for next ICM cycle.
        # Set based on whether _goal_sks is known (not on current mode) to break
        # the circular dependency: GOAL_SEEKING requires cost.goal > threshold,
        # but goal_cost would only be set positive if already in GOAL_SEEKING.
        if self._goal_sks is not None:
            self.causal_agent.pipeline.cost_module.set_goal_cost(
                self.config.goal_cost_value
            )
        else:
            self.causal_agent.pipeline.cost_module.set_goal_cost(0.0)

        # 3. Action selection by mode
        if mode == "goal_seeking" and self.config.use_stochastic_planner and self._goal_sks:
            plan, _ = self.simulator.find_plan_stochastic(
                sks,
                self._goal_sks,
                n_actions=self.n_actions,
                n_samples=self.config.n_plan_samples,
                max_depth=self.config.max_plan_depth,
            )
            if plan:
                return plan[0]

        if mode == "explore":
            return random.randint(0, self.n_actions - 1)

        # consolidate / neutral / goal_seeking fallback: CausalAgent default
        return action

    def observe_result(self, obs: np.ndarray) -> float:
        """Delegate to CausalAgent.observe_result().

        Args:
            obs: Next observation (H, W, 3) uint8 from env.step(action).

        Returns:
            prediction_error (float).
        """
        return self.causal_agent.observe_result(obs)
