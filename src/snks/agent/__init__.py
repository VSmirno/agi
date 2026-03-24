"""Agent module: causal agent with intrinsic motivation and mental simulation."""

from snks.agent.motor import MotorEncoder
from snks.agent.causal_model import CausalWorldModel, CausalLink
from snks.agent.simulation import MentalSimulator
from snks.agent.motivation import IntrinsicMotivation
from snks.agent.agent import CausalAgent

__all__ = [
    "MotorEncoder",
    "CausalWorldModel",
    "CausalLink",
    "MentalSimulator",
    "IntrinsicMotivation",
    "CausalAgent",
]
