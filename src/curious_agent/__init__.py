"""
Curious Agent RL - Implementation of Schmidhuber's 1991 Curious Model-Building Control System

This package implements a curious agent that uses intrinsic motivation to explore
its environment by improving its world model predictions.

Based on: Jürgen Schmidhuber, "Curious Model-Building Control Systems",
Proc. International Joint Conference on Neural Networks, Singapore, 1991.
"""

__version__ = "0.1.0"
__author__ = "Nahomsen Ayele"

from curious_agent.env.grid_world import GridWorld
from curious_agent.agents.tabular_curious import TabularCuriousAgent
from curious_agent.agents.dqn_curious import DNQCuriousAgent

__all__ = [
    "GridWorld",
    "TabularCuriousAgent",
    "DNQCuriousAgent",
]