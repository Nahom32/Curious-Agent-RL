"""Vanilla Deep Q-Network agent.

This module is the external-reward-only ablation for the curious DQN. It does
not construct a world model, confidence network, or intrinsic-reward module.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from curious_agent.models.q_network import DuelingQNetwork, QNetwork
from curious_agent.utils.replay_buffer import ReplayBuffer

logger = logging.getLogger(__name__)


class DQNAgent:
    """Standard replay-buffer DQN trained only on environment rewards."""

    CHECKPOINT_VERSION = 1

    def __init__(
        self,
        state_dim: int = 2,
        num_actions: int = 4,
        config: dict[str, Any] | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.config = config or {}
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        agent_config = self.config.get("agent", {})
        self.gamma = float(agent_config.get("gamma", 0.99))
        self.epsilon = float(agent_config.get("epsilon", 1.0))
        self.epsilon_min = float(agent_config.get("epsilon_min", 0.01))
        self.epsilon_decay = float(agent_config.get("epsilon_decay", 0.9995))

        q_config = self.config.get("networks", {}).get("q_network", {})
        network_type = (
            DuelingQNetwork if q_config.get("use_dueling", False) else QNetwork
        )
        network_args: dict[str, Any] = {
            "state_dim": state_dim,
            "num_actions": num_actions,
            "hidden_dims": q_config.get("hidden_dims", [128, 128]),
            "learning_rate": q_config.get("learning_rate", 0.001),
        }
        if network_type is DuelingQNetwork:
            network_args.update(
                value_hidden_dim=q_config.get("value_hidden_dim", 64),
                advantage_hidden_dim=q_config.get("advantage_hidden_dim", 64),
            )

        self.q_network = network_type(**network_args).to(self.device)
        self.target_q_network = copy.deepcopy(self.q_network).to(self.device)
        self.target_q_network.eval()
        for parameter in self.target_q_network.parameters():
            parameter.requires_grad_(False)

        training_config = self.config.get("training", {})
        self.replay_buffer = ReplayBuffer(
            capacity=int(training_config.get("buffer_size", 10000))
        )
        self.batch_size = int(training_config.get("batch_size", 64))
        self.min_buffer_size = int(training_config.get("min_buffer_size", 1000))
        self.target_update_frequency = int(
            training_config.get("target_update_frequency", 100)
        )
        self.tau = float(training_config.get("tau", 0.005))
        self.gradient_steps = 0
        self.q_losses: list[float] = []

        logger.info(
            "DQNAgent initialized: state_dim=%s, num_actions=%s, device=%s",
            state_dim,
            num_actions,
            self.device,
        )

    def _encode_state(self, state: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(state, dtype=torch.float32, device=self.device)

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        """Select an epsilon-greedy action, or a greedy action for evaluation."""
        epsilon = 0.0 if evaluate else self.epsilon
        if np.random.random() < epsilon:
            return int(np.random.randint(0, self.num_actions))
        return self.q_network.get_action(self._encode_state(state), epsilon=0.0)

    def store_experience(
        self,
        state: np.ndarray,
        action: int,
        external_reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store an environment transition without reward shaping."""
        self.replay_buffer.push(
            state, action, external_reward, next_state, done
        )

    def update_q_network(self) -> float | None:
        """Perform one DQN update once the replay warm-up is complete."""
        if not self.replay_buffer.is_ready(self.min_buffer_size):
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size, device=self.device
        )
        loss = self.q_network.update(
            states,
            actions,
            rewards,
            next_states,
            dones,
            gamma=self.gamma,
            target_network=self.target_q_network,
        )
        self.q_losses.append(loss)
        self.gradient_steps += 1

        if self.gradient_steps % self.target_update_frequency == 0:
            self._update_target_network()
        return loss

    def _update_target_network(self) -> None:
        with torch.no_grad():
            for target_parameter, online_parameter in zip(
                self.target_q_network.parameters(), self.q_network.parameters()
            ):
                target_parameter.mul_(1.0 - self.tau)
                target_parameter.add_(online_parameter, alpha=self.tau)

    def decay_epsilon(self) -> None:
        self.epsilon = max(
            self.epsilon_min, self.epsilon * self.epsilon_decay
        )

    def get_statistics(self) -> dict[str, float | int]:
        return {
            "epsilon": self.epsilon,
            "gradient_steps": self.gradient_steps,
            "buffer_size": len(self.replay_buffer),
            "avg_q_loss": (
                float(np.mean(self.q_losses[-100:])) if self.q_losses else 0.0
            ),
        }

    def save(self, path: str | Path) -> None:
        """Save a versioned vanilla-DQN checkpoint."""
        torch.save(
            {
                "checkpoint_version": self.CHECKPOINT_VERSION,
                "agent_type": "vanilla_dqn",
                "state_dim": self.state_dim,
                "num_actions": self.num_actions,
                "q_network": self.q_network.state_dict(),
                "target_q_network": self.target_q_network.state_dict(),
                "optimizer": self.q_network.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "gradient_steps": self.gradient_steps,
            },
            Path(path),
        )

    def load(self, path: str | Path) -> None:
        checkpoint = torch.load(Path(path), map_location=self.device)
        if checkpoint.get("agent_type") != "vanilla_dqn":
            raise ValueError("Checkpoint is not a vanilla DQN checkpoint")
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_q_network.load_state_dict(checkpoint["target_q_network"])
        self.q_network.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = float(checkpoint["epsilon"])
        self.gradient_steps = int(checkpoint["gradient_steps"])

    def __repr__(self) -> str:
        return (
            f"DQNAgent(state_dim={self.state_dim}, "
            f"num_actions={self.num_actions}, epsilon={self.epsilon:.3f}, "
            f"device={self.device})"
        )
