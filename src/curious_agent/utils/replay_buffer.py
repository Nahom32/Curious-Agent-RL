"""
Replay Buffer for DQN Training

Stores experiences (state, action, reward, next_state, done) and provides
mini-batch sampling for training the Q-network.
"""

import logging
import random
from collections import deque, namedtuple
from typing import List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Experience tuple
Experience = namedtuple(
    "Experience", ["state", "action", "reward", "next_state", "done"]
)


class ReplayBuffer:
    """
    Experience replay buffer for DQN training.
    
    Stores transitions and provides random sampling for mini-batch training.
    """
    
    def __init__(self, capacity: int = 10000):
        """
        Initialize the replay buffer.
        
        Args:
            capacity: Maximum number of experiences to store
        """
        self.capacity = capacity
        self.buffer: deque[Experience] = deque(maxlen=capacity)
        
        # Statistics
        self.total_added = 0
        
        logger.info(f"ReplayBuffer initialized: capacity={capacity}")
    
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        Add an experience to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        experience = Experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
        )
        
        self.buffer.append(experience)
        self.total_added += 1
        
        logger.debug(
            f"Experience added: state={state}, action={action}, "
            f"reward={reward:.4f}, done={done}"
        )
    
    def sample(
        self, batch_size: int, device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a mini-batch from the buffer.
        
        Args:
            batch_size: Number of experiences to sample
            device: Device to place tensors on
        
        Returns:
            Tuple of (states, actions, rewards, next_states, dones) tensors
        """
        if len(self.buffer) < batch_size:
            raise ValueError(
                f"Buffer has {len(self.buffer)} experiences, "
                f"but batch_size is {batch_size}"
            )
        
        # Sample experiences
        experiences = random.sample(self.buffer, batch_size)
        
        # Unzip experiences
        states = np.array([e.state for e in experiences])
        actions = np.array([e.action for e in experiences])
        rewards = np.array([e.reward for e in experiences])
        next_states = np.array([e.next_state for e in experiences])
        dones = np.array([e.done for e in experiences], dtype=np.float32)
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(states)
        actions_tensor = torch.LongTensor(actions)
        rewards_tensor = torch.FloatTensor(rewards)
        next_states_tensor = torch.FloatTensor(next_states)
        dones_tensor = torch.FloatTensor(dones)
        
        # Move to device if specified
        if device is not None:
            states_tensor = states_tensor.to(device)
            actions_tensor = actions_tensor.to(device)
            rewards_tensor = rewards_tensor.to(device)
            next_states_tensor = next_states_tensor.to(device)
            dones_tensor = dones_tensor.to(device)
        
        logger.debug(f"Sampled batch of size {batch_size}")
        
        return states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor
    
    def __len__(self) -> int:
        """Return current size of the buffer."""
        return len(self.buffer)
    
    def is_ready(self, min_size: int) -> bool:
        """
        Check if buffer has enough experiences.
        
        Args:
            min_size: Minimum number of experiences needed
        
        Returns:
            True if buffer is ready
        """
        return len(self.buffer) >= min_size
    
    def clear(self) -> None:
        """Clear the buffer."""
        self.buffer.clear()
        logger.debug("ReplayBuffer cleared")
    
    def get_statistics(self) -> dict:
        """Get buffer statistics."""
        if len(self.buffer) == 0:
            return {
                "size": 0,
                "total_added": self.total_added,
                "avg_reward": 0.0,
                "avg_done_rate": 0.0,
            }
        
        rewards = [e.reward for e in self.buffer]
        dones = [e.done for e in self.buffer]
        
        return {
            "size": len(self.buffer),
            "total_added": self.total_added,
            "avg_reward": np.mean(rewards),
            "avg_done_rate": np.mean(dones),
        }


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer.
    
    Samples experiences with probability proportional to their TD error,
    which can help with more efficient learning.
    """
    
    def __init__(
        self,
        capacity: int = 10000,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
    ):
        """
        Initialize the prioritized replay buffer.
        
        Args:
            capacity: Maximum number of experiences
            alpha: Priority exponent (0 = uniform, 1 = full prioritization)
            beta: Importance sampling exponent (0 = no correction, 1 = full correction)
            beta_increment: How much to increase beta per sample
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        
        self.buffer: List[Experience] = []
        self.priorities: np.ndarray = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
        
        # Statistics
        self.total_added = 0
        
        logger.info(
            f"PrioritizedReplayBuffer initialized: capacity={capacity}, "
            f"alpha={alpha}, beta={beta}"
        )
    
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Add an experience with maximum priority."""
        experience = Experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
        )
        
        # Add experience
        if self.size < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        # Set maximum priority for new experience
        self.priorities[self.position] = (
            self.priorities[: self.size].max() if self.size > 0 else 1.0
        )
        
        # Update position and size
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.total_added += 1
        
        logger.debug(
            f"Experience added with max priority: state={state}, "
            f"action={action}, reward={reward:.4f}"
        )
    
    def sample(
        self, batch_size: int, device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
        """
        Sample a mini-batch with prioritization.
        
        Args:
            batch_size: Number of experiences to sample
            device: Device to place tensors on
        
        Returns:
            Tuple of (states, actions, rewards, next_states, dones, weights)
        """
        if self.size < batch_size:
            raise ValueError(
                f"Buffer has {self.size} experiences, "
                f"but batch_size is {batch_size}"
            )
        
        # Calculate sampling probabilities
        priorities = self.priorities[: self.size]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # Sample indices
        indices = np.random.choice(self.size, batch_size, p=probabilities)
        
        # Calculate importance sampling weights
        weights = (self.size * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        
        # Get experiences
        experiences = [self.buffer[i] for i in indices]
        
        # Unzip experiences
        states = np.array([e.state for e in experiences])
        actions = np.array([e.action for e in experiences])
        rewards = np.array([e.reward for e in experiences])
        next_states = np.array([e.next_state for e in experiences])
        dones = np.array([e.done for e in experiences], dtype=np.float32)
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(states)
        actions_tensor = torch.LongTensor(actions)
        rewards_tensor = torch.FloatTensor(rewards)
        next_states_tensor = torch.FloatTensor(next_states)
        dones_tensor = torch.FloatTensor(dones)
        weights_tensor = torch.FloatTensor(weights)
        
        # Move to device if specified
        if device is not None:
            states_tensor = states_tensor.to(device)
            actions_tensor = actions_tensor.to(device)
            rewards_tensor = rewards_tensor.to(device)
            next_states_tensor = next_states_tensor.to(device)
            dones_tensor = dones_tensor.to(device)
            weights_tensor = weights_tensor.to(device)
        
        # Update beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        logger.debug(f"Sampled batch of size {batch_size} with priorities")
        
        return states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor, weights_tensor
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """
        Update priorities for sampled experiences.
        
        Args:
            indices: Experience indices
            priorities: New priorities
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + 1e-6  # Add small epsilon to avoid zero
    
    def __len__(self) -> int:
        """Return current size of the buffer."""
        return self.size
    
    def is_ready(self, min_size: int) -> bool:
        """Check if buffer has enough experiences."""
        return self.size >= min_size
    
    def clear(self) -> None:
        """Clear the buffer."""
        self.buffer.clear()
        self.priorities = np.zeros(self.capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
        logger.debug("PrioritizedReplayBuffer cleared")
    
    def get_statistics(self) -> dict:
        """Get buffer statistics."""
        if self.size == 0:
            return {
                "size": 0,
                "total_added": self.total_added,
                "avg_reward": 0.0,
                "avg_done_rate": 0.0,
                "avg_priority": 0.0,
            }
        
        rewards = [e.reward for e in self.buffer[: self.size]]
        dones = [e.done for e in self.buffer[: self.size]]
        priorities = self.priorities[: self.size]
        
        return {
            "size": self.size,
            "total_added": self.total_added,
            "avg_reward": np.mean(rewards),
            "avg_done_rate": np.mean(dones),
            "avg_priority": np.mean(priorities),
        }