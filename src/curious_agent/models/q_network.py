"""
Q-Network for Action-Value Estimation

Evaluates state-action pairs using either standard MLP or Dueling DQN architecture.
The Q-network learns to maximize cumulative reward (external + curiosity).
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger(__name__)


class QNetwork(nn.Module):
    """
    Standard Q-Network using MLP.
    
    Architecture:
    - Input: state (2D) + action one-hot (4D) = 6D
    - Hidden: MLP with configurable layers
    - Output: Q-values for all actions
    """
    
    def __init__(
        self,
        state_dim: int = 2,
        num_actions: int = 4,
        hidden_dims: Optional[List[int]] = None,
        learning_rate: float = 0.001,
    ):
        """
        Initialize the Q-network.
        
        Args:
            state_dim: Dimension of state space
            num_actions: Number of discrete actions
            hidden_dims: List of hidden layer dimensions
            learning_rate: Learning rate for optimizer
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.num_actions = num_actions
        
        # Default hidden dimensions
        if hidden_dims is None:
            hidden_dims = [128, 128]
        
        # Build network
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_actions))
        
        self.network = nn.Sequential(*layers)
        
        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        # Statistics
        self.training_losses: List[float] = []
        
        logger.info(
            f"QNetwork initialized: input={state_dim}, "
            f"hidden={hidden_dims}, output={num_actions}"
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            state: State tensor [batch_size, state_dim]
        
        Returns:
            Q-values [batch_size, num_actions]
        """
        return self.network(state)
    
    def get_action(
        self,
        state: torch.Tensor,
        epsilon: float = 0.0,
    ) -> int:
        """
        Get action using epsilon-greedy policy.
        
        Args:
            state: State tensor [state_dim]
            epsilon: Exploration rate
        
        Returns:
            Action index
        """
        if torch.rand(1).item() < epsilon:
            return torch.randint(0, self.num_actions, (1,)).item()
        
        self.eval()
        with torch.no_grad():
            if state.dim() == 1:
                state = state.unsqueeze(0)
            
            q_values = self.forward(state)
            return q_values.argmax(dim=-1).item()
    
    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        gamma: float = 0.99,
    ) -> float:
        """
        Update Q-network using DQN loss.
        
        Args:
            states: Current states [batch_size, state_dim]
            actions: Actions taken [batch_size]
            rewards: Rewards received [batch_size]
            next_states: Next states [batch_size, state_dim]
            dones: Done flags [batch_size]
            gamma: Discount factor
        
        Returns:
            Loss value
        """
        self.train()
        
        # Get current Q-values
        current_q = self.forward(states)
        current_q = current_q.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Get target Q-values
        with torch.no_grad():
            next_q = self.forward(next_states)
            next_q = next_q.max(dim=1)[0]
            target_q = rewards + gamma * next_q * (1 - dones)
        
        # Compute loss
        loss = nn.MSELoss()(current_q, target_q)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Record loss
        loss_value = loss.item()
        self.training_losses.append(loss_value)
        
        logger.debug(f"QNetwork update: loss={loss_value:.6f}")
        
        return loss_value
    
    def get_device(self) -> torch.device:
        """Get the device of the model."""
        return next(self.parameters()).device
    
    def save(self, path: str) -> None:
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_losses': self.training_losses,
        }, path)
        logger.info(f"QNetwork saved to {path}")
    
    def load(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.get_device())
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_losses = checkpoint.get('training_losses', [])
        logger.info(f"QNetwork loaded from {path}")
    
    def __repr__(self) -> str:
        return (
            f"QNetwork("
            f"state_dim={self.state_dim}, "
            f"num_actions={self.num_actions}, "
            f"params={sum(p.numel() for p in self.parameters())})"
        )


class DuelingQNetwork(nn.Module):
    """
    Dueling DQN architecture.
    
    Separates state value V(s) and advantage A(s,a):
    Q(s,a) = V(s) + A(s,a) - mean(A(s,a'))
    
    This can help with learning stable values.
    """
    
    def __init__(
        self,
        state_dim: int = 2,
        num_actions: int = 4,
        hidden_dims: Optional[List[int]] = None,
        value_hidden_dim: int = 64,
        advantage_hidden_dim: int = 64,
        learning_rate: float = 0.001,
    ):
        """
        Initialize the dueling Q-network.
        
        Args:
            state_dim: Dimension of state space
            num_actions: Number of discrete actions
            hidden_dims: List of hidden layer dimensions (for feature extraction)
            value_hidden_dim: Hidden dimension for value stream
            advantage_hidden_dim: Hidden dimension for advantage stream
            learning_rate: Learning rate for optimizer
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.num_actions = num_actions
        
        # Default hidden dimensions
        if hidden_dims is None:
            hidden_dims = [128, 128]
        
        # Feature extraction layers
        feature_layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            feature_layers.append(nn.Linear(prev_dim, hidden_dim))
            feature_layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        self.feature_network = nn.Sequential(*feature_layers)
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(prev_dim, value_hidden_dim),
            nn.ReLU(),
            nn.Linear(value_hidden_dim, 1),
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(prev_dim, advantage_hidden_dim),
            nn.ReLU(),
            nn.Linear(advantage_hidden_dim, num_actions),
        )
        
        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        # Statistics
        self.training_losses: List[float] = []
        
        logger.info(
            f"DuelingQNetwork initialized: input={state_dim}, "
            f"hidden={hidden_dims}, output={num_actions}"
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            state: State tensor [batch_size, state_dim]
        
        Returns:
            Q-values [batch_size, num_actions]
        """
        # Feature extraction
        features = self.feature_network(state)
        
        # Value stream
        value = self.value_stream(features)
        
        # Advantage stream
        advantage = self.advantage_stream(features)
        
        # Combine: Q = V + A - mean(A)
        q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)
        
        return q_values
    
    def get_action(
        self,
        state: torch.Tensor,
        epsilon: float = 0.0,
    ) -> int:
        """
        Get action using epsilon-greedy policy.
        
        Args:
            state: State tensor [state_dim]
            epsilon: Exploration rate
        
        Returns:
            Action index
        """
        if torch.rand(1).item() < epsilon:
            return torch.randint(0, self.num_actions, (1,)).item()
        
        self.eval()
        with torch.no_grad():
            if state.dim() == 1:
                state = state.unsqueeze(0)
            
            q_values = self.forward(state)
            return q_values.argmax(dim=-1).item()
    
    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        gamma: float = 0.99,
    ) -> float:
        """
        Update Q-network using DQN loss.
        
        Args:
            states: Current states [batch_size, state_dim]
            actions: Actions taken [batch_size]
            rewards: Rewards received [batch_size]
            next_states: Next states [batch_size, state_dim]
            dones: Done flags [batch_size]
            gamma: Discount factor
        
        Returns:
            Loss value
        """
        self.train()
        
        # Get current Q-values
        current_q = self.forward(states)
        current_q = current_q.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Get target Q-values
        with torch.no_grad():
            next_q = self.forward(next_states)
            next_q = next_q.max(dim=1)[0]
            target_q = rewards + gamma * next_q * (1 - dones)
        
        # Compute loss
        loss = nn.MSELoss()(current_q, target_q)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Record loss
        loss_value = loss.item()
        self.training_losses.append(loss_value)
        
        logger.debug(f"DuelingQNetwork update: loss={loss_value:.6f}")
        
        return loss_value
    
    def get_device(self) -> torch.device:
        """Get the device of the model."""
        return next(self.parameters()).device
    
    def save(self, path: str) -> None:
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_losses': self.training_losses,
        }, path)
        logger.info(f"DuelingQNetwork saved to {path}")
    
    def load(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.get_device())
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_losses = checkpoint.get('training_losses', [])
        logger.info(f"DuelingQNetwork loaded from {path}")
    
    def __repr__(self) -> str:
        return (
            f"DuelingQNetwork("
            f"state_dim={self.state_dim}, "
            f"num_actions={self.num_actions}, "
            f"params={sum(p.numel() for p in self.parameters())})"
        )