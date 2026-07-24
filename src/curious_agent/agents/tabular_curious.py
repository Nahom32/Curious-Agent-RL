"""
Tabular Curious Agent - Phase 1 Implementation

This module implements Schmidhuber's curious agent using lookup tables
for the world model (M), confidence network (C), and Q-function (Q).

The key insight is that curiosity reward = improvement in prediction ability:
r_curiosity = o_C_before - o_C_after

This means the agent is attracted to "zone of proximal learning" - not too easy,
not too hard.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class TabularCuriousAgent:
    """
    Tabular implementation of Schmidhuber's curious agent.
    
    Uses lookup tables for:
    - M[state][action] → predicted_next_state
    - C[state][action] → predicted_error
    - Q[state][action] → value
    """
    
    def __init__(
        self,
        num_states: int,
        num_actions: int,
        config: Optional[Dict] = None,
    ):
        """
        Initialize the tabular curious agent.
        
        Args:
            num_states: Number of discrete states
            num_actions: Number of discrete actions
            config: Configuration dictionary
        """
        self.num_states = num_states
        self.num_actions = num_actions
        self.config = config or {}
        
        # Hyperparameters
        agent_config = self.config.get("agent", {})
        self.alpha_m = agent_config.get("alpha_m", 0.2)  # Model learning rate
        self.alpha_c = agent_config.get("alpha_c", 0.05)  # Confidence learning rate
        self.alpha_q = agent_config.get("alpha_q", 0.1)  # Q-learning rate
        self.beta = agent_config.get("beta", 1.0)  # Curiosity weight
        self.gamma = agent_config.get("gamma", 0.99)  # Discount factor
        self.epsilon = agent_config.get("epsilon", 1.0)  # Exploration rate
        self.epsilon_min = agent_config.get("epsilon_min", 0.01)
        self.epsilon_decay = agent_config.get("epsilon_decay", 0.9995)
        
        # Lookup tables
        # M[state][action] = predicted_next_state (as state index)
        self.M = np.zeros((num_states, num_actions), dtype=np.int32)
        
        # C[state][action] = predicted_error (MSE)
        self.C = np.zeros((num_states, num_actions), dtype=np.float32)
        
        # Q[state][action] = value
        self.Q = np.zeros((num_states, num_actions), dtype=np.float32)
        
        # Statistics
        self.curiosity_rewards: List[float] = []
        self.external_rewards: List[float] = []
        self.model_errors: List[float] = []
        
        logger.info(
            f"TabularCuriousAgent initialized: "
            f"states={num_states}, actions={num_actions}"
        )
    
    def select_action(self, state_idx: int) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state_idx: Current state index
        
        Returns:
            Selected action index
        """
        if np.random.random() < self.epsilon:
            # Random exploration
            action = np.random.randint(0, self.num_actions)
            logger.debug(f"Random action: {action}")
        else:
            # Greedy action selection
            action = np.argmax(self.Q[state_idx])
            logger.debug(f"Greedy action: {action}, Q-values: {self.Q[state_idx]}")
        
        return action
    
    def get_confidence_before(
        self, state_idx: int, action: int
    ) -> float:
        """
        Get C's predicted error BEFORE taking action.
        
        Args:
            state_idx: Current state index
            action: Action to take
        
        Returns:
            Predicted error
        """
        return float(self.C[state_idx, action])
    
    def update_model(
        self,
        state_idx: int,
        action: int,
        next_state_idx: int,
    ) -> float:
        """
        Update world model M.
        
        For tabular model, we simply store the observed transition.
        
        Args:
            state_idx: Current state index
            action: Action taken
            next_state_idx: Actual next state index
        
        Returns:
            Model error (0 for perfect prediction, 1 for wrong)
        """
        # Tabular model: just store the transition
        predicted_next = self.M[state_idx, action]
        
        # Calculate error (1 if wrong, 0 if correct)
        error = 1.0 if predicted_next != next_state_idx else 0.0
        
        # Update model (tabular: just store the correct transition)
        self.M[state_idx, action] = next_state_idx
        
        logger.debug(
            f"Model update: state={state_idx}, action={action}, "
            f"predicted={predicted_next}, actual={next_state_idx}, error={error}"
        )
        
        return error
    
    def update_confidence(
        self,
        state_idx: int,
        action: int,
        actual_error: float,
    ) -> float:
        """
        Update confidence network C.
        
        C learns to predict M's error for a given state-action pair.
        
        Args:
            state_idx: Current state index
            action: Action taken
            actual_error: Actual error from model update
        
        Returns:
            C's new prediction
        """
        # C's current prediction
        current_pred = self.C[state_idx, action]
        
        # Update C towards actual error
        # C[state][action] += alpha_c * (actual_error - C[state][action])
        self.C[state_idx, action] += self.alpha_c * (actual_error - current_pred)
        
        # Clamp to [0, 1] range
        self.C[state_idx, action] = np.clip(self.C[state_idx, action], 0.0, 1.0)
        
        new_pred = self.C[state_idx, action]
        
        logger.debug(
            f"Confidence update: state={state_idx}, action={action}, "
            f"old_pred={current_pred:.4f}, actual_error={actual_error}, "
            f"new_pred={new_pred:.4f}"
        )
        
        return new_pred
    
    def compute_curiosity_reward(
        self,
        o_c_before: float,
        o_c_after: float,
    ) -> float:
        """
        Compute curiosity reward.
        
        r_curiosity = o_C_before - o_C_after
        
        Positive reward when C's error prediction drops (M improved).
        Negative reward when C's error prediction increases (M got worse).
        
        Args:
            o_c_before: C's prediction before model update
            o_c_after: C's prediction after model update
        
        Returns:
            Curiosity reward
        """
        return o_c_before - o_c_after
    
    def update_q(
        self,
        state_idx: int,
        action: int,
        reward: float,
        next_state_idx: int,
        done: bool,
    ) -> float:
        """
        Update Q-function using Watkins' Q-learning.
        
        Q[s][a] += alpha_q * (r + gamma * max(Q[s']) - Q[s][a])
        
        Args:
            state_idx: Current state index
            action: Action taken
            reward: Total reward (external + curiosity)
            next_state_idx: Next state index
            done: Whether episode is done
        
        Returns:
            TD error
        """
        # Current Q-value
        current_q = self.Q[state_idx, action]
        
        # Best next Q-value
        if done:
            best_next_q = 0.0
        else:
            best_next_q = np.max(self.Q[next_state_idx])
        
        # TD target
        td_target = reward + self.gamma * best_next_q
        
        # TD error
        td_error = td_target - current_q
        
        # Update Q
        self.Q[state_idx, action] += self.alpha_q * td_error
        
        logger.debug(
            f"Q update: state={state_idx}, action={action}, "
            f"reward={reward:.4f}, td_error={td_error:.4f}"
        )
        
        return td_error
    
    def decay_epsilon(self) -> None:
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        logger.debug(f"Epsilon decayed to: {self.epsilon:.4f}")
    
    def get_action_probs(self, state_idx: int) -> np.ndarray:
        """
        Get action probabilities for a state (for analysis).
        
        Args:
            state_idx: State index
        
        Returns:
            Array of action probabilities
        """
        q_values = self.Q[state_idx]
        
        # Softmax for probabilities
        exp_q = np.exp(q_values - np.max(q_values))
        probs = exp_q / np.sum(exp_q)
        
        # Add exploration
        probs = (1 - self.epsilon) * probs + self.epsilon / self.num_actions
        
        return probs
    
    def get_model_accuracy(self) -> float:
        """Calculate overall model accuracy."""
        # This is a placeholder - in practice, we'd track accuracy over time
        return 0.0
    
    def get_statistics(self) -> Dict:
        """Get agent statistics."""
        return {
            "epsilon": self.epsilon,
            "avg_curiosity_reward": (
                np.mean(self.curiosity_rewards) if self.curiosity_rewards else 0.0
            ),
            "avg_external_reward": (
                np.mean(self.external_rewards) if self.external_rewards else 0.0
            ),
            "avg_model_error": (
                np.mean(self.model_errors) if self.model_errors else 0.0
            ),
        }
    
    def reset_statistics(self) -> None:
        """Reset episode statistics."""
        self.curiosity_rewards = []
        self.external_rewards = []
        self.model_errors = []
    
    def __repr__(self) -> str:
        return (
            f"TabularCuriousAgent("
            f"states={self.num_states}, "
            f"actions={self.num_actions}, "
            f"epsilon={self.epsilon:.3f})"
        )