"""
Tests for Curious Agents
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from curious_agent.agents.tabular_curious import TabularCuriousAgent


class TestTabularCuriousAgent:
    """Test cases for TabularCuriousAgent."""
    
    def test_initialization(self):
        """Test agent initialization."""
        agent = TabularCuriousAgent(
            num_states=100,
            num_actions=4,
            config={"agent": {"alpha_m": 0.2, "alpha_c": 0.05}},
        )
        
        assert agent.num_states == 100
        assert agent.num_actions == 4
        assert agent.M.shape == (100, 4)
        assert agent.C.shape == (100, 4)
        assert agent.Q.shape == (100, 4)
    
    def test_action_selection(self):
        """Test action selection."""
        agent = TabularCuriousAgent(
            num_states=100,
            num_actions=4,
            config={"agent": {"epsilon": 0.0}},
        )
        
        action = agent.select_action(0)
        assert 0 <= action < 4
    
    def test_confidence_before(self):
        """Test getting confidence prediction."""
        agent = TabularCuriousAgent(
            num_states=100,
            num_actions=4,
        )
        
        pred = agent.get_confidence_before(0, 0)
        assert isinstance(pred, float)
        assert 0 <= pred <= 1
    
    def test_model_update(self):
        """Test model update."""
        agent = TabularCuriousAgent(
            num_states=100,
            num_actions=4,
        )
        
        # Initial prediction is 0
        assert agent.M[0, 0] == 0
        
        # Update model
        error = agent.update_model(0, 0, 5)
        
        # Should have error (predicted 0, actual 5)
        assert error == 1.0
        # Model should be updated
        assert agent.M[0, 0] == 5
        
        # Update again with same transition
        error = agent.update_model(0, 0, 5)
        
        # Should have no error (predicted 5, actual 5)
        assert error == 0.0
    
    def test_confidence_update(self):
        """Test confidence update."""
        agent = TabularCuriousAgent(
            num_states=100,
            num_actions=4,
            config={"agent": {"alpha_c": 0.1}},
        )
        
        # Initial prediction
        initial_pred = agent.C[0, 0]
        
        # Update with error
        new_pred = agent.update_confidence(0, 0, 1.0)
        
        # Should move towards actual error
        assert new_pred != initial_pred
        assert 0 <= new_pred <= 1
    
    def test_curiosity_reward(self):
        """Test curiosity reward calculation."""
        agent = TabularCuriousAgent(
            num_states=100,
            num_actions=4,
        )
        
        # Positive reward when prediction improves
        r = agent.compute_curiosity_reward(0.5, 0.3)
        assert r == 0.2
        
        # Negative reward when prediction worsens
        r = agent.compute_curiosity_reward(0.3, 0.5)
        assert r == -0.2
        
        # Zero reward when no change
        r = agent.compute_curiosity_reward(0.5, 0.5)
        assert r == 0.0
    
    def test_q_update(self):
        """Test Q-learning update."""
        agent = TabularCuriousAgent(
            num_states=100,
            num_actions=4,
            config={"agent": {"alpha_q": 0.1, "gamma": 0.99}},
        )
        
        # Initial Q-value
        initial_q = agent.Q[0, 0]
        
        # Update Q
        td_error = agent.update_q(0, 0, 1.0, 5, done=False)
        
        # Q-value should change
        assert agent.Q[0, 0] != initial_q
    
    def test_epsilon_decay(self):
        """Test epsilon decay."""
        agent = TabularCuriousAgent(
            num_states=100,
            num_actions=4,
            config={"agent": {"epsilon": 1.0, "epsilon_min": 0.01, "epsilon_decay": 0.99}},
        )
        
        initial_epsilon = agent.epsilon
        agent.decay_epsilon()
        
        assert agent.epsilon < initial_epsilon
        assert agent.epsilon >= 0.01
    
    def test_action_probs(self):
        """Test action probability calculation."""
        agent = TabularCuriousAgent(
            num_states=100,
            num_actions=4,
            config={"agent": {"epsilon": 0.1}},
        )
        
        probs = agent.get_action_probs(0)
        
        assert len(probs) == 4
        assert abs(sum(probs) - 1.0) < 1e-6
    
    def test_statistics(self):
        """Test statistics collection."""
        agent = TabularCuriousAgent(
            num_states=100,
            num_actions=4,
        )
        
        stats = agent.get_statistics()
        
        assert "epsilon" in stats
        assert "avg_curiosity_reward" in stats
        assert "avg_external_reward" in stats


class TestCuriosityMechanism:
    """Test the core curiosity mechanism."""
    
    def test_curiosity_drives_exploration(self):
        """Test that curiosity reward encourages exploration."""
        agent = TabularCuriousAgent(
            num_states=100,
            num_actions=4,
            config={"agent": {"beta": 1.0}},
        )
        
        # Simulate learning a new transition
        state_idx = 0
        action = 0
        next_state_idx = 5
        
        # The first observation reveals that the model is wrong.
        initial_error = agent.update_model(state_idx, action, next_state_idx)
        agent.update_confidence(state_idx, action, initial_error)

        # On the next observation, the learned transition is predicted correctly.
        o_c_before = agent.get_confidence_before(state_idx, action)
        improved_error = agent.update_model(state_idx, action, next_state_idx)
        agent.update_confidence(state_idx, action, improved_error)
        o_c_after = agent.get_confidence_before(state_idx, action)
        
        # Compute curiosity reward
        r_curiosity = agent.compute_curiosity_reward(o_c_before, o_c_after)
        
        # Should be positive (improvement)
        assert r_curiosity > 0
    
    def test_boredom_in_known_zones(self):
        """Test that agent gets bored in known zones."""
        agent = TabularCuriousAgent(
            num_states=100,
            num_actions=4,
            config={"agent": {"beta": 1.0}},
        )
        
        state_idx = 0
        action = 0
        next_state_idx = 5
        
        # Learn the transition perfectly
        agent.update_model(state_idx, action, next_state_idx)
        agent.update_confidence(state_idx, action, 0.0)  # No error
        
        # Get confidence before (should be low)
        o_c_before = agent.get_confidence_before(state_idx, action)
        
        # Update again with same transition
        agent.update_model(state_idx, action, next_state_idx)
        agent.update_confidence(state_idx, action, 0.0)
        o_c_after = agent.get_confidence_before(state_idx, action)
        
        # Curiosity reward should be small or zero
        r_curiosity = agent.compute_curiosity_reward(o_c_before, o_c_after)
        assert r_curiosity <= 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
