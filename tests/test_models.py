"""
Tests for Neural Network Models
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from curious_agent.models.world_model import WorldModel
from curious_agent.models.confidence_net import ConfidenceNetwork
from curious_agent.models.q_network import QNetwork, DuelingQNetwork


class TestWorldModel:
    """Test cases for WorldModel."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = WorldModel(
            state_dim=2,
            num_actions=4,
            hidden_dims=[64, 64],
            learning_rate=0.001,
        )
        
        assert model.state_dim == 2
        assert model.num_actions == 4
    
    def test_forward_pass(self):
        """Test forward pass."""
        model = WorldModel(state_dim=2, num_actions=4)
        
        state = torch.randn(1, 2)
        action = torch.zeros(1, 4)
        action[0, 0] = 1.0
        
        output = model(state, action)
        
        assert output.shape == (1, 2)
    
    def test_predict(self):
        """Test single prediction."""
        model = WorldModel(state_dim=2, num_actions=4)
        
        state = torch.randn(2)
        action = 0
        
        predicted = model.predict(state, action)
        
        assert predicted.shape == (2,)
    
    def test_update(self):
        """Test model update."""
        model = WorldModel(state_dim=2, num_actions=4, learning_rate=0.01)
        
        state = torch.randn(4, 2)
        action = torch.zeros(4, 4)
        action[0, 0] = 1.0
        action[1, 1] = 1.0
        action[2, 2] = 1.0
        action[3, 3] = 1.0
        
        next_state = torch.randn(4, 2)
        
        loss = model.update(state, action, next_state)
        
        assert isinstance(loss, float)
        assert loss >= 0
    
    def test_compute_error(self):
        """Test error computation."""
        model = WorldModel(state_dim=2, num_actions=4)
        
        state = torch.randn(1, 2)
        action = torch.zeros(1, 4)
        action[0, 0] = 1.0
        next_state = torch.randn(1, 2)
        
        error = model.compute_error(state, action, next_state)
        
        assert isinstance(error, float)
        assert error >= 0


class TestConfidenceNetwork:
    """Test cases for ConfidenceNetwork."""
    
    def test_initialization(self):
        """Test network initialization."""
        net = ConfidenceNetwork(
            state_dim=2,
            num_actions=4,
            hidden_dims=[32, 32],
            learning_rate=0.0005,
        )
        
        assert net.state_dim == 2
        assert net.num_actions == 4
    
    def test_forward_pass(self):
        """Test forward pass."""
        net = ConfidenceNetwork(state_dim=2, num_actions=4)
        
        state = torch.randn(1, 2)
        action = torch.zeros(1, 4)
        action[0, 0] = 1.0
        
        output = net(state, action)
        
        assert output.shape == (1, 1)
        assert 0 <= output.item() <= 1  # Sigmoid output
    
    def test_predict(self):
        """Test single prediction."""
        net = ConfidenceNetwork(state_dim=2, num_actions=4)
        
        state = torch.randn(2)
        action = 0
        
        predicted = net.predict(state, action)
        
        assert isinstance(predicted, float)
        assert 0 <= predicted <= 1
    
    def test_update(self):
        """Test network update."""
        net = ConfidenceNetwork(state_dim=2, num_actions=4, learning_rate=0.01)
        
        state = torch.randn(4, 2)
        action = torch.zeros(4, 4)
        action[0, 0] = 1.0
        action[1, 1] = 1.0
        action[2, 2] = 1.0
        action[3, 3] = 1.0
        
        actual_error = torch.tensor([[0.5], [0.8], [0.2], [0.9]])
        
        loss = net.update(state, action, actual_error)
        
        assert isinstance(loss, float)
        assert loss >= 0


class TestQNetwork:
    """Test cases for QNetwork."""
    
    def test_initialization(self):
        """Test network initialization."""
        net = QNetwork(
            state_dim=2,
            num_actions=4,
            hidden_dims=[128, 128],
            learning_rate=0.001,
        )
        
        assert net.state_dim == 2
        assert net.num_actions == 4
    
    def test_forward_pass(self):
        """Test forward pass."""
        net = QNetwork(state_dim=2, num_actions=4)
        
        state = torch.randn(1, 2)
        
        output = net(state)
        
        assert output.shape == (1, 4)
    
    def test_get_action(self):
        """Test action selection."""
        net = QNetwork(state_dim=2, num_actions=4)
        
        state = torch.randn(2)
        
        action = net.get_action(state, epsilon=0.0)
        
        assert 0 <= action < 4
    
    def test_update(self):
        """Test network update."""
        net = QNetwork(state_dim=2, num_actions=4, learning_rate=0.001)
        
        batch_size = 8
        states = torch.randn(batch_size, 2)
        actions = torch.randint(0, 4, (batch_size,))
        rewards = torch.randn(batch_size)
        next_states = torch.randn(batch_size, 2)
        dones = torch.zeros(batch_size)
        
        loss = net.update(states, actions, rewards, next_states, dones)
        
        assert isinstance(loss, float)
        assert loss >= 0


class TestDuelingQNetwork:
    """Test cases for DuelingQNetwork."""
    
    def test_initialization(self):
        """Test network initialization."""
        net = DuelingQNetwork(
            state_dim=2,
            num_actions=4,
            hidden_dims=[128, 128],
            value_hidden_dim=64,
            advantage_hidden_dim=64,
            learning_rate=0.001,
        )
        
        assert net.state_dim == 2
        assert net.num_actions == 4
    
    def test_forward_pass(self):
        """Test forward pass."""
        net = DuelingQNetwork(state_dim=2, num_actions=4)
        
        state = torch.randn(1, 2)
        
        output = net(state)
        
        assert output.shape == (1, 4)
    
    def test_dueling_architecture(self):
        """Test that dueling architecture produces valid Q-values."""
        net = DuelingQNetwork(state_dim=2, num_actions=4)
        
        state = torch.randn(10, 2)
        
        q_values = net(state)
        
        # Q-values should be different for different actions
        assert q_values.shape == (10, 4)
        
        # Check that value and advantage streams are separate
        assert hasattr(net, 'value_stream')
        assert hasattr(net, 'advantage_stream')


class TestModelIntegration:
    """Integration tests for models working together."""
    
    def test_world_model_to_confidence(self):
        """Test passing world model error to confidence network."""
        wm = WorldModel(state_dim=2, num_actions=4)
        cn = ConfidenceNetwork(state_dim=2, num_actions=4)
        
        state = torch.randn(2)
        action = 0
        next_state = torch.randn(2)
        
        # Get prediction from world model
        predicted_next = wm.predict(state, action)
        
        # Compute error
        error = torch.nn.MSELoss()(predicted_next, next_state)
        
        # Get confidence prediction
        confidence = cn.predict(state, action)
        
        assert isinstance(error.item(), float)
        assert isinstance(confidence, float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])