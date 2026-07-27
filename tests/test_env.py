"""
Tests for GridWorld Environment
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from curious_agent.env.grid_world import ActionType, GridWorld, Zone, ZoneType


class TestGridWorld:
    """Test cases for GridWorld environment."""
    
    def test_initialization(self):
        """Test environment initialization."""
        env = GridWorld(grid_size=5)
        
        assert env.grid_size == 5
        assert env.num_states == 25
        assert env.num_actions == 4
        assert len(env.grid) == 5
        assert len(env.grid[0]) == 5
    
    def test_reset(self):
        """Test environment reset."""
        env = GridWorld(grid_size=5)
        
        state = env.reset()
        
        assert isinstance(state, np.ndarray)
        assert state.shape == (2,)
        assert state[0] == 0 and state[1] == 0  # Agent starts at (0, 0)
        assert env.step_count == 0
    
    def test_step(self):
        """Test environment step."""
        env = GridWorld(grid_size=5)
        env.reset()
        
        # Take an action
        next_state, reward, done, info = env.step(ActionType.RIGHT.value)
        
        assert isinstance(next_state, np.ndarray)
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        assert env.step_count == 1
    
    def test_state_index(self):
        """Test state to index conversion."""
        env = GridWorld(grid_size=5)
        
        state_idx = env.get_state_idx(np.array([2, 3]))
        assert state_idx == 2 * 5 + 3  # 13
        
        state_idx = env.get_state_idx(np.array([0, 0]))
        assert state_idx == 0
    
    def test_zone_types(self):
        """Test that all zone types are present."""
        env = GridWorld(grid_size=10)
        
        zone_types = set()
        for i in range(env.grid_size):
            for j in range(env.grid_size):
                zone_types.add(env.grid[i][j].zone_type)
        
        # Should have at least some zone types
        assert len(zone_types) > 0
    
    def test_agent_movement(self):
        """Test agent movement within bounds."""
        env = GridWorld(grid_size=5)
        env.reset()
        
        # Try to move up from (0, 0) - should stay at (0, 0)
        next_state, _, _, _ = env.step(ActionType.UP.value)
        assert next_state[0] == 0
        
        # Try to move left from (0, 0) - should stay at (0, 0)
        next_state, _, _, _ = env.step(ActionType.LEFT.value)
        assert next_state[1] == 0

    def test_static_zone_preserves_selected_action(self):
        """A static zone adds no motion but does not suppress the action."""
        env = GridWorld(grid_size=5)
        zone = Zone(zone_type=ZoneType.STATIC, color=(128, 128, 128))

        assert env._get_next_position(2, 2, ActionType.LEFT.value, zone) == (2, 1)
        assert env._get_next_position(2, 2, ActionType.RIGHT.value, zone) == (2, 3)

    def test_deterministic_zone_modifies_selected_action(self):
        """A deterministic shift is applied after the intended movement."""
        env = GridWorld(grid_size=5)
        zone = Zone(
            zone_type=ZoneType.DETERMINISTIC_A,
            color=(0, 255, 0),
            rule="shift_up",
        )

        assert env._get_next_position(2, 2, ActionType.LEFT.value, zone) == (1, 1)
        assert env._get_next_position(2, 2, ActionType.RIGHT.value, zone) == (1, 3)

    def test_noisy_zone_perturbs_instead_of_replacing_action(self, monkeypatch):
        """Noise is local to the intended destination, not a random teleport."""
        env = GridWorld(grid_size=5)
        zone = Zone(zone_type=ZoneType.NOISY, color=(255, 0, 0))
        monkeypatch.setattr(np.random, "randint", lambda *_: ActionType.UP.value)

        assert env._get_next_position(2, 2, ActionType.LEFT.value, zone) == (1, 1)
        assert env._get_next_position(2, 2, ActionType.RIGHT.value, zone) == (1, 3)
    
    def test_goal_reward(self):
        """Test goal reward."""
        config = {
            "environment": {
                "goal": {"position": [4, 4], "reward": 10.0}
            }
        }
        env = GridWorld(grid_size=5, config=config)
        
        # Manually set agent position to goal
        env.agent_pos = (4, 4)
        
        # Calculate reward
        reward = env._calculate_reward((4, 4))
        assert reward == 10.0
    
    def test_zone_stats(self):
        """Test zone statistics tracking."""
        env = GridWorld(grid_size=5)
        env.reset()
        
        # Take some steps
        for _ in range(10):
            env.step(np.random.randint(0, 4))
        
        stats = env.get_zone_stats()
        assert isinstance(stats, dict)
        assert sum(stats.values()) == pytest.approx(1.0, abs=1e-6)


class TestZone:
    """Test cases for Zone class."""
    
    def test_static_zone(self):
        """Test static zone behavior."""
        zone = Zone(
            zone_type=ZoneType.STATIC,
            color=(128, 128, 128),
        )
        
        assert zone.get_rule() == "stay"
    
    def test_noisy_zone(self):
        """Test noisy zone behavior."""
        zone = Zone(
            zone_type=ZoneType.NOISY,
            color=(255, 0, 0),
        )
        
        assert zone.get_rule() == "random"
    
    def test_dynamic_zone_switching(self):
        """Test dynamic zone rule switching."""
        zone = Zone(
            zone_type=ZoneType.DYNAMIC,
            color=(255, 255, 0),
            switch_interval=5,
            rules=["rule_a", "rule_b"],
        )
        
        # Initial rule
        assert zone.get_rule() == "rule_a"
        
        # Step until switch
        for _ in range(5):
            zone.step()
        
        assert zone.get_rule() == "rule_b"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
