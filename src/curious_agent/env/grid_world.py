"""
Grid World Environment for Schmidhuber's Curious Agent

A 10x10 grid with 5 zone types. Every transition starts with the agent's
selected action, after which the current zone may modify the result:
- Static: no additional modification (boring and predictable)
- Deterministic-A: adds a learnable shift (primary learning target)
- Deterministic-B: adds a different learnable shift (secondary target)
- Noisy: adds a random one-step perturbation (partly unpredictable)
- Dynamic: switches between deterministic shifts (tests re-curiosity)
"""

import logging
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pygame

logger = logging.getLogger(__name__)


class ActionType(Enum):
    """Action directions."""
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class ZoneType(Enum):
    """Zone types in the grid world."""
    STATIC = "static"
    DETERMINISTIC_A = "deterministic_a"
    DETERMINISTIC_B = "deterministic_b"
    NOISY = "noisy"
    DYNAMIC = "dynamic"


class Zone:
    """Represents a zone in the grid world."""
    
    def __init__(
        self,
        zone_type: ZoneType,
        color: Tuple[int, int, int],
        rule: Optional[str] = None,
        switch_interval: Optional[int] = None,
        rules: Optional[List[str]] = None,
    ):
        self.zone_type = zone_type
        self.color = color
        self.rule = rule
        self.switch_interval = switch_interval
        self.rules = rules or []
        self.current_rule_index = 0
        self.step_counter = 0
    
    def get_rule(self) -> str:
        """Get current rule for this zone."""
        if self.zone_type == ZoneType.STATIC:
            return "stay"
        elif self.zone_type == ZoneType.NOISY:
            return "random"
        elif self.zone_type == ZoneType.DYNAMIC:
            if self.rules:
                return self.rules[self.current_rule_index]
            return "stay"
        else:
            return self.rule or "stay"
    
    def step(self) -> None:
        """Increment step counter and switch rules if needed."""
        if self.zone_type == ZoneType.DYNAMIC and self.switch_interval:
            self.step_counter += 1
            if self.step_counter >= self.switch_interval:
                self.current_rule_index = (self.current_rule_index + 1) % len(self.rules)
                self.step_counter = 0
                logger.debug(
                    f"Dynamic zone switched to rule: {self.rules[self.current_rule_index]}"
                )


class GridWorld:
    """
    Grid World environment for curious agent experiments.
    
    The environment consists of a grid with different zone types, each with
    specific transition rules. The agent learns to navigate while being
    intrinsically motivated by curiosity.
    """
    
    def __init__(
        self,
        grid_size: int = 10,
        config: Optional[Dict[str, Any]] = None,
        render: bool = False,
    ):
        """
        Initialize the grid world environment.
        
        Args:
            grid_size: Size of the grid (grid_size x grid_size)
            config: Configuration dictionary
            render: Whether to render with Pygame
        """
        self.grid_size = grid_size
        self.config = config or {}
        self.render = render
        
        # State space
        self.num_states = grid_size * grid_size
        self.state_dim = 2  # (row, col)
        
        # Action space
        self.num_actions = len(ActionType)
        self.action_space = list(ActionType)
        
        # Grid initialization
        self.grid: List[List[Zone]] = []
        self.goal_pos = (grid_size - 1, grid_size - 1)
        
        # Agent state
        self.agent_pos = (0, 0)
        self.step_count = 0
        self.episode_count = 0
        
        # Zone tracking
        self.zone_stats: Dict[ZoneType, int] = {z: 0 for z in ZoneType}
        
        # Pygame initialization
        self.screen: Optional[pygame.Surface] = None
        self.clock: Optional[pygame.time.Clock] = None
        self.cell_size = self.config.get("visualization", {}).get("cell_size", 50)
        
        # Initialize grid
        self._initialize_grid()
        
        logger.info(
            f"GridWorld initialized: {grid_size}x{grid_size}, "
            f"{self.num_states} states, {self.num_actions} actions"
        )
    
    def _initialize_grid(self) -> None:
        """Initialize the grid with zones."""
        zones_config = self.config.get("environment", {}).get("zones", {})
        
        # Create zone map
        self.grid = [[None for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        
        # Calculate number of cells for each zone type
        total_cells = self.grid_size * self.grid_size
        zone_counts = {}
        
        for zone_name, zone_conf in zones_config.items():
            percentage = zone_conf.get("percentage", 0.1)
            zone_counts[ZoneType(zone_name)] = int(total_cells * percentage)
        
        # Fill remaining cells with deterministic_a
        assigned = sum(zone_counts.values())
        remaining = total_cells - assigned
        if remaining > 0:
            zone_counts[ZoneType.DETERMINISTIC_A] = (
                zone_counts.get(ZoneType.DETERMINISTIC_A, 0) + remaining
            )
        
        # Create zone list
        zone_list = []
        for zone_type, count in zone_counts.items():
            for _ in range(count):
                zone_list.append(zone_type)
        
        # Shuffle and assign to grid
        np.random.shuffle(zone_list)
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                zone_type = zone_list.pop()
                zone_config = zones_config.get(zone_type.value, {})
                
                color = tuple(zone_config.get("color", [128, 128, 128]))
                rule = zone_config.get("rule", None)
                switch_interval = zone_config.get("switch_interval", None)
                rules = zone_config.get("rules", [])
                
                self.grid[i][j] = Zone(
                    zone_type=zone_type,
                    color=color,
                    rule=rule,
                    switch_interval=switch_interval,
                    rules=rules,
                )
        
        logger.info(f"Grid initialized with zone types: {list(zone_counts.keys())}")
    
    def reset(self) -> np.ndarray:
        """
        Reset the environment to initial state.
        
        Returns:
            Initial state as numpy array [row, col]
        """
        self.agent_pos = (0, 0)
        self.step_count = 0
        self.episode_count += 1
        
        # Reset zone stats
        self.zone_stats = {z: 0 for z in ZoneType}
        
        # Reset dynamic zones
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.grid[i][j].zone_type == ZoneType.DYNAMIC:
                    self.grid[i][j].current_rule_index = 0
                    self.grid[i][j].step_counter = 0
        
        logger.debug(f"Episode {self.episode_count} started")
        return self._get_state()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Take an action and return the next state, reward, done flag, and info.
        
        Args:
            action: Action to take (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT)
        
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        row, col = self.agent_pos
        zone = self.grid[row][col]
        
        # Track zone usage
        self.zone_stats[zone.zone_type] += 1
        
        # Get transition based on zone type
        next_pos = self._get_next_position(row, col, action, zone)
        
        # Update agent position
        self.agent_pos = next_pos
        self.step_count += 1
        
        # Update dynamic zones
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                self.grid[i][j].step()
        
        # Calculate reward
        reward = self._calculate_reward(next_pos)
        
        # Check if done
        done = next_pos == self.goal_pos or self.step_count >= self.config.get(
            "training", {}
        ).get("max_steps_per_episode", 200)
        
        # Info dictionary
        info = {
            "zone_type": zone.zone_type.value,
            "step_count": self.step_count,
            "zone_stats": dict(self.zone_stats),
        }
        
        logger.debug(
            f"Step {self.step_count}: action={action}, "
            f"pos={self.agent_pos}, reward={reward:.3f}"
        )
        
        return self._get_state(), reward, done, info
    
    def _get_next_position(
        self, row: int, col: int, action: int, zone: Zone
    ) -> Tuple[int, int]:
        """Apply the selected action, then the current zone's modifier.

        Applying the action first is important: the controller must be able to
        affect transitions for action values to be learnable. Zone rules add
        dynamics to the intended move instead of replacing it.
        """
        rule = zone.get_rule()

        # The agent's intended movement is always the base transition.
        next_row, next_col = self._apply_action(row, col, action)

        if rule == "stay" or zone.zone_type == ZoneType.STATIC:
            # Static means the environment adds no movement of its own.
            return (next_row, next_col)

        elif rule == "random" or zone.zone_type == ZoneType.NOISY:
            # Add local stochasticity without discarding the intended action.
            perturbation = np.random.randint(0, self.num_actions)
            return self._apply_action(
                next_row,
                next_col,
                perturbation,
            )

        elif rule == "shift_up":
            # Deterministic-A: add an upward cyclic shift.
            return ((next_row - 1) % self.grid_size, next_col)

        elif rule == "shift_right":
            # Deterministic-B: add a rightward cyclic shift.
            return (next_row, (next_col + 1) % self.grid_size)

        else:
            # Unknown rules must not remove control from the agent.
            return (next_row, next_col)
    
    def _apply_action(self, row: int, col: int, action: int) -> Tuple[int, int]:
        """Apply action to get next position."""
        if action == ActionType.UP.value:
            next_row = max(0, row - 1)
            return (next_row, col)
        elif action == ActionType.DOWN.value:
            next_row = min(self.grid_size - 1, row + 1)
            return (next_row, col)
        elif action == ActionType.LEFT.value:
            next_col = max(0, col - 1)
            return (row, next_col)
        elif action == ActionType.RIGHT.value:
            next_col = min(self.grid_size - 1, col + 1)
            return (row, next_col)
        else:
            return (row, col)
    
    def _calculate_reward(self, pos: Tuple[int, int]) -> float:
        """Calculate reward for reaching a position."""
        if pos == self.goal_pos:
            return self.config.get("environment", {}).get("goal", {}).get("reward", 1.0)
        return 0.0
    
    def _get_state(self) -> np.ndarray:
        """Get current state as numpy array."""
        return np.array(self.agent_pos, dtype=np.float32)
    
    def get_state_idx(self, state: Optional[np.ndarray] = None) -> int:
        """Convert state to discrete index."""
        if state is None:
            state = self.agent_pos
        else:
            state = tuple(state)
        
        row, col = state
        return int(row * self.grid_size + col)
    
    def get_zone_at(self, pos: Tuple[int, int]) -> Zone:
        """Get zone at a specific position."""
        row, col = pos
        return self.grid[row][col]
    
    def get_zone_stats(self) -> Dict[str, float]:
        """Get normalized zone usage statistics."""
        total = sum(self.zone_stats.values())
        if total == 0:
            return {z.value: 0.0 for z in ZoneType}
        return {z.value: count / total for z, count in self.zone_stats.items()}
    
    def init_pygame(self) -> None:
        """Initialize Pygame for rendering."""
        if not self.render:
            return
        
        pygame.init()
        width = self.grid_size * self.cell_size
        height = self.grid_size * self.cell_size
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Curious Agent Grid World")
        self.clock = pygame.time.Clock()
        
        logger.info("Pygame initialized for rendering")
    
    def render_frame(self) -> None:
        """Render the current frame."""
        if not self.render or self.screen is None:
            return
        
        # Clear screen
        self.screen.fill((0, 0, 0))
        
        # Draw grid
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                zone = self.grid[i][j]
                x = j * self.cell_size
                y = i * self.cell_size
                
                # Draw zone
                pygame.draw.rect(
                    self.screen,
                    zone.color,
                    (x, y, self.cell_size, self.cell_size),
                )
                
                # Draw grid lines
                pygame.draw.rect(
                    self.screen,
                    (50, 50, 50),
                    (x, y, self.cell_size, self.cell_size),
                    1,
                )
        
        # Draw goal
        goal_x = self.goal_pos[1] * self.cell_size
        goal_y = self.goal_pos[0] * self.cell_size
        pygame.draw.rect(
            self.screen,
            (0, 255, 0),
            (goal_x + 5, goal_y + 5, self.cell_size - 10, self.cell_size - 10),
        )
        
        # Draw agent
        agent_x = self.agent_pos[1] * self.cell_size
        agent_y = self.agent_pos[0] * self.cell_size
        pygame.draw.circle(
            self.screen,
            (255, 255, 255),
            (agent_x + self.cell_size // 2, agent_y + self.cell_size // 2),
            self.cell_size // 3,
        )
        
        # Update display
        pygame.display.flip()
        
        # Control frame rate
        if self.clock:
            self.clock.tick(self.config.get("visualization", {}).get("fps", 10))
    
    def close(self) -> None:
        """Close Pygame."""
        if self.render:
            pygame.quit()
            logger.info("Pygame closed")
    
    def __repr__(self) -> str:
        return (
            f"GridWorld(grid_size={self.grid_size}, "
            f"num_states={self.num_states}, "
            f"num_actions={self.num_actions})"
        )
