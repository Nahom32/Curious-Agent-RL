"""
Training Script for Tabular Curious Agent

This script trains the tabular version of Schmidhuber's curious agent
using lookup tables for M, C, and Q.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from curious_agent.env.grid_world import GridWorld
from curious_agent.agents.tabular_curious import TabularCuriousAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("training_tabular.log"),
    ],
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train_tabular(config: dict, render: bool = False) -> None:
    """
    Train the tabular curious agent.
    
    Args:
        config: Configuration dictionary
        render: Whether to render with Pygame
    """
    # Create directories
    paths = config.get("paths", {})
    checkpoint_dir = Path(paths.get("checkpoints", "checkpoints/tabular"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize environment
    env_config = config.get("environment", {})
    grid_size = env_config.get("grid_size", 10)
    
    env = GridWorld(
        grid_size=grid_size,
        config=config,
        render=render,
    )
    
    if render:
        env.init_pygame()
    
    # Initialize agent
    agent = TabularCuriousAgent(
        num_states=env.num_states,
        num_actions=env.num_actions,
        config=config,
    )
    
    # Training settings
    training_config = config.get("training", {})
    num_episodes = training_config.get("num_episodes", 1000)
    max_steps = training_config.get("max_steps_per_episode", 200)
    log_interval = training_config.get("log_interval", 10)
    save_interval = training_config.get("save_interval", 100)
    
    # Training loop
    episode_rewards = []
    episode_curiosity_rewards = []
    episode_lengths = []
    
    logger.info(f"Starting training for {num_episodes} episodes")
    logger.info(f"Grid size: {grid_size}x{grid_size}")
    logger.info(f"Agent: {agent}")
    
    for episode in range(num_episodes):
        state = env.reset()
        state_idx = env.get_state_idx(state)
        
        total_reward = 0.0
        total_curiosity = 0.0
        done = False
        step = 0
        
        while not done and step < max_steps:
            # 1. Get C's confidence BEFORE action
            o_c_before = agent.get_confidence_before(state_idx, 0)  # Action doesn't matter yet
            
            # 2. Choose action (ε-greedy)
            action = agent.select_action(state_idx)
            
            # Get C's confidence for the actual state-action pair
            o_c_before = agent.get_confidence_before(state_idx, action)
            
            # 3. Execute action
            next_state, r_ext, done, info = env.step(action)
            next_state_idx = env.get_state_idx(next_state)
            
            # 4. Compute actual model error
            predicted_next = agent.M[state_idx, action]
            actual_error = 1.0 if predicted_next != next_state_idx else 0.0
            
            # 5. Update Model M
            agent.update_model(state_idx, action, next_state_idx)
            
            # 6. Update Confidence C
            agent.update_confidence(state_idx, action, actual_error)
            o_c_after = agent.get_confidence_before(state_idx, action)
            
            # 7. Compute curiosity reward
            r_curiosity = agent.compute_curiosity_reward(o_c_before, o_c_after)
            
            # 8. Compute total reward
            beta = agent.beta
            r_total = r_ext + beta * r_curiosity
            
            # 9. Update Q
            agent.update_q(state_idx, action, r_total, next_state_idx, done)
            
            # Update statistics
            total_reward += r_total
            total_curiosity += r_curiosity
            
            # Move to next state
            state_idx = next_state_idx
            step += 1
            
            # Render if enabled
            if render:
                env.render_frame()
        
        # Decay epsilon
        agent.decay_epsilon()
        
        # Record episode statistics
        episode_rewards.append(total_reward)
        episode_curiosity_rewards.append(total_curiosity)
        episode_lengths.append(step)
        
        # Log progress
        if (episode + 1) % log_interval == 0:
            avg_reward = sum(episode_rewards[-log_interval:]) / log_interval
            avg_curiosity = sum(episode_curiosity_rewards[-log_interval:]) / log_interval
            avg_length = sum(episode_lengths[-log_interval:]) / log_interval
            
            logger.info(
                f"Episode {episode + 1}/{num_episodes}: "
                f"Avg Reward={avg_reward:.3f}, "
                f"Avg Curiosity={avg_curiosity:.3f}, "
                f"Avg Length={avg_length:.1f}, "
                f"Epsilon={agent.epsilon:.3f}"
            )
        
        # Save checkpoint
        if (episode + 1) % save_interval == 0:
            checkpoint_path = checkpoint_dir / f"agent_episode_{episode + 1}.pt"
            # Save Q-table
            import numpy as np
            np.savez(
                checkpoint_path,
                Q=agent.Q,
                M=agent.M,
                C=agent.C,
                epsilon=agent.epsilon,
            )
            logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    # Save final model
    final_path = checkpoint_dir / "agent_final.npz"
    import numpy as np
    np.savez(
        final_path,
        Q=agent.Q,
        M=agent.M,
        C=agent.C,
        epsilon=agent.epsilon,
    )
    logger.info(f"Final model saved to {final_path}")
    
    # Print summary
    logger.info("\n" + "=" * 50)
    logger.info("Training Complete!")
    logger.info(f"Total Episodes: {num_episodes}")
    logger.info(f"Final Epsilon: {agent.epsilon:.3f}")
    summary_window = min(100, len(episode_rewards))
    logger.info(
        f"Average Reward (last {summary_window}): "
        f"{sum(episode_rewards[-summary_window:]) / summary_window:.3f}"
    )
    logger.info(
        f"Average Curiosity (last {summary_window}): "
        f"{sum(episode_curiosity_rewards[-summary_window:]) / summary_window:.3f}"
    )
    logger.info("=" * 50)
    
    # Close environment
    env.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train Tabular Curious Agent")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/tabular.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render with Pygame",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed",
    )
    
    args = parser.parse_args()
    
    # Set random seed
    if args.seed is not None:
        import numpy as np
        np.random.seed(args.seed)
    
    # Load config
    config = load_config(args.config)
    
    # Train
    train_tabular(config, render=args.render)


if __name__ == "__main__":
    main()
