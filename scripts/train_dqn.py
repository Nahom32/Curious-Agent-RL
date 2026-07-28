"""
Training Script for DQN Curious Agent

This script trains the DQN version of Schmidhuber's curious agent
using neural networks for M, C, and Q.
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
from curious_agent.agents.dqn_curious import DNQCuriousAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("training_dqn.log"),
    ],
)
logger = logging.getLogger(__name__)


def ensure_training_log(log_path: str | Path = "training_dqn.log") -> logging.Handler | None:
    """Ensure DQN progress is written even when a parent configured logging.

    ``logging.basicConfig`` is intentionally a no-op when the unified runner
    has already installed a root handler. In that case the stream output still
    appeared, but no DQN file handler was created.
    """
    resolved_path = Path(log_path).resolve()

    for configured_logger in (logger, logging.getLogger()):
        for handler in configured_logger.handlers:
            if not isinstance(handler, logging.FileHandler):
                continue
            if Path(handler.baseFilename).resolve() == resolved_path:
                return None

    file_handler = logging.FileHandler(resolved_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    )
    if logger.getEffectiveLevel() > logging.INFO:
        logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    return file_handler


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train_dqn(config: dict, render: bool = False) -> None:
    """
    Train the DQN curious agent.
    
    Args:
        config: Configuration dictionary
        render: Whether to render with Pygame
    """
    # The unified runner configures console logging before importing this
    # module, which makes the module-level basicConfig call a no-op. Install
    # the expected file handler explicitly so visualization data is retained.
    ensure_training_log()

    # Create directories
    paths = config.get("paths", {})
    checkpoint_dir = Path(paths.get("checkpoints", "checkpoints/dqn"))
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
    agent = DNQCuriousAgent(
        state_dim=env.state_dim,
        num_actions=env.num_actions,
        config=config,
    )
    
    # Training settings
    training_config = config.get("training", {})
    num_episodes = training_config.get("num_episodes", 2000)
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
        
        total_reward = 0.0
        total_curiosity = 0.0
        done = False
        step = 0
        
        while not done and step < max_steps:
            # 1. Get C's confidence BEFORE action
            action = agent.select_action(state)
            o_c_before = agent.get_confidence_before(state, action)
            
            # 2. Execute action
            next_state, r_ext, done, info = env.step(action)
            
            # 3. Compute actual model error
            import torch
            state_tensor = agent._encode_state(state).unsqueeze(0)
            action_tensor = agent._encode_action(action).unsqueeze(0)
            next_state_tensor = agent._encode_state(next_state).unsqueeze(0)
            
            predicted_next = agent.world_model.predict(state_tensor[0], action)
            actual_error = torch.nn.MSELoss()(predicted_next, next_state_tensor[0]).item()
            
            # 4. Update Model M
            agent.update_model(state, action, next_state)
            
            # 5. Update Confidence C
            agent.update_confidence(state, action, actual_error)
            o_c_after = agent.get_confidence_before(state, action)
            
            # 6. Compute curiosity reward
            r_curiosity = agent.compute_curiosity_reward(o_c_before, o_c_after)
            
            # 7. Compute total reward
            beta = agent.beta
            r_total = r_ext + beta * r_curiosity
            
            # 8. Store experience
            agent.store_experience(state, action, r_total, next_state, done)
            
            # 9. Update Q-network
            agent.update_q_network()
            
            # Update statistics
            total_reward += r_total
            total_curiosity += r_curiosity
            
            # Move to next state
            state = next_state
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
            
            stats = agent.get_statistics()
            
            logger.info(
                f"Episode {episode + 1}/{num_episodes}: "
                f"Avg Reward={avg_reward:.3f}, "
                f"Avg Curiosity={avg_curiosity:.3f}, "
                f"Avg Length={avg_length:.1f}, "
                f"Epsilon={stats['epsilon']:.3f}, "
                f"Buffer Size={stats['buffer_size']}"
            )
        
        # Save checkpoint
        if (episode + 1) % save_interval == 0:
            checkpoint_path = checkpoint_dir / f"agent_episode_{episode + 1}.pt"
            agent.save(str(checkpoint_path))
            logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    # Save final model
    final_path = checkpoint_dir / "agent_final.pt"
    agent.save(str(final_path))
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
    parser = argparse.ArgumentParser(description="Train DQN Curious Agent")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dqn.yaml",
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
        import torch
        torch.manual_seed(args.seed)
    
    # Load config
    config = load_config(args.config)
    
    # Train
    train_dqn(config, render=args.render)


if __name__ == "__main__":
    main()
