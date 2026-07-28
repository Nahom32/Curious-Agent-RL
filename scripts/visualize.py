"""
Visualization Script for Curious Agent

This script provides visualization tools for the curious agent:
1. Pygame renderer for grid world
2. Heatmap overlay for C's predicted error
3. Real-time matplotlib plots
"""

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from curious_agent.env.grid_world import GridWorld
from curious_agent.agents.tabular_curious import TabularCuriousAgent
from curious_agent.agents.dqn_curious import DNQCuriousAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

TRAINING_METRIC_PATTERN = re.compile(
    r"Episode (?P<episode>\d+)/\d+: "
    r"Avg Reward=(?P<reward>[-+]?\d*\.?\d+), "
    r"Avg Curiosity=(?P<curiosity>[-+]?\d*\.?\d+), "
    r"Avg Length=(?P<length>[-+]?\d*\.?\d+), "
    r"Epsilon=(?P<epsilon>[-+]?\d*\.?\d+)"
)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def visualize_tabular_agent(config: dict, num_episodes: int = 100) -> None:
    """
    Visualize the tabular curious agent.
    
    Args:
        config: Configuration dictionary
        num_episodes: Number of episodes to visualize
    """
    # Initialize environment with rendering
    env_config = config.get("environment", {})
    grid_size = env_config.get("grid_size", 10)
    
    env = GridWorld(
        grid_size=grid_size,
        config=config,
        render=True,
    )
    env.init_pygame()
    
    # Initialize agent
    agent = TabularCuriousAgent(
        num_states=env.num_states,
        num_actions=env.num_actions,
        config=config,
    )
    
    # Training settings
    training_config = config.get("training", {})
    max_steps = training_config.get("max_steps_per_episode", 200)
    
    logger.info(f"Visualizing for {num_episodes} episodes")
    
    for episode in range(num_episodes):
        state = env.reset()
        state_idx = env.get_state_idx(state)
        
        done = False
        step = 0
        
        while not done and step < max_steps:
            # Handle events
            for event in __import__("pygame").event.get():
                if event.type == __import__("pygame").QUIT:
                    env.close()
                    return
            
            # Get action
            action = agent.select_action(state_idx)
            
            # Get C's confidence
            o_c_before = agent.get_confidence_before(state_idx, action)
            
            # Execute action
            next_state, r_ext, done, info = env.step(action)
            next_state_idx = env.get_state_idx(next_state)
            
            # Update model and confidence
            agent.update_model(state_idx, action, next_state_idx)
            agent.update_confidence(
                state_idx, action, 
                1.0 if agent.M[state_idx, action] != next_state_idx else 0.0
            )
            o_c_after = agent.get_confidence_before(state_idx, action)
            
            # Compute curiosity reward
            r_curiosity = agent.compute_curiosity_reward(o_c_before, o_c_after)
            r_total = r_ext + agent.beta * r_curiosity
            
            # Update Q
            agent.update_q(state_idx, action, r_total, next_state_idx, done)
            
            state_idx = next_state_idx
            step += 1
            
            # Render
            env.render_frame()
        
        agent.decay_epsilon()
        
        logger.info(
            f"Episode {episode + 1}/{num_episodes} completed, "
            f"steps={step}, epsilon={agent.epsilon:.3f}"
        )
    
    env.close()


def visualize_heatmap(agent: TabularCuriousAgent, grid_size: int = 10) -> None:
    """
    Visualize C's predicted error as a heatmap.
    
    Args:
        agent: Trained tabular agent
        grid_size: Size of the grid
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("matplotlib not installed. Run: pip install matplotlib")
        return
    
    # Create heatmap of C's predictions
    heatmap = np.zeros((grid_size, grid_size))
    
    for i in range(grid_size):
        for j in range(grid_size):
            state_idx = i * grid_size + j
            # Average C's prediction across all actions
            heatmap[i, j] = np.mean(agent.C[state_idx])
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Heatmap
    im = axes[0].imshow(heatmap, cmap="hot", interpolation="nearest")
    axes[0].set_title("C's Predicted Error (Average)")
    axes[0].set_xlabel("Column")
    axes[0].set_ylabel("Row")
    plt.colorbar(im, ax=axes[0])
    
    # Q-values
    q_heatmap = np.max(agent.Q, axis=1).reshape(grid_size, grid_size)
    im2 = axes[1].imshow(q_heatmap, cmap="viridis", interpolation="nearest")
    axes[1].set_title("Max Q-Values")
    axes[1].set_xlabel("Column")
    axes[1].set_ylabel("Row")
    plt.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig("heatmap_analysis.png", dpi=150)
    plt.show()
    
    logger.info("Heatmap saved to heatmap_analysis.png")


def plot_training_curves(log_file: str) -> None:
    """
    Plot training curves from log file.
    
    Args:
        log_file: Path to log file
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("matplotlib not installed. Run: pip install matplotlib")
        return
    
    # Parse log file
    episodes = []
    rewards = []
    curiosities = []
    lengths = []
    epsilons = []
    
    log_path = Path(log_file)
    if not log_path.is_file():
        logger.error(
            "Training log not found: %s. Capture a DQN run before plotting.",
            log_path,
        )
        return

    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            match = TRAINING_METRIC_PATTERN.search(line)
            if match is None:
                continue

            episodes.append(int(match.group("episode")))
            rewards.append(float(match.group("reward")))
            curiosities.append(float(match.group("curiosity")))
            lengths.append(float(match.group("length")))
            epsilons.append(float(match.group("epsilon")))
    
    if not episodes:
        logger.warning(
            "No episode metrics found in %s. The file must contain progress "
            "lines with 'Episode' and 'Avg Reward'. A completed checkpoint "
            "does not contain the historical curve.",
            log_path,
        )
        return
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Rewards
    axes[0, 0].plot(episodes, rewards, "b-", alpha=0.7, label="Reward")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Average Reward")
    axes[0, 0].set_title("Training Rewards")
    axes[0, 0].grid(True)
    axes[0, 0].legend()
    
    # Curiosity
    axes[0, 1].plot(episodes, curiosities, "r-", alpha=0.7, label="Curiosity")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Average Curiosity Reward")
    axes[0, 1].set_title("Curiosity Rewards")
    axes[0, 1].grid(True)
    axes[0, 1].legend()
    
    # Episode lengths
    axes[1, 0].plot(episodes, lengths, "g-", alpha=0.7, label="Length")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Average Episode Length")
    axes[1, 0].set_title("Episode Lengths")
    axes[1, 0].grid(True)
    axes[1, 0].legend()
    
    # Epsilon
    axes[1, 1].plot(episodes, epsilons, "m-", alpha=0.7, label="Epsilon")
    axes[1, 1].set_xlabel("Episode")
    axes[1, 1].set_ylabel("Epsilon")
    axes[1, 1].set_title("Exploration Rate")
    axes[1, 1].grid(True)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    plt.show()
    
    logger.info("Training curves saved to training_curves.png")


def visualize_zone_distribution(agent: TabularCuriousAgent, grid_size: int = 10) -> None:
    """
    Visualize time spent in each zone type.
    
    Args:
        agent: Trained tabular agent
        grid_size: Size of the grid
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("matplotlib not installed. Run: pip install matplotlib")
        return
    
    # This is a placeholder - in practice, we'd track this during training
    # For now, create a simple visualization of Q-values by zone type
    
    # Zone types (simplified)
    zone_values = {
        "Static": [],
        "Deterministic A": [],
        "Deterministic B": [],
        "Noisy": [],
        "Dynamic": [],
    }
    
    # This would need to be integrated with actual zone tracking
    # For now, just show a sample plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    zones = list(zone_values.keys())
    values = [np.random.rand() for _ in zones]  # Placeholder
    
    bars = ax.bar(zones, values, color=["gray", "green", "blue", "red", "yellow"])
    ax.set_xlabel("Zone Type")
    ax.set_ylabel("Average Q-Value")
    ax.set_title("Agent Performance by Zone Type")
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{val:.2f}",
            ha="center",
            va="bottom",
        )
    
    plt.tight_layout()
    plt.savefig("zone_distribution.png", dpi=150)
    plt.show()
    
    logger.info("Zone distribution saved to zone_distribution.png")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Visualize Curious Agent")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/tabular.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["live", "heatmap", "curves", "zones"],
        default="live",
        help="Visualization mode",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of episodes to visualize",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="training_tabular.log",
        help="Path to log file for curve plotting",
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    if args.mode == "live":
        visualize_tabular_agent(config, num_episodes=args.episodes)
    elif args.mode == "heatmap":
        # Load trained agent
        agent = TabularCuriousAgent(
            num_states=100,
            num_actions=4,
            config=config,
        )
        # Load from checkpoint if available
        checkpoint_path = Path(config.get("paths", {}).get("checkpoints", "checkpoints/tabular"))
        final_path = checkpoint_path / "agent_final.npz"
        if final_path.exists():
            data = np.load(final_path)
            agent.Q = data["Q"]
            agent.M = data["M"]
            agent.C = data["C"]
            agent.epsilon = float(data["epsilon"])
            logger.info(f"Loaded agent from {final_path}")
        
        visualize_heatmap(agent, grid_size=config.get("environment", {}).get("grid_size", 10))
    elif args.mode == "curves":
        plot_training_curves(args.log_file)
    elif args.mode == "zones":
        agent = TabularCuriousAgent(
            num_states=100,
            num_actions=4,
            config=config,
        )
        visualize_zone_distribution(agent, grid_size=config.get("environment", {}).get("grid_size", 10))


if __name__ == "__main__":
    main()
