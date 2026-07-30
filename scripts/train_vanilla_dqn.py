"""Train the external-reward-only vanilla DQN ablation."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

project_root = Path(__file__).resolve().parent.parent
src_root = project_root / "src"
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from curious_agent.agents.dqn import DQNAgent
from curious_agent.env.grid_world import GridWorld

logger = logging.getLogger(__name__)


def load_config(config_path: str | Path) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError("Vanilla DQN configuration must be a YAML mapping")
    return config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _configure_file_logging(log_dir: Path) -> logging.Handler:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = (log_dir / "training.log").resolve()
    for configured_logger in (logger, logging.getLogger()):
        for handler in configured_logger.handlers:
            if (
                isinstance(handler, logging.FileHandler)
                and Path(handler.baseFilename).resolve() == log_path
            ):
                return handler

    handler = logging.FileHandler(log_path)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return handler


def _write_results(
    records: list[dict[str, float | int]],
    result_dir: Path,
) -> dict[str, float | int | str]:
    result_dir.mkdir(parents=True, exist_ok=True)
    csv_path = result_dir / "episodes.csv"
    fieldnames = [
        "episode",
        "external_return",
        "success",
        "positive_feedback_count",
        "positive_feedback_rate",
        "episode_length",
        "epsilon",
        "mean_q_loss",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    window = records[-min(100, len(records)) :]
    summary: dict[str, float | int | str] = {
        "agent": "vanilla_dqn",
        "episodes": len(records),
        "summary_window": len(window),
        "mean_external_return": float(
            np.mean([row["external_return"] for row in window])
        ),
        "success_rate": float(np.mean([row["success"] for row in window])),
        "mean_positive_feedback_rate": float(
            np.mean([row["positive_feedback_rate"] for row in window])
        ),
        "mean_episode_length": float(
            np.mean([row["episode_length"] for row in window])
        ),
    }
    with (result_dir / "summary.json").open("w", encoding="utf-8") as output:
        json.dump(summary, output, indent=2)
        output.write("\n")
    return summary


def train_vanilla_dqn(
    config: dict[str, Any],
    render: bool = False,
) -> dict[str, float | int | str]:
    """Train vanilla DQN and return its external-feedback summary."""
    paths = config.get("paths", {})
    checkpoint_dir = Path(paths.get("checkpoints", "checkpoints/vanilla_dqn"))
    log_dir = Path(paths.get("logs", "logs/vanilla_dqn"))
    result_dir = Path(paths.get("results", "results/vanilla_dqn"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    _configure_file_logging(log_dir)

    training_config = config.get("training", {})
    seed = int(training_config.get("seed", 42))
    set_seed(seed)

    env_config = config.get("environment", {})
    env = GridWorld(
        grid_size=int(env_config.get("grid_size", 10)),
        config=config,
        render=render,
    )
    if render:
        env.init_pygame()

    agent = DQNAgent(
        state_dim=env.state_dim,
        num_actions=env.num_actions,
        config=config,
    )
    num_episodes = int(training_config.get("num_episodes", 2000))
    max_steps = int(training_config.get("max_steps_per_episode", 200))
    log_interval = int(training_config.get("log_interval", 10))
    save_interval = int(training_config.get("save_interval", 100))
    records: list[dict[str, float | int]] = []

    logger.info(
        "Starting vanilla DQN: episodes=%s, grid=%sx%s, seed=%s",
        num_episodes,
        env.grid_size,
        env.grid_size,
        seed,
    )

    try:
        for episode in range(1, num_episodes + 1):
            state = env.reset()
            external_return = 0.0
            positive_feedback_count = 0
            losses: list[float] = []
            done = False
            step = 0

            while not done and step < max_steps:
                action = agent.select_action(state)
                next_state, external_reward, done, _ = env.step(action)
                agent.store_experience(
                    state, action, external_reward, next_state, done
                )
                loss = agent.update_q_network()
                if loss is not None:
                    losses.append(loss)

                external_return += external_reward
                positive_feedback_count += int(external_reward > 0)
                state = next_state
                step += 1
                if render:
                    env.render_frame()

            agent.decay_epsilon()
            record: dict[str, float | int] = {
                "episode": episode,
                "external_return": external_return,
                "success": int(positive_feedback_count > 0),
                "positive_feedback_count": positive_feedback_count,
                "positive_feedback_rate": (
                    positive_feedback_count / step if step else 0.0
                ),
                "episode_length": step,
                "epsilon": agent.epsilon,
                "mean_q_loss": float(np.mean(losses)) if losses else 0.0,
            }
            records.append(record)

            if episode % log_interval == 0:
                window = records[-log_interval:]
                logger.info(
                    "Episode %s/%s: Avg External Return=%.3f, "
                    "Success Rate=%.3f, Avg Length=%.1f, Epsilon=%.3f",
                    episode,
                    num_episodes,
                    np.mean([row["external_return"] for row in window]),
                    np.mean([row["success"] for row in window]),
                    np.mean([row["episode_length"] for row in window]),
                    agent.epsilon,
                )

            if episode % save_interval == 0:
                agent.save(checkpoint_dir / f"agent_episode_{episode}.pt")

        agent.save(checkpoint_dir / "agent_final.pt")
        summary = _write_results(records, result_dir)
        logger.info(
            "Vanilla DQN complete: mean external return=%.3f, "
            "success rate=%.3f",
            summary["mean_external_return"],
            summary["success_rate"],
        )
        return summary
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the external-reward-only vanilla DQN baseline."
    )
    parser.add_argument(
        "--config",
        default="configs/vanilla_dqn.yaml",
        help="Path to the vanilla DQN YAML configuration.",
    )
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    config = load_config(args.config)
    if args.seed is not None:
        config.setdefault("training", {})["seed"] = args.seed
    train_vanilla_dqn(config, render=args.render)


if __name__ == "__main__":
    main()
