"""End-to-end smoke test for the vanilla DQN trainer."""

import csv
import json

from scripts.train_vanilla_dqn import train_vanilla_dqn


def test_train_vanilla_dqn_writes_external_reward_results(tmp_path):
    config = {
        "environment": {
            "grid_size": 3,
            "zones": {
                "static": {
                    "percentage": 1.0,
                    "color": [128, 128, 128],
                }
            },
            "goal": {"position": [2, 2], "reward": 1.0},
        },
        "networks": {
            "q_network": {
                "hidden_dims": [8],
                "learning_rate": 0.001,
                "use_dueling": False,
            }
        },
        "agent": {
            "epsilon": 0.5,
            "epsilon_min": 0.1,
            "epsilon_decay": 0.9,
            "gamma": 0.99,
        },
        "training": {
            "num_episodes": 2,
            "max_steps_per_episode": 3,
            "batch_size": 1,
            "buffer_size": 10,
            "min_buffer_size": 1,
            "target_update_frequency": 1,
            "tau": 0.5,
            "log_interval": 1,
            "save_interval": 2,
            "seed": 7,
        },
        "paths": {
            "checkpoints": str(tmp_path / "checkpoints"),
            "logs": str(tmp_path / "logs"),
            "results": str(tmp_path / "results"),
        },
    }

    summary = train_vanilla_dqn(config)

    assert summary["agent"] == "vanilla_dqn"
    assert (tmp_path / "checkpoints" / "agent_final.pt").is_file()
    with (tmp_path / "results" / "episodes.csv").open(
        encoding="utf-8"
    ) as result_file:
        rows = list(csv.DictReader(result_file))
    assert len(rows) == 2
    assert "external_return" in rows[0]
    assert "positive_feedback_rate" in rows[0]

    saved_summary = json.loads(
        (tmp_path / "results" / "summary.json").read_text(encoding="utf-8")
    )
    assert saved_summary == summary
