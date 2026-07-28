"""Regression tests for DQN training-log configuration."""

import logging

from scripts import train_dqn


def test_ensure_training_log_adds_file_handler(tmp_path):
    """File logging works when another entry point configured the root logger."""
    log_path = tmp_path / "training_dqn.log"
    handler = train_dqn.ensure_training_log(log_path)

    assert handler is not None
    try:
        train_dqn.logger.info(
            "Episode 10/20: Avg Reward=1.000, Avg Curiosity=0.000"
        )
        handler.flush()
        assert "Episode 10/20" in log_path.read_text(encoding="utf-8")
    finally:
        train_dqn.logger.removeHandler(handler)
        handler.close()


def test_ensure_training_log_does_not_duplicate_handler(tmp_path):
    """Repeated setup calls do not duplicate records in the same file."""
    log_path = tmp_path / "training_dqn.log"
    handler = train_dqn.ensure_training_log(log_path)

    assert handler is not None
    try:
        assert train_dqn.ensure_training_log(log_path) is None
    finally:
        train_dqn.logger.removeHandler(handler)
        handler.close()
