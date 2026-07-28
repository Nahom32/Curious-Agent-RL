"""Tests for training-log visualization."""

from scripts.visualize import TRAINING_METRIC_PATTERN


def test_training_metric_parser_maps_each_series_correctly():
    """Logger prefixes must not shift reward, curiosity, length, or epsilon."""
    line = (
        "2026-07-28 10:04:27,392 - scripts.train_dqn - INFO - "
        "Episode 10/2000: Avg Reward=0.229, Avg Curiosity=-0.071, "
        "Avg Length=168.4, Epsilon=0.995, Buffer Size=1684"
    )

    match = TRAINING_METRIC_PATTERN.search(line)

    assert match is not None
    assert int(match.group("episode")) == 10
    assert float(match.group("reward")) == 0.229
    assert float(match.group("curiosity")) == -0.071
    assert float(match.group("length")) == 168.4
    assert float(match.group("epsilon")) == 0.995
