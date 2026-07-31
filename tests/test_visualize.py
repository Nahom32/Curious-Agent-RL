"""Tests for training-log visualization."""

import json
from pathlib import Path

from scripts.visualize import (
    TRAINING_METRIC_PATTERN,
    VANILLA_DQN_PATTERN,
    parse_log_file,
)


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


def test_vanilla_dqn_metric_parser():
    """Vanilla DQN uses different metric names in its log lines."""
    line = (
        "2026-07-30 12:00:00,000 - scripts.train_vanilla_dqn - INFO - "
        "Episode 50/2000: Avg External Return=0.350, "
        "Success Rate=0.200, Avg Length=120.0, Epsilon=0.975"
    )

    match = VANILLA_DQN_PATTERN.search(line)

    assert match is not None
    assert int(match.group("episode")) == 50
    assert float(match.group("external_return")) == 0.350
    assert float(match.group("success_rate")) == 0.200
    assert float(match.group("length")) == 120.0
    assert float(match.group("epsilon")) == 0.975


def test_parse_log_file_curious(tmp_path: Path):
    """parse_log_file correctly reads a curious agent log file."""
    log = tmp_path / "training.log"
    log.write_text(
        "2026-07-28 10:04:27,392 - INFO - "
        "Episode 10/2000: Avg Reward=0.229, Avg Curiosity=-0.071, "
        "Avg Length=168.4, Epsilon=0.995, Buffer Size=1684\n"
        "2026-07-28 10:04:28,696 - INFO - "
        "Episode 20/2000: Avg Reward=0.399, Avg Curiosity=-0.001, "
        "Avg Length=137.4, Epsilon=0.990, Buffer Size=3058\n"
    )

    records = parse_log_file(str(log))
    assert records is not None
    assert records["episode"] == [10.0, 20.0]
    assert records["reward"] == [0.229, 0.399]
    assert records["curiosity"] == [-0.071, -0.001]
    assert records["length"] == [168.4, 137.4]
    assert records["epsilon"] == [0.995, 0.990]
    assert "success_rate" not in records


def test_parse_log_file_vanilla(tmp_path: Path):
    """parse_log_file correctly reads a vanilla DQN log file."""
    log = tmp_path / "training.log"
    log.write_text(
        "2026-07-30 12:00:00,000 - INFO - "
        "Episode 50/2000: Avg External Return=0.350, "
        "Success Rate=0.200, Avg Length=120.0, Epsilon=0.975\n"
        "2026-07-30 12:00:01,000 - INFO - "
        "Episode 60/2000: Avg External Return=0.450, "
        "Success Rate=0.300, Avg Length=110.0, Epsilon=0.970\n"
    )

    records = parse_log_file(str(log))
    assert records is not None
    assert records["episode"] == [50.0, 60.0]
    assert records["reward"] == [0.350, 0.450]
    assert records["success_rate"] == [0.200, 0.300]
    assert records["length"] == [120.0, 110.0]
    assert records["epsilon"] == [0.975, 0.970]
    assert "curiosity" not in records


def test_parse_log_file_empty_file(tmp_path: Path):
    """parse_log_file returns None for a file with no matching lines."""
    log = tmp_path / "empty.log"
    log.write_text("some random text\nno metrics here\n")

    records = parse_log_file(str(log))
    assert records is None


def test_parse_log_file_missing_file(tmp_path: Path):
    """parse_log_file returns None when the file does not exist."""
    records = parse_log_file(str(tmp_path / "nonexistent.log"))
    assert records is None


def test_comparison_legend_has_three_entries(tmp_path: Path, monkeypatch):
    """A three-agent comparison plot shows one legend entry per agent."""
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")
    import scripts.visualize as visualize

    monkeypatch.setattr(plt, "show", lambda: None)

    def write_log(name: str, reward: float) -> Path:
        path = tmp_path / f"{name}.log"
        path.write_text(
            f"INFO - Episode 10/100: Avg Reward={reward}, "
            f"Avg Curiosity=0.001, Avg Length=50.0, Epsilon=0.9\n"
            f"INFO - Episode 20/100: Avg Reward={reward + 0.1}, "
            f"Avg Curiosity=0.002, Avg Length=40.0, Epsilon=0.8\n"
        )
        return path

    log_paths = [
        write_log("tabular", 0.5),
        write_log("vanilla", 0.7),
        write_log("dqn", 0.9),
    ]
    labels = ["Tabular Q-Learning", "Vanilla DQN", "Curious DQN"]

    visualize.plot_comparison(
        [str(p) for p in log_paths], labels=labels
    )

    fig = plt.gcf()
    try:
        assert len(fig.axes) == 4
        for ax in fig.axes:
            legend_labels = [
                text.get_text() for text in ax.get_legend().get_texts()
            ]
            assert set(legend_labels) == set(labels)
    finally:
        plt.close(fig)
