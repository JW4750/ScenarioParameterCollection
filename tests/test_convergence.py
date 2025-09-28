"""Tests for the convergence analysis workflow."""

from __future__ import annotations

from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from scenario_parameter_collection.convergence import (
    DistanceTriple,
    ScenarioConvergenceAnalyzer,
    convergence_steps_to_dataframe,
)
from scenario_parameter_collection.convergence_cli import main as convergence_main


def make_synthetic_tracks() -> pd.DataFrame:
    frames = np.arange(0, 50)
    base_columns = {
        "frame": frames,
        "leftPrecedingId": np.zeros_like(frames),
        "leftAlongsideId": np.zeros_like(frames),
        "leftFollowingId": np.zeros_like(frames),
        "rightPrecedingId": np.zeros_like(frames),
        "rightAlongsideId": np.zeros_like(frames),
        "rightFollowingId": np.zeros_like(frames),
        "yAcceleration": np.zeros_like(frames, dtype=float),
    }

    lane_series = np.where(frames < 30, 2, 1)
    y_velocity = np.zeros_like(frames, dtype=float)
    y_velocity[(frames >= 28) & (frames <= 32)] = 0.5

    ego = pd.DataFrame(base_columns)
    ego["id"] = 1
    ego["precedingId"] = 2
    ego["laneId"] = lane_series
    ego["xVelocity"] = 30.0
    ego["yVelocity"] = y_velocity
    ego["xAcceleration"] = 0.0
    ego["dhw"] = 20.0
    ego["thw"] = 1.0
    ego["ttc"] = 10.0

    lead = pd.DataFrame(base_columns)
    lead["id"] = 2
    lead["precedingId"] = 0
    lead["laneId"] = 2
    lead["xVelocity"] = 30.0
    lead["yVelocity"] = 0.0
    lead["xAcceleration"] = 0.0
    lead["dhw"] = np.nan
    lead["thw"] = np.nan
    lead["ttc"] = np.nan

    tracks = pd.concat([ego, lead], ignore_index=True, sort=False)
    return tracks[
        [
            "id",
            "frame",
            "precedingId",
            "leftPrecedingId",
            "leftAlongsideId",
            "leftFollowingId",
            "rightPrecedingId",
            "rightAlongsideId",
            "rightFollowingId",
            "laneId",
            "xVelocity",
            "yVelocity",
            "xAcceleration",
            "yAcceleration",
            "dhw",
            "thw",
            "ttc",
        ]
    ]


def to_highd_tracks(df: pd.DataFrame) -> pd.DataFrame:
    augmented = df.copy()
    frame_time_s = augmented["frame"].astype(float) / 25.0
    augmented["x"] = 100.0 + frame_time_s * augmented["xVelocity"].astype(float)
    lane_width = 3.5
    augmented["y"] = augmented["laneId"].astype(float) * lane_width - lane_width / 2
    augmented["width"] = 2.0
    augmented["length"] = 4.5
    augmented["class"] = 2
    augmented["followingId"] = 0
    return augmented[
        [
            "id",
            "frame",
            "x",
            "y",
            "width",
            "length",
            "class",
            "precedingId",
            "followingId",
            "leftPrecedingId",
            "leftAlongsideId",
            "leftFollowingId",
            "rightPrecedingId",
            "rightAlongsideId",
            "rightFollowingId",
            "laneId",
            "xVelocity",
            "yVelocity",
            "xAcceleration",
            "yAcceleration",
            "dhw",
            "thw",
            "ttc",
        ]
    ]


def _write_recording(path: Path, name: str, dataframe: pd.DataFrame) -> Path:
    file_path = path / name
    dataframe.to_csv(file_path, index=False)
    return file_path


def _make_lane_change_right_tracks() -> pd.DataFrame:
    tracks = make_synthetic_tracks()
    ego_mask = tracks["id"] == 1
    frames = tracks.loc[ego_mask, "frame"].to_numpy()
    y_velocity = np.zeros_like(frames, dtype=float)
    y_velocity[(frames >= 28) & (frames <= 32)] = -0.5
    tracks.loc[ego_mask, "yVelocity"] = y_velocity
    tracks.loc[ego_mask & (tracks["frame"] >= 30), "laneId"] = 3
    return tracks


def test_convergence_analyzer_detects_shifts(tmp_path: Path):
    directory = tmp_path / "recordings"
    directory.mkdir()

    first = to_highd_tracks(make_synthetic_tracks())
    second = to_highd_tracks(_make_lane_change_right_tracks())

    _write_recording(directory, "01_tracks.csv", first)
    _write_recording(directory, "02_tracks.csv", second)

    analyzer = ScenarioConvergenceAnalyzer(frame_rate=25.0, grid_size=32, parameter_grid_points=64)
    thresholds = DistanceTriple(mise=1.0, kl_divergence=1.0, hellinger=1.0)

    summary = analyzer.analyze(directory, thresholds=thresholds)

    assert len(summary.steps) == 2
    assert summary.steps[0].frequency_shift is None

    step_two = summary.steps[1]
    assert step_two.frequency_shift is not None
    assert step_two.max_parameter_distance is not None
    assert step_two.max_parameter_distance.mise >= 0.0

    steps_df = convergence_steps_to_dataframe(summary)
    assert not steps_df.empty
    assert "frequency_mise" in steps_df.columns


def test_convergence_cli_creates_outputs(tmp_path: Path, capsys):
    directory = tmp_path / "recordings"
    directory.mkdir()

    first = to_highd_tracks(make_synthetic_tracks())
    second = to_highd_tracks(_make_lane_change_right_tracks())

    _write_recording(directory, "01_tracks.csv", first)
    _write_recording(directory, "02_tracks.csv", second)

    output_dir = tmp_path / "out"

    convergence_main(
        [
            "--tracks",
            str(directory),
            "--output-dir",
            str(output_dir),
            "--grid-size",
            "32",
            "--parameter-grid",
            "64",
        ]
    )

    captured = capsys.readouterr()
    assert "Processed steps" in captured.out

    steps_path = output_dir / "convergence_steps.csv"
    params_path = output_dir / "parameter_shifts.csv"
    summary_path = output_dir / "convergence_summary.json"

    assert steps_path.exists()
    assert summary_path.exists()
    # parameter shifts may be empty if no overlapping parameters; check file optional
    if params_path.exists():
        params_df = pd.read_csv(params_path)
        assert {"scenario", "parameter"}.issubset(params_df.columns)
