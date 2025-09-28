"""Integration tests for HighD scenario detection and statistics."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("plotly")

from scenario_parameter_collection import statistics as stats_module
from scenario_parameter_collection.cli import main as cli_main
from scenario_parameter_collection.coverage import compute_erwin_coverage
from scenario_parameter_collection.detection import HighDScenarioDetector, ScenarioEvent
from scenario_parameter_collection.highd_loader import load_tracks
from scenario_parameter_collection.statistics import estimate_parameter_distributions
from scenario_parameter_collection.visualization import generate_report


def make_synthetic_tracks() -> pd.DataFrame:
    """Build a compact set of tracks triggering multiple scenarios."""

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
    """Add HighD specific columns so the CSV looks like the real dataset."""

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


def write_highd_csv(tmp_path: Path) -> Path:
    data = to_highd_tracks(make_synthetic_tracks())
    file_path = tmp_path / "01_tracks.csv"
    data.to_csv(file_path, index=False)
    return file_path


def test_detector_finds_car_following_and_lane_change():
    tracks = make_synthetic_tracks()
    detector = HighDScenarioDetector(frame_rate=25.0)
    events = detector.detect(tracks)
    scenarios = {event.scenario for event in events}
    assert "car_following" in scenarios
    assert "ego_lane_change_left" in scenarios

    stats = estimate_parameter_distributions(events)
    assert stats.counts["car_following"] >= 1
    assert "mean_thw" in stats.parameter_distributions["car_following"]

    coverage = compute_erwin_coverage(events, frame_rate=25.0)
    assert coverage.total_events == len(events)
    assert coverage.mapped_events >= 1


def test_statistics_fallback_without_scipy(monkeypatch):
    tracks = make_synthetic_tracks()
    detector = HighDScenarioDetector(frame_rate=25.0)
    events = detector.detect(tracks)

    monkeypatch.setattr(stats_module, "gaussian_kde", None, raising=False)
    stats = stats_module.estimate_parameter_distributions(events)

    assert stats.counts["car_following"] >= 1
    assert "mean_thw" in stats.parameter_distributions["car_following"]


def test_highd_loader_handles_csv_directory(tmp_path: Path):
    write_highd_csv(tmp_path)
    loaded = load_tracks(tmp_path)
    assert set(loaded["source_file"].unique()) == {"01_tracks.csv"}
    assert set(loaded["recording_id"].unique()) == {"01"}

    detector = HighDScenarioDetector(frame_rate=25.0)
    events = detector.detect(loaded)
    assert any(event.scenario == "car_following" for event in events)


def test_cli_generates_outputs(tmp_path: Path, capsys, monkeypatch):
    directory = tmp_path / "recording"
    directory.mkdir()
    write_highd_csv(directory)

    output_dir = tmp_path / "outputs"
    monkeypatch.setattr(stats_module, "gaussian_kde", None, raising=False)

    cli_main(
        [
            "--tracks",
            str(directory),
            "--output-dir",
            str(output_dir),
            "--grid-size",
            "32",
            "--frame-rate",
            "25.0",
        ]
    )

    captured = capsys.readouterr()
    assert "Scenario frequencies" in captured.out

    events_path = output_dir / "scenario_events.csv"
    counts_path = output_dir / "scenario_counts.csv"
    distributions_path = output_dir / "parameter_distributions.json"
    erwin_counts_path = output_dir / "erwin_coverage.csv"
    erwin_summary_path = output_dir / "erwin_coverage_summary.json"
    unmapped_path = output_dir / "unmapped_events.csv"

    assert events_path.exists()
    assert counts_path.exists()
    assert distributions_path.exists()
    assert erwin_counts_path.exists()
    assert erwin_summary_path.exists()
    assert unmapped_path.exists()


def test_erwin_coverage_reports_unmapped():
    events = [
        ScenarioEvent(
            scenario="car_following",
            track_id=1,
            start_frame=0,
            end_frame=24,
            parameters={},
        ),
        ScenarioEvent(
            scenario="free_driving",
            track_id=1,
            start_frame=100,
            end_frame=124,
            parameters={},
        ),
    ]
    coverage = compute_erwin_coverage(events, frame_rate=25.0)
    assert coverage.total_events == 2
    assert coverage.mapped_events == 1
    assert coverage.coverage_ratio() == pytest.approx(0.5)
    assert len(coverage.unmatched_events) == 1
    assert coverage.unmatched_events[0].scenario == "free_driving"


def test_visualization_report_creation(tmp_path: Path, monkeypatch):
    directory = tmp_path / "recording"
    directory.mkdir()
    write_highd_csv(directory)

    output_dir = tmp_path / "outputs"
    monkeypatch.setattr(stats_module, "gaussian_kde", None, raising=False)

    cli_main(
        [
            "--tracks",
            str(directory),
            "--output-dir",
            str(output_dir),
            "--grid-size",
            "32",
            "--frame-rate",
            "25.0",
        ]
    )

    report_path = output_dir / "report.html"
    result_path = generate_report(output_dir, html_path=report_path, title="Synthetic Report")

    assert result_path == report_path
    assert report_path.exists()

    html = report_path.read_text(encoding="utf-8")
    assert "Synthetic Report" in html
    assert "Scenario Frequency Overview" in html
    assert "car_following" in html
