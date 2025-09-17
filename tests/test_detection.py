import subprocess
import sys

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("scipy")

from scenario_parameter_collection.detection import HighDScenarioDetector
from scenario_parameter_collection.statistics import estimate_parameter_distributions


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


def make_synthetic_recording(recording_id: str) -> pd.DataFrame:
    tracks = make_synthetic_tracks().copy()
    tracks["recording_id"] = recording_id
    tracks["source_file"] = f"{recording_id}_tracks.csv"
    return tracks


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


def test_multiple_recordings_do_not_mix_events(tmp_path):
    recording_a = make_synthetic_recording("recording_a")
    recording_b = make_synthetic_recording("recording_b")
    combined = pd.concat([recording_a, recording_b], ignore_index=True)

    detector = HighDScenarioDetector(frame_rate=25.0)
    events = detector.detect(combined)
    car_following_events = [event for event in events if event.scenario == "car_following"]
    assert len(car_following_events) == 2
    assert {event.track_id for event in car_following_events} == {1}
    assert {(event.start_frame, event.end_frame) for event in car_following_events} == {(0, 49)}

    # Write each recording to its own CSV file and run the CLI
    input_dir = tmp_path / "recordings"
    input_dir.mkdir()
    for df in (recording_a, recording_b):
        file_name = df["source_file"].iloc[0]
        df.to_csv(input_dir / file_name, index=False)

    output_dir = tmp_path / "outputs"
    cmd = [
        sys.executable,
        "-m",
        "scenario_parameter_collection.cli",
        "--tracks",
        str(input_dir),
        "--output-dir",
        str(output_dir),
        "--frame-rate",
        "25.0",
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)

    events_path = output_dir / "scenario_events.csv"
    cli_events = pd.read_csv(events_path)
    cli_car_following = cli_events[cli_events["scenario"] == "car_following"]
    assert len(cli_car_following) == 2
    assert set(cli_car_following["track_id"].astype(int)) == {1}
    assert set(zip(cli_car_following["start_frame"], cli_car_following["end_frame"])) == {(0, 49)}
