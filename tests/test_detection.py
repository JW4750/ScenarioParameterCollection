import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

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
