"""Scenario detection for the HighD dataset."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from .utils import find_boolean_segments


@dataclass
class ScenarioEvent:
    """Detected scenario event for a specific track segment."""

    scenario: str
    track_id: int
    start_frame: int
    end_frame: int
    parameters: Dict[str, float] = field(default_factory=dict)

    def duration(self, frame_rate: float) -> float:
        """Return the event duration in seconds for the given frame rate."""

        return (self.end_frame - self.start_frame + 1) / frame_rate


class HighDScenarioDetector:
    """Rule-based scenario detector tailored to HighD trajectory data."""

    required_columns: Iterable[str] = (
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
    )

    def __init__(
        self,
        frame_rate: float = 25.0,
        min_free_speed: float = 20.0,
        free_gap: float = 120.0,
        following_rel_speed: float = 3.0,
        braking_threshold: float = -3.0,
        lead_braking_threshold: float = -2.5,
        slow_speed_threshold: float = 8.0,
        stationary_lead_speed: float = 2.0,
    ) -> None:
        self.frame_rate = frame_rate
        self.min_free_speed = min_free_speed
        self.free_gap = free_gap
        self.following_rel_speed = following_rel_speed
        self.braking_threshold = braking_threshold
        self.lead_braking_threshold = lead_braking_threshold
        self.slow_speed_threshold = slow_speed_threshold
        self.stationary_lead_speed = stationary_lead_speed

    # ------------------------------------------------------------------
    # public API
    def detect(self, tracks: pd.DataFrame) -> List[ScenarioEvent]:
        """Detect scenarios in the provided HighD tracks dataframe."""

        self._validate_columns(tracks)
        prepared = self._prepare_dataframe(tracks)
        events: List[ScenarioEvent] = []
        group_columns = self._groupby_columns(prepared)
        for _, track_df in prepared.groupby(group_columns, sort=False):
            sorted_track = track_df.sort_values("frame").reset_index(drop=True)
            track_id = int(track_df["id"].iloc[0])
            events.extend(self._detect_for_track(track_id=track_id, track=sorted_track))
        return events

    # ------------------------------------------------------------------
    # dataframe preparation
    def _validate_columns(self, tracks: pd.DataFrame) -> None:
        missing = [col for col in self.required_columns if col not in tracks.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def _groupby_columns(self, df: pd.DataFrame) -> List[str]:
        """Return the dataframe columns used to isolate individual tracks."""

        discriminators = [
            column for column in ("recording_id", "source_file") if column in df.columns
        ]
        if discriminators:
            return discriminators + ["id"]
        return ["id"]

    def _prepare_dataframe(self, tracks: pd.DataFrame) -> pd.DataFrame:
        df = tracks.copy()
        # normalise NaN encodings for THW/TTC
        df["thw"] = df["thw"].replace({0: np.nan, -1: np.nan})
        df["ttc"] = df["ttc"].replace({0: np.nan, -1: np.nan})
        # ensure floats
        numeric_cols = [
            "xVelocity",
            "yVelocity",
            "xAcceleration",
            "yAcceleration",
            "dhw",
            "thw",
            "ttc",
        ]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
        # add preceding vehicle information
        lead_cols = {
            "id": "precedingId",
            "xVelocity": "preceding_xVelocity",
            "xAcceleration": "preceding_xAcceleration",
            "yVelocity": "preceding_yVelocity",
        }
        discriminator_cols = [
            column for column in ("recording_id", "source_file") if column in df.columns
        ]
        join_keys = ["precedingId", "frame", *discriminator_cols]
        lead_info = (
            df[
                [
                    "id",
                    "frame",
                    *discriminator_cols,
                    "xVelocity",
                    "xAcceleration",
                    "yVelocity",
                ]
            ]
            .rename(columns=lead_cols)
            .drop_duplicates(subset=join_keys, keep="last")
        )
        df = df.merge(lead_info, on=join_keys, how="left")
        df["relative_speed"] = df["xVelocity"] - df["preceding_xVelocity"]
        df["relative_acceleration"] = df["xAcceleration"] - df["preceding_xAcceleration"]
        return df

    # ------------------------------------------------------------------
    # track-level detection
    def _detect_for_track(self, track_id: int, track: pd.DataFrame) -> List[ScenarioEvent]:
        events: List[ScenarioEvent] = []
        events.extend(self._detect_free_driving(track_id, track))
        events.extend(self._detect_car_following(track_id, track))
        events.extend(self._detect_lead_braking(track_id, track))
        events.extend(self._detect_ego_braking(track_id, track))
        events.extend(self._detect_cut_in(track_id, track, direction="left"))
        events.extend(self._detect_cut_in(track_id, track, direction="right"))
        events.extend(self._detect_cut_out(track_id, track, direction="left"))
        events.extend(self._detect_cut_out(track_id, track, direction="right"))
        events.extend(self._detect_lane_change(track_id, track, direction="left"))
        events.extend(self._detect_lane_change(track_id, track, direction="right"))
        events.extend(self._detect_slow_traffic(track_id, track))
        events.extend(self._detect_stationary_lead(track_id, track))
        return events

    # ------------------------------------------------------------------
    # scenario specific detectors
    def _detect_free_driving(self, track_id: int, track: pd.DataFrame) -> List[ScenarioEvent]:
        mask = (
            (track["precedingId"] <= 0) | (track["dhw"] > self.free_gap)
        ) & (track["xVelocity"] >= self.min_free_speed)
        min_length = int(self.frame_rate * 2)
        segments = find_boolean_segments(track["frame"].tolist(), mask.tolist(), min_length)
        events: List[ScenarioEvent] = []
        for seg in segments:
            window = track[(track["frame"] >= seg.start_frame) & (track["frame"] <= seg.end_frame)]
            params = {
                "speed": float(window["xVelocity"].mean()),
                "acceleration": float(window["xAcceleration"].mean()),
                "duration_s": seg.length / self.frame_rate,
            }
            events.append(
                ScenarioEvent(
                    scenario="free_driving",
                    track_id=track_id,
                    start_frame=int(seg.start_frame),
                    end_frame=int(seg.end_frame),
                    parameters=params,
                )
            )
        return events

    def _detect_car_following(self, track_id: int, track: pd.DataFrame) -> List[ScenarioEvent]:
        mask = (
            (track["precedingId"] > 0)
            & (track["thw"].between(0.7, 3.0))
            & (track["ttc"].fillna(np.inf) > 3.0)
            & (track["relative_speed"].abs() <= self.following_rel_speed)
        )
        min_length = int(self.frame_rate * 2)
        segments = find_boolean_segments(track["frame"].tolist(), mask.tolist(), min_length)
        events: List[ScenarioEvent] = []
        for seg in segments:
            window = track[(track["frame"] >= seg.start_frame) & (track["frame"] <= seg.end_frame)]
            # enforce lane stability
            if window["laneId"].nunique(dropna=True) > 1:
                continue
            params = {
                "mean_thw": float(window["thw"].mean()),
                "mean_dhw": float(window["dhw"].mean()),
                "mean_relative_speed": float(window["relative_speed"].mean()),
                "duration_s": seg.length / self.frame_rate,
            }
            events.append(
                ScenarioEvent(
                    scenario="car_following",
                    track_id=track_id,
                    start_frame=int(seg.start_frame),
                    end_frame=int(seg.end_frame),
                    parameters=params,
                )
            )
        return events

    def _detect_lead_braking(self, track_id: int, track: pd.DataFrame) -> List[ScenarioEvent]:
        mask = (
            (track["precedingId"] > 0)
            & (track["preceding_xAcceleration"] <= self.lead_braking_threshold)
            & (track["thw"].fillna(np.inf) < 3.5)
        )
        min_length = max(1, int(self.frame_rate * 0.6))
        segments = find_boolean_segments(track["frame"].tolist(), mask.tolist(), min_length)
        events: List[ScenarioEvent] = []
        for seg in segments:
            window = track[(track["frame"] >= seg.start_frame) & (track["frame"] <= seg.end_frame)]
            params = {
                "min_lead_acc": float(window["preceding_xAcceleration"].min()),
                "min_ttc": float(window["ttc"].min()),
                "min_thw": float(window["thw"].min()),
                "duration_s": seg.length / self.frame_rate,
            }
            events.append(
                ScenarioEvent(
                    scenario="lead_vehicle_braking",
                    track_id=track_id,
                    start_frame=int(seg.start_frame),
                    end_frame=int(seg.end_frame),
                    parameters=params,
                )
            )
        return events

    def _detect_ego_braking(self, track_id: int, track: pd.DataFrame) -> List[ScenarioEvent]:
        mask = track["xAcceleration"] <= self.braking_threshold
        min_length = max(1, int(self.frame_rate * 0.6))
        segments = find_boolean_segments(track["frame"].tolist(), mask.tolist(), min_length)
        events: List[ScenarioEvent] = []
        for seg in segments:
            window = track[(track["frame"] >= seg.start_frame) & (track["frame"] <= seg.end_frame)]
            speeds = window["xVelocity"].dropna()
            if speeds.empty:
                continue
            speed_drop = float(speeds.iloc[0] - speeds.iloc[-1])
            if speed_drop < 1.0:
                continue
            params = {
                "min_acc": float(window["xAcceleration"].min()),
                "speed_drop": speed_drop,
                "duration_s": seg.length / self.frame_rate,
            }
            events.append(
                ScenarioEvent(
                    scenario="ego_braking",
                    track_id=track_id,
                    start_frame=int(seg.start_frame),
                    end_frame=int(seg.end_frame),
                    parameters=params,
                )
            )
        return events

    def _detect_cut_in(self, track_id: int, track: pd.DataFrame, direction: str) -> List[ScenarioEvent]:
        neighbour_cols = (
            ["leftPrecedingId", "leftAlongsideId", "leftFollowingId"]
            if direction == "left"
            else ["rightPrecedingId", "rightAlongsideId", "rightFollowingId"]
        )
        scenario_name = "cut_in_from_left" if direction == "left" else "cut_in_from_right"
        current_lead = track["precedingId"]
        previous_lead = current_lead.shift(1)
        mask = (current_lead > 0) & (current_lead != previous_lead)
        events: List[ScenarioEvent] = []
        half_window = int(self.frame_rate * 0.5)
        for idx in np.where(mask)[0]:
            new_lead = current_lead.iloc[idx]
            row = track.iloc[idx]
            if row.get("laneId") != track.iloc[max(idx - 1, 0)]["laneId"]:
                # lane change events are handled separately
                continue
            neighbour_match = False
            for col in neighbour_cols:
                value = row.get(col)
                if pd.notna(value) and value == new_lead:
                    neighbour_match = True
                    break
            if not neighbour_match:
                continue
            frame = int(row["frame"])
            start_frame = int(max(track.iloc[0]["frame"], frame - half_window))
            end_frame = int(min(track.iloc[len(track) - 1]["frame"], frame + half_window))
            params = {
                "gap_post_cut": float(row["dhw"]),
                "relative_speed_post": float(row["relative_speed"]),
                "ttc_post": float(row["ttc"] if not pd.isna(row["ttc"]) else np.nan),
            }
            events.append(
                ScenarioEvent(
                    scenario=scenario_name,
                    track_id=track_id,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    parameters=params,
                )
            )
        return events

    def _detect_cut_out(self, track_id: int, track: pd.DataFrame, direction: str) -> List[ScenarioEvent]:
        neighbour_cols = (
            ["leftPrecedingId", "leftAlongsideId", "leftFollowingId"]
            if direction == "left"
            else ["rightPrecedingId", "rightAlongsideId", "rightFollowingId"]
        )
        scenario_name = "cut_out_to_left" if direction == "left" else "cut_out_to_right"
        current_lead = track["precedingId"]
        next_lead = current_lead.shift(-1)
        mask = (current_lead > 0) & (current_lead != next_lead)
        events: List[ScenarioEvent] = []
        half_window = int(self.frame_rate * 0.5)
        for idx in np.where(mask)[0]:
            old_lead = current_lead.iloc[idx]
            if old_lead <= 0:
                continue
            row_next = track.iloc[min(idx + 1, len(track) - 1)]
            neighbour_match = False
            for col in neighbour_cols:
                value = row_next.get(col)
                if pd.notna(value) and value == old_lead:
                    neighbour_match = True
                    break
            if not neighbour_match:
                continue
            frame = int(track.iloc[idx]["frame"])
            start_frame = int(max(track.iloc[0]["frame"], frame - half_window))
            end_frame = int(min(track.iloc[len(track) - 1]["frame"], frame + half_window))
            params = {
                "gap_before_cut": float(track.iloc[idx]["dhw"]),
                "relative_speed_before": float(track.iloc[idx]["relative_speed"]),
            }
            events.append(
                ScenarioEvent(
                    scenario=scenario_name,
                    track_id=track_id,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    parameters=params,
                )
            )
        return events

    def _detect_lane_change(self, track_id: int, track: pd.DataFrame, direction: str) -> List[ScenarioEvent]:
        lane_series = track["laneId"]
        lane_diff = lane_series.diff()
        if direction == "left":
            change_mask = lane_diff < 0
            scenario_name = "ego_lane_change_left"
        else:
            change_mask = lane_diff > 0
            scenario_name = "ego_lane_change_right"
        events: List[ScenarioEvent] = []
        half_window = int(self.frame_rate * 0.6)
        for idx in np.where(change_mask)[0]:
            frame = int(track.iloc[idx]["frame"])
            start_frame = int(max(track.iloc[0]["frame"], frame - half_window))
            end_frame = int(min(track.iloc[len(track) - 1]["frame"], frame + half_window))
            window = track[(track["frame"] >= start_frame) & (track["frame"] <= end_frame)]
            params = {
                "duration_s": (end_frame - start_frame + 1) / self.frame_rate,
                "max_abs_y_velocity": float(window["yVelocity"].abs().max()),
                "speed_mean": float(window["xVelocity"].mean()),
            }
            events.append(
                ScenarioEvent(
                    scenario=scenario_name,
                    track_id=track_id,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    parameters=params,
                )
            )
        return events

    def _detect_slow_traffic(self, track_id: int, track: pd.DataFrame) -> List[ScenarioEvent]:
        mask = (
            (track["precedingId"] > 0)
            & (track["xVelocity"] <= self.slow_speed_threshold)
            & (track["thw"].fillna(np.inf) <= 2.0)
        )
        min_length = int(self.frame_rate * 3)
        segments = find_boolean_segments(track["frame"].tolist(), mask.tolist(), min_length)
        events: List[ScenarioEvent] = []
        for seg in segments:
            window = track[(track["frame"] >= seg.start_frame) & (track["frame"] <= seg.end_frame)]
            params = {
                "mean_speed": float(window["xVelocity"].mean()),
                "mean_thw": float(window["thw"].mean()),
                "duration_s": seg.length / self.frame_rate,
            }
            events.append(
                ScenarioEvent(
                    scenario="slow_traffic",
                    track_id=track_id,
                    start_frame=int(seg.start_frame),
                    end_frame=int(seg.end_frame),
                    parameters=params,
                )
            )
        return events

    def _detect_stationary_lead(self, track_id: int, track: pd.DataFrame) -> List[ScenarioEvent]:
        mask = (
            (track["precedingId"] > 0)
            & (track["preceding_xVelocity"].fillna(np.inf) <= self.stationary_lead_speed)
            & (track["ttc"].fillna(np.inf) <= 4.0)
        )
        min_length = max(1, int(self.frame_rate * 0.6))
        segments = find_boolean_segments(track["frame"].tolist(), mask.tolist(), min_length)
        events: List[ScenarioEvent] = []
        for seg in segments:
            window = track[(track["frame"] >= seg.start_frame) & (track["frame"] <= seg.end_frame)]
            params = {
                "min_ttc": float(window["ttc"].min()),
                "lead_speed": float(window["preceding_xVelocity"].mean()),
                "duration_s": seg.length / self.frame_rate,
            }
            events.append(
                ScenarioEvent(
                    scenario="stationary_lead",
                    track_id=track_id,
                    start_frame=int(seg.start_frame),
                    end_frame=int(seg.end_frame),
                    parameters=params,
                )
            )
        return events
