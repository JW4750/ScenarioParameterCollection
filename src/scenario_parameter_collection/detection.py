"""Scenario detection for the HighD dataset."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping

import numpy as np
import pandas as pd

from .utils import find_boolean_segments
from .tagging import assign_action_tags, summarise_tags, TAG_COLUMNS


@dataclass
class ScenarioEvent:
    """Detected scenario event for a specific track segment."""

    scenario: str
    track_id: int
    start_frame: int
    end_frame: int
    parameters: Dict[str, float] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)

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
        lead_acceleration_threshold: float = 1.5,
        slow_speed_threshold: float = 8.0,
        stationary_lead_speed: float = 2.0,
    ) -> None:
        self.frame_rate = frame_rate
        self.min_free_speed = min_free_speed
        self.free_gap = free_gap
        self.following_rel_speed = following_rel_speed
        self.braking_threshold = braking_threshold
        self.lead_braking_threshold = lead_braking_threshold
        self.lead_acceleration_threshold = lead_acceleration_threshold
        self.slow_speed_threshold = slow_speed_threshold
        self.stationary_lead_speed = stationary_lead_speed

    # ------------------------------------------------------------------
    # public API
    def detect(self, tracks: pd.DataFrame) -> List[ScenarioEvent]:
        """Detect scenarios in the provided HighD tracks dataframe."""

        self._validate_columns(tracks)
        prepared = self._prepare_dataframe(tracks)
        track_map: Dict[int, pd.DataFrame] = {}
        for track_id, track_df in prepared.groupby("id"):
            track_map[int(track_id)] = track_df.sort_values("frame").reset_index(drop=True)

        events: List[ScenarioEvent] = []
        for track_id, track_df in track_map.items():
            events.extend(
                self._detect_for_track(
                    track_id=int(track_id),
                    track=track_df,
                    all_tracks=track_map,
                )
            )
        return events

    # ------------------------------------------------------------------
    # dataframe preparation
    def _validate_columns(self, tracks: pd.DataFrame) -> None:
        missing = [col for col in self.required_columns if col not in tracks.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

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
        lead_info = (
            df[["id", "frame", "xVelocity", "xAcceleration", "yVelocity"]]
            .rename(columns=lead_cols)
            .drop_duplicates(subset=["precedingId", "frame"], keep="last")
        )
        df = df.merge(lead_info, on=["precedingId", "frame"], how="left")
        df["relative_speed"] = df["xVelocity"] - df["preceding_xVelocity"]
        df["relative_acceleration"] = df["xAcceleration"] - df["preceding_xAcceleration"]
        tags = assign_action_tags(df, frame_rate=self.frame_rate)
        for column in TAG_COLUMNS:
            df[column] = tags[column]
        return df

    # ------------------------------------------------------------------
    # track-level detection
    def _detect_for_track(
        self,
        track_id: int,
        track: pd.DataFrame,
        all_tracks: Mapping[int, pd.DataFrame],
    ) -> List[ScenarioEvent]:
        events: List[ScenarioEvent] = []
        events.extend(self._detect_free_driving(track_id, track))
        events.extend(self._detect_car_following(track_id, track))
        events.extend(self._detect_lead_braking(track_id, track))
        events.extend(self._detect_lead_accelerating(track_id, track))
        events.extend(self._detect_ego_braking(track_id, track))
        events.extend(self._detect_cut_in(track_id, track, direction="left"))
        events.extend(self._detect_cut_in(track_id, track, direction="right"))
        events.extend(self._detect_cut_out(track_id, track, direction="left"))
        events.extend(self._detect_cut_out(track_id, track, direction="right"))
        events.extend(self._detect_lane_change(track_id, track, direction="left"))
        events.extend(self._detect_lane_change(track_id, track, direction="right"))
        events.extend(self._detect_slow_traffic(track_id, track))
        events.extend(self._detect_stationary_lead(track_id, track))
        events.extend(self._detect_merge(track_id, track))
        events.extend(self._detect_overtaking(track_id, track))
        events.extend(self._detect_overtaken(track_id, track, all_tracks))
        return events

    # ------------------------------------------------------------------
    def _summarise_tags(self, window: pd.DataFrame) -> Dict[str, str]:
        if window.empty:
            return {}
        return summarise_tags(window)

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
                    tags=self._summarise_tags(window),
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
                    tags=self._summarise_tags(window),
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
                    tags=self._summarise_tags(window),
                )
            )
        return events

    def _detect_lead_accelerating(self, track_id: int, track: pd.DataFrame) -> List[ScenarioEvent]:
        mask = (
            (track["precedingId"] > 0)
            & (track["preceding_xAcceleration"] >= self.lead_acceleration_threshold)
            & (track["relative_speed"] <= 0.0)
        )
        min_length = max(1, int(self.frame_rate * 0.6))
        segments = find_boolean_segments(track["frame"].tolist(), mask.tolist(), min_length)
        events: List[ScenarioEvent] = []
        for seg in segments:
            window = track[(track["frame"] >= seg.start_frame) & (track["frame"] <= seg.end_frame)]
            params = {
                "max_lead_acc": float(window["preceding_xAcceleration"].max()),
                "mean_relative_speed": float(window["relative_speed"].mean()),
                "duration_s": seg.length / self.frame_rate,
            }
            events.append(
                ScenarioEvent(
                    scenario="lead_vehicle_accelerating",
                    track_id=track_id,
                    start_frame=int(seg.start_frame),
                    end_frame=int(seg.end_frame),
                    parameters=params,
                    tags=self._summarise_tags(window),
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
                    tags=self._summarise_tags(window),
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
            window = track[
                (track["frame"] >= start_frame) & (track["frame"] <= end_frame)
            ]
            events.append(
                ScenarioEvent(
                    scenario=scenario_name,
                    track_id=track_id,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    parameters=params,
                    tags=self._summarise_tags(window),
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
            window = track[
                (track["frame"] >= start_frame) & (track["frame"] <= end_frame)
            ]
            events.append(
                ScenarioEvent(
                    scenario=scenario_name,
                    track_id=track_id,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    parameters=params,
                    tags=self._summarise_tags(window),
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
                    tags=self._summarise_tags(window),
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
                    tags=self._summarise_tags(window),
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
                    tags=self._summarise_tags(window),
                )
            )
        return events

    def _detect_merge(self, track_id: int, track: pd.DataFrame) -> List[ScenarioEvent]:
        events: List[ScenarioEvent] = []
        lanes = track["laneId"].to_numpy()
        frames = track["frame"].to_numpy()
        half_window = int(self.frame_rate)
        last_frame = int(frames[-1]) if len(frames) else 0
        first_frame = int(frames[0]) if len(frames) else 0
        for idx in range(1, len(track)):
            prev_lane = lanes[idx - 1]
            curr_lane = lanes[idx]
            if prev_lane <= 0 < curr_lane:
                direction = "right" if curr_lane > prev_lane else "left"
                trailing_col = (
                    "rightFollowingId" if direction == "right" else "leftFollowingId"
                )
                trailing_id = track.iloc[idx].get(trailing_col, 0)
                if pd.isna(trailing_id) or int(trailing_id) <= 0:
                    continue
                start_frame = int(max(first_frame, frames[idx] - half_window))
                end_frame = int(min(last_frame, frames[idx] + half_window))
                window = track[
                    (track["frame"] >= start_frame) & (track["frame"] <= end_frame)
                ]
                params = {
                    "merge_direction": direction,
                    "trailing_vehicle_id": int(trailing_id),
                    "duration_s": (end_frame - start_frame + 1) / self.frame_rate,
                }
                events.append(
                    ScenarioEvent(
                        scenario="ego_merge_with_trailing_vehicle",
                        track_id=track_id,
                        start_frame=start_frame,
                        end_frame=end_frame,
                        parameters=params,
                        tags=self._summarise_tags(window),
                    )
                )
        return events

    def _detect_overtaking(self, track_id: int, track: pd.DataFrame) -> List[ScenarioEvent]:
        events: List[ScenarioEvent] = []
        if track.empty:
            return events
        lanes = track["laneId"].to_numpy()
        frames = track["frame"].to_numpy()
        last_frame = int(frames[-1])
        first_frame = int(frames[0])
        half_window = int(self.frame_rate)

        lane_changes: List[tuple[int, float, float]] = []
        for idx in range(1, len(track)):
            prev_lane = lanes[idx - 1]
            curr_lane = lanes[idx]
            if curr_lane != prev_lane:
                lane_changes.append((idx, prev_lane, curr_lane))

        for i, (idx, prev_lane, curr_lane) in enumerate(lane_changes):
            if curr_lane >= prev_lane:
                continue
            for j in range(i + 1, len(lane_changes)):
                idx2, prev_lane2, curr_lane2 = lane_changes[j]
                if prev_lane2 == curr_lane and curr_lane2 == prev_lane:
                    start_frame = int(max(first_frame, frames[idx] - half_window))
                    end_frame = int(min(last_frame, frames[idx2] + half_window))
                    window = track[
                        (track["frame"] >= start_frame)
                        & (track["frame"] <= end_frame)
                    ]
                    params = {
                        "initial_gap": float(track.iloc[max(idx - 1, 0)]["dhw"]),
                        "completion_gap": float(track.iloc[min(idx2, len(track) - 1)]["dhw"]),
                        "overtake_duration_s": (end_frame - start_frame + 1)
                        / self.frame_rate,
                    }
                    events.append(
                        ScenarioEvent(
                            scenario="ego_overtaking",
                            track_id=track_id,
                            start_frame=start_frame,
                            end_frame=end_frame,
                            parameters=params,
                            tags=self._summarise_tags(window),
                        )
                    )
                    break
        return events

    def _detect_overtaken(
        self,
        track_id: int,
        track: pd.DataFrame,
        all_tracks: Mapping[int, pd.DataFrame],
    ) -> List[ScenarioEvent]:
        events: List[ScenarioEvent] = []
        if track.empty:
            return events
        frames = track["frame"].to_numpy()
        last_frame = int(frames[-1])
        first_frame = int(frames[0])
        for side in ("left", "right"):
            following_col = f"{side}FollowingId"
            alongside_col = f"{side}AlongsideId"
            preceding_col = f"{side}PrecedingId"
            candidates = (
                pd.concat(
                    [
                        track[following_col],
                        track[alongside_col],
                        track[preceding_col],
                    ]
                )
                .dropna()
                .astype(int)
            )
            for vehicle_id in sorted(set(candidates)):
                if vehicle_id <= 0:
                    continue
                state = "none"
                start_idx: int | None = None
                end_idx: int | None = None
                for idx in range(len(track)):
                    row = track.iloc[idx]
                    current_state = "none"
                    if row.get(following_col) == vehicle_id:
                        current_state = "following"
                    elif row.get(alongside_col) == vehicle_id:
                        current_state = "alongside"
                    elif row.get(preceding_col) == vehicle_id:
                        current_state = "preceding"

                    if state == "none":
                        if current_state == "following":
                            state = "following"
                            start_idx = idx
                    elif state == "following":
                        if current_state == "alongside":
                            state = "alongside"
                        elif current_state == "following":
                            continue
                        else:
                            state = "none"
                            start_idx = None
                    elif state == "alongside":
                        if current_state == "preceding":
                            end_idx = idx
                            break
                        elif current_state == "alongside":
                            continue
                        else:
                            state = "none"
                            start_idx = None

                if start_idx is None or end_idx is None:
                    continue
                start_frame = int(max(first_frame, frames[max(start_idx - 1, 0)]))
                end_frame = int(min(last_frame, frames[min(end_idx + 1, len(track) - 1)]))
                window = track[
                    (track["frame"] >= start_frame) & (track["frame"] <= end_frame)
                ]
                other_track = all_tracks.get(int(vehicle_id))
                overtaker_speed = float("nan")
                if other_track is not None and not other_track.empty:
                    mask = (other_track["frame"] >= start_frame) & (
                        other_track["frame"] <= end_frame
                    )
                    if mask.any():
                        overtaker_speed = float(other_track.loc[mask, "xVelocity"].mean())
                params = {
                    "overtaker_id": int(vehicle_id),
                    "side": side,
                    "overtaker_speed": overtaker_speed,
                    "ego_speed": float(window["xVelocity"].mean()),
                    "event_duration_s": (end_frame - start_frame + 1) / self.frame_rate,
                }
                events.append(
                    ScenarioEvent(
                        scenario="ego_overtaken_by_vehicle",
                        track_id=track_id,
                        start_frame=start_frame,
                        end_frame=end_frame,
                        parameters=params,
                        tags=self._summarise_tags(window),
                    )
                )
        return events
