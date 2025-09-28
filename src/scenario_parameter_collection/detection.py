"""Scenario detection for the HighD dataset."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

from .utils import find_boolean_segments


@dataclass(frozen=True)
class TagRule:
    """Boolean rule that describes a scenario using tag combinations."""

    all_tags: Sequence[str] = ()
    any_tags: Sequence[str] = ()
    none_tags: Sequence[str] = ()
    min_duration_s: float = 0.0


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
        for track_id, track_df in prepared.groupby("id"):
            sorted_track = track_df.sort_values("frame").reset_index(drop=True)
            tag_df = self._build_tag_timeline(sorted_track)
            events.extend(
                self._detect_for_track(
                    track_id=int(track_id), track=sorted_track, tags=tag_df
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
        return df

    # ------------------------------------------------------------------
    # tagging
    def _build_tag_timeline(self, track: pd.DataFrame) -> pd.DataFrame:
        tags = pd.DataFrame(index=track.index.copy())
        tags["frame"] = track["frame"].astype(int)

        tags["has_lead"] = track["precedingId"] > 0
        tags["free_gap"] = (~tags["has_lead"]) | (track["dhw"] > self.free_gap)
        tags["high_speed"] = track["xVelocity"] >= self.min_free_speed

        thw = track["thw"]
        tags["car_following_gap"] = thw.between(0.7, 3.0, inclusive="both")
        tags["ttc_safe"] = track["ttc"].fillna(np.inf) > 3.0
        tags["relative_speed_small"] = track["relative_speed"].abs() <= self.following_rel_speed

        tags["lead_braking"] = (
            track["preceding_xAcceleration"].fillna(np.inf) <= self.lead_braking_threshold
        )
        tags["ego_braking"] = track["xAcceleration"] <= self.braking_threshold
        tags["slow_speed"] = track["xVelocity"] <= self.slow_speed_threshold
        tags["stationary_lead"] = tags["has_lead"] & (
            track["preceding_xVelocity"].fillna(np.inf) <= self.stationary_lead_speed
        )

        accel = track["xAcceleration"].fillna(0.0)
        tags["accelerating"] = accel >= 0.3
        tags["decelerating_light"] = (accel < 0.3) & (accel > self.braking_threshold)
        tags["strong_brake"] = accel <= self.braking_threshold
        tags["cruising"] = (~tags["accelerating"]) & (~tags["strong_brake"])

        tags["lane_change_left"] = False
        tags["lane_change_right"] = False
        tags["lane_stable"] = True

        lane_diff = track["laneId"].diff()
        half_window_frames = int(self.frame_rate * 0.6)

        def _mark_window(frame: int, column: str) -> None:
            start_frame = int(max(track.iloc[0]["frame"], frame - half_window_frames))
            end_frame = int(min(track.iloc[len(track) - 1]["frame"], frame + half_window_frames))
            mask = (track["frame"] >= start_frame) & (track["frame"] <= end_frame)
            tags.loc[mask, column] = True
            if column in ("lane_change_left", "lane_change_right"):
                tags.loc[mask, "lane_stable"] = False

        for idx in np.where(lane_diff < 0)[0]:
            frame = int(track.iloc[idx]["frame"])
            _mark_window(frame, "lane_change_left")
        for idx in np.where(lane_diff > 0)[0]:
            frame = int(track.iloc[idx]["frame"])
            _mark_window(frame, "lane_change_right")

        tags["cut_in_from_left"] = False
        tags["cut_in_from_right"] = False
        tags["cut_out_to_left"] = False
        tags["cut_out_to_right"] = False

        current_lead = track["precedingId"]
        previous_lead = current_lead.shift(1)
        next_lead = current_lead.shift(-1)
        half_cut_window = int(self.frame_rate * 0.5)

        def _mark_cut_window(frame: int, column: str) -> None:
            start_frame = int(max(track.iloc[0]["frame"], frame - half_cut_window))
            end_frame = int(min(track.iloc[len(track) - 1]["frame"], frame + half_cut_window))
            mask = (track["frame"] >= start_frame) & (track["frame"] <= end_frame)
            tags.loc[mask, column] = True

        # cut-ins: new lead appears from neighbour lane
        for _, neighbour_cols, column in (
            (
                "left",
                ["leftPrecedingId", "leftAlongsideId", "leftFollowingId"],
                "cut_in_from_left",
            ),
            (
                "right",
                ["rightPrecedingId", "rightAlongsideId", "rightFollowingId"],
                "cut_in_from_right",
            ),
        ):
            mask = (current_lead > 0) & (current_lead != previous_lead)
            for idx in np.where(mask)[0]:
                row = track.iloc[idx]
                if row.get("laneId") != track.iloc[max(idx - 1, 0)]["laneId"]:
                    continue
                new_lead = current_lead.iloc[idx]
                neighbour_match = False
                for col in neighbour_cols:
                    value = row.get(col)
                    if pd.notna(value) and value == new_lead:
                        neighbour_match = True
                        break
                if not neighbour_match:
                    continue
                frame = int(row["frame"])
                _mark_cut_window(frame, column)

        # cut-outs: previous lead disappears to neighbour lane
        for _, neighbour_cols, column in (
            (
                "left",
                ["leftPrecedingId", "leftAlongsideId", "leftFollowingId"],
                "cut_out_to_left",
            ),
            (
                "right",
                ["rightPrecedingId", "rightAlongsideId", "rightFollowingId"],
                "cut_out_to_right",
            ),
        ):
            mask = (current_lead > 0) & (current_lead != next_lead)
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
                _mark_cut_window(frame, column)

        return tags

    # ------------------------------------------------------------------
    # track-level detection
    def _detect_for_track(
        self, track_id: int, track: pd.DataFrame, tags: pd.DataFrame
    ) -> List[ScenarioEvent]:
        events: List[ScenarioEvent] = []
        events.extend(self._detect_free_driving(track_id, track, tags))
        events.extend(self._detect_car_following(track_id, track, tags))
        events.extend(self._detect_lead_braking(track_id, track, tags))
        events.extend(self._detect_ego_braking(track_id, track, tags))
        events.extend(self._detect_cut_in(track_id, track, tags, direction="left"))
        events.extend(self._detect_cut_in(track_id, track, tags, direction="right"))
        events.extend(self._detect_cut_out(track_id, track, tags, direction="left"))
        events.extend(self._detect_cut_out(track_id, track, tags, direction="right"))
        events.extend(self._detect_lane_change(track_id, track, tags, direction="left"))
        events.extend(self._detect_lane_change(track_id, track, tags, direction="right"))
        events.extend(self._detect_slow_traffic(track_id, track, tags))
        events.extend(self._detect_stationary_lead(track_id, track, tags))
        return events

    def _evaluate_rule(self, tags: pd.DataFrame, rule: TagRule) -> pd.Series:
        mask = pd.Series(True, index=tags.index)
        if rule.all_tags:
            mask &= tags[list(rule.all_tags)].all(axis=1)
        if rule.any_tags:
            mask &= tags[list(rule.any_tags)].any(axis=1)
        for column in rule.none_tags:
            mask &= ~tags[column]
        return mask

    def _segments_from_rule(
        self, track: pd.DataFrame, mask: pd.Series, rule: TagRule
    ) -> List:
        min_length = 1
        if rule.min_duration_s > 0.0:
            min_length = max(1, int(self.frame_rate * rule.min_duration_s))
        frames = track["frame"].astype(int).tolist()
        return find_boolean_segments(frames, mask.tolist(), min_length)

    # ------------------------------------------------------------------
    # scenario specific detectors
    def _detect_free_driving(
        self, track_id: int, track: pd.DataFrame, tags: pd.DataFrame
    ) -> List[ScenarioEvent]:
        rule = TagRule(
            all_tags=("free_gap", "high_speed"),
            none_tags=("has_lead",),
            min_duration_s=2.0,
        )
        mask = self._evaluate_rule(tags, rule)
        segments = self._segments_from_rule(track, mask, rule)
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

    def _detect_car_following(
        self, track_id: int, track: pd.DataFrame, tags: pd.DataFrame
    ) -> List[ScenarioEvent]:
        rule = TagRule(
            all_tags=("has_lead", "car_following_gap", "ttc_safe", "relative_speed_small"),
            min_duration_s=2.0,
        )
        mask = self._evaluate_rule(tags, rule)
        segments = self._segments_from_rule(track, mask, rule)
        events: List[ScenarioEvent] = []
        for seg in segments:
            window = track[(track["frame"] >= seg.start_frame) & (track["frame"] <= seg.end_frame)]
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

    def _detect_lead_braking(
        self, track_id: int, track: pd.DataFrame, tags: pd.DataFrame
    ) -> List[ScenarioEvent]:
        rule = TagRule(
            all_tags=("has_lead", "lead_braking"),
            min_duration_s=0.6,
        )
        mask = self._evaluate_rule(tags, rule)
        mask &= track["thw"].fillna(np.inf) < 3.5
        segments = self._segments_from_rule(track, mask, rule)
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

    def _detect_ego_braking(
        self, track_id: int, track: pd.DataFrame, tags: pd.DataFrame
    ) -> List[ScenarioEvent]:
        rule = TagRule(all_tags=("ego_braking",), min_duration_s=0.6)
        mask = self._evaluate_rule(tags, rule)
        segments = self._segments_from_rule(track, mask, rule)
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

    def _detect_cut_in(
        self, track_id: int, track: pd.DataFrame, tags: pd.DataFrame, direction: str
    ) -> List[ScenarioEvent]:
        scenario_name = "cut_in_from_left" if direction == "left" else "cut_in_from_right"
        tag_column = "cut_in_from_left" if direction == "left" else "cut_in_from_right"
        mask = tags[tag_column]
        min_length = max(1, int(self.frame_rate * 0.5))
        segments = find_boolean_segments(track["frame"].tolist(), mask.tolist(), min_length)
        events: List[ScenarioEvent] = []
        for seg in segments:
            window = track[(track["frame"] >= seg.start_frame) & (track["frame"] <= seg.end_frame)]
            row = window.iloc[len(window) // 2]
            params = {
                "gap_post_cut": float(row["dhw"]),
                "relative_speed_post": float(row["relative_speed"]),
                "ttc_post": float(row["ttc"] if not pd.isna(row["ttc"]) else np.nan),
            }
            events.append(
                ScenarioEvent(
                    scenario=scenario_name,
                    track_id=track_id,
                    start_frame=int(seg.start_frame),
                    end_frame=int(seg.end_frame),
                    parameters=params,
                )
            )
        return events

    def _detect_cut_out(
        self, track_id: int, track: pd.DataFrame, tags: pd.DataFrame, direction: str
    ) -> List[ScenarioEvent]:
        scenario_name = "cut_out_to_left" if direction == "left" else "cut_out_to_right"
        tag_column = "cut_out_to_left" if direction == "left" else "cut_out_to_right"
        mask = tags[tag_column]
        min_length = max(1, int(self.frame_rate * 0.5))
        segments = find_boolean_segments(track["frame"].tolist(), mask.tolist(), min_length)
        events: List[ScenarioEvent] = []
        for seg in segments:
            window = track[(track["frame"] >= seg.start_frame) & (track["frame"] <= seg.end_frame)]
            row = window.iloc[len(window) // 2]
            params = {
                "gap_before_cut": float(row["dhw"]),
                "relative_speed_before": float(row["relative_speed"]),
            }
            events.append(
                ScenarioEvent(
                    scenario=scenario_name,
                    track_id=track_id,
                    start_frame=int(seg.start_frame),
                    end_frame=int(seg.end_frame),
                    parameters=params,
                )
            )
        return events

    def _detect_lane_change(
        self, track_id: int, track: pd.DataFrame, tags: pd.DataFrame, direction: str
    ) -> List[ScenarioEvent]:
        scenario_name = "ego_lane_change_left" if direction == "left" else "ego_lane_change_right"
        tag_column = "lane_change_left" if direction == "left" else "lane_change_right"
        mask = tags[tag_column]
        segments = self._segments_from_rule(
            track, mask, TagRule(min_duration_s=0.6)
        )
        events: List[ScenarioEvent] = []
        for seg in segments:
            window = track[(track["frame"] >= seg.start_frame) & (track["frame"] <= seg.end_frame)]
            params = {
                "duration_s": seg.length / self.frame_rate,
                "max_abs_y_velocity": float(window["yVelocity"].abs().max()),
                "speed_mean": float(window["xVelocity"].mean()),
            }
            events.append(
                ScenarioEvent(
                    scenario=scenario_name,
                    track_id=track_id,
                    start_frame=int(seg.start_frame),
                    end_frame=int(seg.end_frame),
                    parameters=params,
                )
            )
        return events

    def _detect_slow_traffic(
        self, track_id: int, track: pd.DataFrame, tags: pd.DataFrame
    ) -> List[ScenarioEvent]:
        rule = TagRule(
            all_tags=("has_lead", "slow_speed"),
            min_duration_s=3.0,
        )
        mask = self._evaluate_rule(tags, rule)
        mask &= track["thw"].fillna(np.inf) <= 2.0
        segments = self._segments_from_rule(track, mask, rule)
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

    def _detect_stationary_lead(
        self, track_id: int, track: pd.DataFrame, tags: pd.DataFrame
    ) -> List[ScenarioEvent]:
        rule = TagRule(all_tags=("stationary_lead",), min_duration_s=0.6)
        mask = self._evaluate_rule(tags, rule)
        mask &= track["ttc"].fillna(np.inf) <= 4.0
        segments = self._segments_from_rule(track, mask, rule)
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
