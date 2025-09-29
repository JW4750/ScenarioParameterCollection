"""Scenario detection for the HighD dataset using tag combinations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from .catalog import SCENARIO_DEFINITIONS
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


@dataclass(frozen=True)
class ScenarioPattern:
    """Definition of a scenario in terms of tag combinations."""

    name: str
    required_tags: Tuple[str, ...] = ()
    any_tags: Tuple[str, ...] = ()
    forbidden_tags: Tuple[str, ...] = ()
    min_duration_s: float = 1.0
    expansion_s: float = 0.0
    parameter_fn: Callable[[pd.DataFrame, float], Dict[str, float]] | None = None


@dataclass
class DetectionResult:
    """Container holding detected events and frame coverage statistics."""

    events: List[ScenarioEvent]
    unmatched_frames: pd.DataFrame
    total_frames: int
    hazard_events: List["HazardEvent"]
    unknown_hazard_events: List["HazardEvent"]
    unknown_hazard_frames: pd.DataFrame
    total_distance_m: float

    def covered_frames(self) -> int:
        return int(self.total_frames - len(self.unmatched_frames))

    def coverage_ratio(self) -> float:
        if self.total_frames == 0:
            return 0.0
        return self.covered_frames() / self.total_frames

    def scenario_counts(self) -> Dict[str, int]:
        """Return the number of occurrences per detected scenario."""

        counts: Dict[str, int] = {}
        for event in self.events:
            counts[event.scenario] = counts.get(event.scenario, 0) + 1
        return counts

    def kilometers_per_unknown_hazard(self) -> float | None:
        """Average kilometres travelled between unknown hazard events."""

        if not self.unknown_hazard_events:
            return None
        if self.total_distance_m <= 0:
            return None
        return (self.total_distance_m / 1000.0) / len(self.unknown_hazard_events)


@dataclass
class HazardEvent:
    """Representation of a hazardous traffic situation."""

    track_id: int
    start_frame: int
    end_frame: int
    reasons: Tuple[str, ...]
    metrics: Dict[str, float] = field(default_factory=dict)

    def duration(self, frame_rate: float) -> float:
        return (self.end_frame - self.start_frame + 1) / frame_rate


def _default_parameter_extractor(window: pd.DataFrame, frame_rate: float) -> Dict[str, float]:
    """Fallback parameter extractor returning only the duration."""

    if window.empty:
        return {"duration_s": 0.0}
    duration = (int(window.iloc[-1]["frame"]) - int(window.iloc[0]["frame"]) + 1) / frame_rate
    return {"duration_s": float(duration)}


def _follow_vehicle_cruise_parameters(window: pd.DataFrame, frame_rate: float) -> Dict[str, float]:
    params = _default_parameter_extractor(window, frame_rate)
    params.update(
        {
            "mean_thw": float(window["thw"].mean()),
            "mean_relative_speed": float(window["relative_speed"].mean()),
            "mean_speed": float(window["xVelocity"].mean()),
        }
    )
    return params


def _approaching_lead_parameters(window: pd.DataFrame, frame_rate: float) -> Dict[str, float]:
    params = _default_parameter_extractor(window, frame_rate)
    params.update(
        {
            "mean_relative_speed": float(window["relative_speed"].mean()),
            "min_ttc": float(window["ttc"].min()),
            "min_thw": float(window["thw"].min()),
        }
    )
    return params


def _lead_braking_parameters(window: pd.DataFrame, frame_rate: float) -> Dict[str, float]:
    params = _default_parameter_extractor(window, frame_rate)
    params.update(
        {
            "min_lead_acc": float(window["preceding_xAcceleration"].min()),
            "min_ttc": float(window["ttc"].min()),
            "min_thw": float(window["thw"].min()),
        }
    )
    return params


def _lead_accelerating_parameters(window: pd.DataFrame, frame_rate: float) -> Dict[str, float]:
    params = _default_parameter_extractor(window, frame_rate)
    lead_acc = window["preceding_xAcceleration"].dropna()
    if not lead_acc.empty:
        params["max_lead_acc"] = float(lead_acc.max())
    else:
        params["max_lead_acc"] = 0.0
    params["mean_relative_speed"] = float(window["relative_speed"].mean())
    return params


def _cut_in_parameters(window: pd.DataFrame, frame_rate: float) -> Dict[str, float]:
    params = _default_parameter_extractor(window, frame_rate)
    params.update(
        {
            "gap_post_cut": float(window["dhw"].iloc[-1]),
            "relative_speed_post": float(window["relative_speed"].iloc[-1]),
            "ttc_post": float(window["ttc"].iloc[-1]),
        }
    )
    return params


def _cut_out_parameters(window: pd.DataFrame, frame_rate: float) -> Dict[str, float]:
    params = _default_parameter_extractor(window, frame_rate)
    params.update(
        {
            "gap_before_cut": float(window["dhw"].iloc[0]),
            "relative_speed_before": float(window["relative_speed"].iloc[0]),
        }
    )
    return params


def _lane_change_parameters(window: pd.DataFrame, frame_rate: float) -> Dict[str, float]:
    params = _default_parameter_extractor(window, frame_rate)
    params.update(
        {
            "max_abs_y_velocity": float(window["yVelocity"].abs().max()),
            "speed_mean": float(window["xVelocity"].mean()),
        }
    )
    return params


def _overtaking_parameters(window: pd.DataFrame, frame_rate: float) -> Dict[str, float]:
    params = _default_parameter_extractor(window, frame_rate)
    params.update(
        {
            "mean_speed": float(window["xVelocity"].mean()),
            "max_relative_speed": float(window["relative_speed"].max()),
        }
    )
    return params


def _overtaken_parameters(window: pd.DataFrame, frame_rate: float) -> Dict[str, float]:
    params = _default_parameter_extractor(window, frame_rate)
    params["mean_speed"] = float(window["xVelocity"].mean())
    side = "both"
    if "tag_overtaken_left" in window and "tag_overtaken_right" in window:
        left = bool(window["tag_overtaken_left"].any())
        right = bool(window["tag_overtaken_right"].any())
        if left and not right:
            side = "left"
        elif right and not left:
            side = "right"
    params["passing_side"] = side
    return params


PARAMETER_FUNCTIONS: Mapping[str, Callable[[pd.DataFrame, float], Dict[str, float]]] = {
    "follow_vehicle_cruise": _follow_vehicle_cruise_parameters,
    "approach_low_speed_vehicle": _approaching_lead_parameters,
    "lead_vehicle_braking": _lead_braking_parameters,
    "lead_vehicle_accelerating": _lead_accelerating_parameters,
    "lead_vehicle_cut_in": _cut_in_parameters,
    "lead_vehicle_cut_out": _cut_out_parameters,
    "ego_lane_change_with_trailing_vehicle": _lane_change_parameters,
    "ego_merge_with_trailing_vehicle": _lane_change_parameters,
    "ego_overtaking": _overtaking_parameters,
    "ego_overtaken_by_vehicle": _overtaken_parameters,
}


class HighDScenarioDetector:
    """Tag-based scenario detector tailored to HighD trajectory data."""

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
        acceleration_threshold: float = 0.3,
        approach_rel_speed: float = 1.0,
        lane_change_window_s: float = 0.6,
        cut_window_s: float = 0.5,
        lead_acceleration_threshold: float = 0.5,
        overtake_window_s: float = 6.0,
        overtake_rel_speed: float = 1.0,
        overtaken_memory_s: float = 4.0,
        hazard_ttc_threshold: float = 1.5,
        hazard_thw_threshold: float = 0.8,
        hazard_dhw_threshold: float = 10.0,
    ) -> None:
        self.frame_rate = frame_rate
        self.min_free_speed = min_free_speed
        self.free_gap = free_gap
        self.following_rel_speed = following_rel_speed
        self.braking_threshold = braking_threshold
        self.lead_braking_threshold = lead_braking_threshold
        self.slow_speed_threshold = slow_speed_threshold
        self.stationary_lead_speed = stationary_lead_speed
        self.acceleration_threshold = acceleration_threshold
        self.approach_rel_speed = approach_rel_speed
        self.lane_change_window_s = lane_change_window_s
        self.cut_window_s = cut_window_s
        self.lead_acceleration_threshold = lead_acceleration_threshold
        self.overtake_window_s = overtake_window_s
        self.overtake_rel_speed = overtake_rel_speed
        self.overtaken_memory_s = overtaken_memory_s
        self.hazard_ttc_threshold = hazard_ttc_threshold
        self.hazard_thw_threshold = hazard_thw_threshold
        self.hazard_dhw_threshold = hazard_dhw_threshold

    # ------------------------------------------------------------------
    # public API
    def detect(self, tracks: pd.DataFrame) -> DetectionResult:
        """Detect scenarios and frame coverage in the provided HighD tracks dataframe."""

        self._validate_columns(tracks)
        prepared = self._prepare_dataframe(tracks)
        events: List[ScenarioEvent] = []
        hazard_events: List[HazardEvent] = []
        unknown_hazard_events: List[HazardEvent] = []
        unknown_rows: List[Dict[str, float | int | str]] = []
        covered_frames: Dict[int, set[int]] = {}
        total_frames = 0
        total_distance_m = 0.0
        unmatched_rows: List[Dict[str, int]] = []

        for track_id, track_df in prepared.groupby("id"):
            sorted_track = track_df.sort_values("frame").reset_index(drop=True)
            tagged_track = self._tag_track(sorted_track)
            track_events = self._detect_for_track(track_id=int(track_id), track=tagged_track)
            events.extend(track_events)

            frame_set: set[int] = set()
            for event in track_events:
                frame_set.update(range(event.start_frame, event.end_frame + 1))
            covered_frames[int(track_id)] = frame_set

            track_frames = sorted_track["frame"].astype(int).tolist()
            total_frames += len(track_frames)
            total_distance_m += float(sorted_track["xVelocity"].fillna(0.0).sum() / self.frame_rate)
            uncovered = [frame for frame in track_frames if frame not in frame_set]
            unmatched_rows.extend(
                {"track_id": int(track_id), "frame": int(frame)} for frame in uncovered
            )

            track_hazards = self._detect_hazardous_events(int(track_id), tagged_track)
            hazard_events.extend(track_hazards)

            for hazard in track_hazards:
                hazard_frames = set(range(hazard.start_frame, hazard.end_frame + 1))
                if hazard_frames & frame_set:
                    continue
                unknown_hazard_events.append(hazard)
                unknown_rows.append(
                    {
                        "track_id": hazard.track_id,
                        "start_frame": hazard.start_frame,
                        "end_frame": hazard.end_frame,
                        "min_ttc": hazard.metrics.get("min_ttc", float("nan")),
                        "min_thw": hazard.metrics.get("min_thw", float("nan")),
                        "min_dhw": hazard.metrics.get("min_dhw", float("nan")),
                        "reasons": ",".join(hazard.reasons),
                    }
                )

        unmatched_frames = pd.DataFrame(unmatched_rows, columns=["track_id", "frame"])
        unknown_hazard_frames = pd.DataFrame(
            unknown_rows,
            columns=["track_id", "start_frame", "end_frame", "min_ttc", "min_thw", "min_dhw", "reasons"],
        )
        return DetectionResult(
            events=events,
            unmatched_frames=unmatched_frames,
            total_frames=total_frames,
            hazard_events=hazard_events,
            unknown_hazard_events=unknown_hazard_events,
            unknown_hazard_frames=unknown_hazard_frames,
            total_distance_m=total_distance_m,
        )

    # ------------------------------------------------------------------
    # dataframe preparation
    def _validate_columns(self, tracks: pd.DataFrame) -> None:
        missing = [col for col in self.required_columns if col not in tracks.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def _prepare_dataframe(self, tracks: pd.DataFrame) -> pd.DataFrame:
        df = tracks.copy()
        df["thw"] = df["thw"].replace({0: np.nan, -1: np.nan})
        df["ttc"] = df["ttc"].replace({0: np.nan, -1: np.nan})
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
    # track-level detection
    def _detect_for_track(self, track_id: int, track: pd.DataFrame) -> List[ScenarioEvent]:
        events: List[ScenarioEvent] = []
        for pattern in self._scenario_patterns():
            events.extend(self._match_pattern(track_id, track, pattern))
        return events

    def _scenario_patterns(self) -> Sequence[ScenarioPattern]:
        patterns: List[ScenarioPattern] = []
        for definition in SCENARIO_DEFINITIONS.values():
            if not definition.tag_combination:
                continue
            combo = definition.tag_combination
            required = tuple(combo.get("required", ()))
            any_tags = tuple(combo.get("any", ()))
            forbidden = tuple(combo.get("forbidden", ()))
            parameter_fn = PARAMETER_FUNCTIONS.get(definition.name)
            patterns.append(
                ScenarioPattern(
                    name=definition.name,
                    required_tags=required,
                    any_tags=any_tags,
                    forbidden_tags=forbidden,
                    min_duration_s=definition.min_duration_s,
                    expansion_s=definition.expansion_s,
                    parameter_fn=parameter_fn,
                )
            )
        return patterns

    # ------------------------------------------------------------------
    # tag computation and pattern matching
    def _tag_track(self, track: pd.DataFrame) -> pd.DataFrame:
        track = track.copy()
        acc = track["xAcceleration"].fillna(0.0)
        track["tag_lon_accelerating"] = acc >= self.acceleration_threshold
        track["tag_lon_decelerating"] = acc <= -self.acceleration_threshold
        track["tag_lon_hard_brake"] = acc <= self.braking_threshold
        track["tag_lon_cruising"] = ~(track["tag_lon_accelerating"] | track["tag_lon_decelerating"])

        velocity = track["xVelocity"].fillna(0.0)
        track["tag_speed_high"] = velocity >= self.min_free_speed
        track["tag_slow_speed"] = velocity <= self.slow_speed_threshold

        lead_present = track["precedingId"].fillna(0) > 0
        track["tag_lead_present"] = lead_present
        track["tag_free_flow"] = (~lead_present) | (track["dhw"].fillna(np.inf) > self.free_gap)

        thw = track["thw"].fillna(np.inf)
        track["tag_following_medium"] = lead_present & thw.between(0.8, 3.0)
        track["tag_following_close"] = lead_present & (thw < 1.0)

        rel_speed = track["relative_speed"].fillna(0.0)
        track["tag_approaching_lead"] = lead_present & (rel_speed > self.approach_rel_speed)

        lead_acc = track["preceding_xAcceleration"].fillna(0.0)
        track["tag_lead_braking"] = lead_present & (lead_acc <= self.lead_braking_threshold)
        track["tag_lead_accelerating"] = lead_present & (lead_acc >= self.lead_acceleration_threshold)

        lead_speed = track["preceding_xVelocity"].fillna(np.inf)
        track["tag_lead_stationary"] = lead_present & (lead_speed <= self.stationary_lead_speed)

        track["tag_stop_and_go"] = track["tag_slow_speed"] & track["tag_lon_accelerating"]

        lane_change_window = max(1, int(round(self.lane_change_window_s * self.frame_rate)))
        cut_window = max(1, int(round(self.cut_window_s * self.frame_rate)))
        overtake_window = max(1, int(round(self.overtake_window_s * self.frame_rate)))
        overtaken_window = max(1, int(round(self.overtaken_memory_s * self.frame_rate)))

        lane_series = track["laneId"].copy()
        lane_series = lane_series.ffill().bfill()
        lane_diff = lane_series.diff().fillna(0)
        left_indices = np.where(lane_diff < 0)[0]
        right_indices = np.where(lane_diff > 0)[0]

        tag_lane_change_left = np.zeros(len(track), dtype=bool)
        tag_lane_change_right = np.zeros(len(track), dtype=bool)
        tag_lane_change_left_trailing = np.zeros(len(track), dtype=bool)
        tag_lane_change_right_trailing = np.zeros(len(track), dtype=bool)
        tag_merge_left = np.zeros(len(track), dtype=bool)
        tag_merge_right = np.zeros(len(track), dtype=bool)

        def _has_trailing(side: str, start: int, end: int) -> bool:
            if start > end:
                return False
            if side == "left":
                columns = ["leftFollowingId", "leftAlongsideId"]
            else:
                columns = ["rightFollowingId", "rightAlongsideId"]
            for col in columns:
                window = track.iloc[start : end + 1][col].fillna(0)
                if (window > 0).any():
                    return True
            return False

        for idx in left_indices:
            start = max(0, idx - lane_change_window)
            end = min(len(track) - 1, idx + lane_change_window)
            tag_lane_change_left[start : end + 1] = True
            if _has_trailing("left", start, end):
                tag_lane_change_left_trailing[start : end + 1] = True

            prev_idx = max(0, start - 1)
            next_idx = min(len(track) - 1, end + 1)
            prev_lead_present = bool(track.iloc[prev_idx]["tag_lead_present"])
            next_lead_present = bool(track.iloc[next_idx]["tag_lead_present"])
            if not prev_lead_present and next_lead_present and _has_trailing("left", start, end):
                tag_merge_left[start : end + 1] = True

        for idx in right_indices:
            start = max(0, idx - lane_change_window)
            end = min(len(track) - 1, idx + lane_change_window)
            tag_lane_change_right[start : end + 1] = True
            if _has_trailing("right", start, end):
                tag_lane_change_right_trailing[start : end + 1] = True

            prev_idx = max(0, start - 1)
            next_idx = min(len(track) - 1, end + 1)
            prev_lead_present = bool(track.iloc[prev_idx]["tag_lead_present"])
            next_lead_present = bool(track.iloc[next_idx]["tag_lead_present"])
            if not prev_lead_present and next_lead_present and _has_trailing("right", start, end):
                tag_merge_right[start : end + 1] = True

        track["tag_lane_change_left"] = tag_lane_change_left
        track["tag_lane_change_right"] = tag_lane_change_right
        track["tag_lane_change_left_trailing"] = tag_lane_change_left_trailing
        track["tag_lane_change_right_trailing"] = tag_lane_change_right_trailing
        track["tag_merge_left"] = tag_merge_left
        track["tag_merge_right"] = tag_merge_right
        track["tag_lane_keep"] = ~(tag_lane_change_left | tag_lane_change_right)

        tag_cut_in_left = np.zeros(len(track), dtype=bool)
        tag_cut_in_right = np.zeros(len(track), dtype=bool)
        tag_cut_out_left = np.zeros(len(track), dtype=bool)
        tag_cut_out_right = np.zeros(len(track), dtype=bool)

        current_lead = track["precedingId"].fillna(0)
        previous_lead = current_lead.shift(1).fillna(current_lead)
        next_lead = current_lead.shift(-1).fillna(current_lead)

        left_cols = ["leftPrecedingId", "leftAlongsideId", "leftFollowingId"]
        right_cols = ["rightPrecedingId", "rightAlongsideId", "rightFollowingId"]

        def _mark_event(series: np.ndarray, index: int) -> None:
            start = max(0, index - cut_window)
            end = min(len(series) - 1, index + cut_window)
            series[start : end + 1] = True

        for idx in np.where((current_lead > 0) & (current_lead != previous_lead))[0]:
            new_lead = current_lead.iloc[idx]
            row = track.iloc[idx]
            if idx > 0 and row.get("laneId") != track.iloc[idx - 1].get("laneId"):
                continue
            if any(row.get(col) == new_lead for col in left_cols):
                _mark_event(tag_cut_in_left, idx)
            if any(row.get(col) == new_lead for col in right_cols):
                _mark_event(tag_cut_in_right, idx)

        for idx in np.where((current_lead > 0) & (current_lead != next_lead))[0]:
            old_lead = current_lead.iloc[idx]
            if old_lead <= 0:
                continue
            row_next = track.iloc[min(idx + 1, len(track) - 1)]
            if any(row_next.get(col) == old_lead for col in left_cols):
                _mark_event(tag_cut_out_left, idx)
            if any(row_next.get(col) == old_lead for col in right_cols):
                _mark_event(tag_cut_out_right, idx)

        track["tag_cut_in_left"] = tag_cut_in_left
        track["tag_cut_in_right"] = tag_cut_in_right
        track["tag_cut_out_left"] = tag_cut_out_left
        track["tag_cut_out_right"] = tag_cut_out_right

        tag_overtaking = np.zeros(len(track), dtype=bool)
        tag_overtaken_left = np.zeros(len(track), dtype=bool)
        tag_overtaken_right = np.zeros(len(track), dtype=bool)

        for idx in left_indices:
            if idx <= 0:
                continue
            original_lane = lane_series.iloc[max(0, idx - 1)]
            search_end = min(len(track) - 1, idx + overtake_window)
            rel_speed_window = track.iloc[max(0, idx - lane_change_window) : min(len(track), idx + lane_change_window + 1)][
                "relative_speed"
            ].fillna(0.0)
            if rel_speed_window.max() <= self.overtake_rel_speed:
                continue
            for j in right_indices:
                if j <= idx or j > search_end:
                    continue
                lane_before = lane_series.iloc[max(0, j - 1)]
                lane_after = lane_series.iloc[j]
                if lane_before < lane_after and lane_after == original_lane:
                    start = max(0, idx - lane_change_window)
                    end = min(len(track) - 1, j + lane_change_window)
                    if bool(track.iloc[start]["tag_lead_present"]):
                        tag_overtaking[start : end + 1] = True
                    break

        left_follow = track["leftFollowingId"].fillna(0).astype(int)
        left_along = track["leftAlongsideId"].fillna(0).astype(int)
        left_prec = track["leftPrecedingId"].fillna(0).astype(int)
        right_follow = track["rightFollowingId"].fillna(0).astype(int)
        right_along = track["rightAlongsideId"].fillna(0).astype(int)
        right_prec = track["rightPrecedingId"].fillna(0).astype(int)

        for idx in range(1, len(track)):
            current = left_prec.iloc[idx]
            if current > 0 and current != left_prec.iloc[idx - 1]:
                window_start = max(0, idx - overtaken_window)
                history_ids = set(left_follow.iloc[window_start:idx]) | set(left_along.iloc[window_start:idx])
                if current in history_ids and bool(track.iloc[idx]["tag_lane_keep"]):
                    start = max(0, idx - lane_change_window)
                    end = min(len(track) - 1, idx + lane_change_window)
                    tag_overtaken_left[start : end + 1] = True

            current_right = right_prec.iloc[idx]
            if current_right > 0 and current_right != right_prec.iloc[idx - 1]:
                window_start = max(0, idx - overtaken_window)
                history_ids = set(right_follow.iloc[window_start:idx]) | set(right_along.iloc[window_start:idx])
                if current_right in history_ids and bool(track.iloc[idx]["tag_lane_keep"]):
                    start = max(0, idx - lane_change_window)
                    end = min(len(track) - 1, idx + lane_change_window)
                    tag_overtaken_right[start : end + 1] = True

        track["tag_overtaking"] = tag_overtaking
        track["tag_overtaken_left"] = tag_overtaken_left
        track["tag_overtaken_right"] = tag_overtaken_right
        track["tag_overtaken"] = tag_overtaken_left | tag_overtaken_right

        return track

    def _match_pattern(self, track_id: int, track: pd.DataFrame, pattern: ScenarioPattern) -> List[ScenarioEvent]:
        mask = np.ones(len(track), dtype=bool)

        for tag in pattern.required_tags:
            if tag not in track:
                mask &= False
                continue
            mask &= track[tag].to_numpy(dtype=bool)

        if pattern.any_tags:
            any_mask = np.zeros(len(track), dtype=bool)
            for tag in pattern.any_tags:
                if tag in track:
                    any_mask |= track[tag].to_numpy(dtype=bool)
            mask &= any_mask

        for tag in pattern.forbidden_tags:
            if tag in track:
                mask &= ~track[tag].to_numpy(dtype=bool)

        if not mask.any():
            return []

        min_length = max(1, int(round(pattern.min_duration_s * self.frame_rate)))
        segments = find_boolean_segments(track["frame"].astype(int).tolist(), mask.tolist(), min_length)

        events: List[ScenarioEvent] = []
        if not segments:
            return events

        expansion = int(round(pattern.expansion_s * self.frame_rate)) if pattern.expansion_s else 0
        min_frame = int(track.iloc[0]["frame"])
        max_frame = int(track.iloc[-1]["frame"])

        parameter_fn = pattern.parameter_fn or _default_parameter_extractor

        for seg in segments:
            start_frame = int(seg.start_frame)
            end_frame = int(seg.end_frame)
            if expansion:
                start_frame = max(min_frame, start_frame - expansion)
                end_frame = min(max_frame, end_frame + expansion)
            window = track[(track["frame"] >= start_frame) & (track["frame"] <= end_frame)]
            params = parameter_fn(window, self.frame_rate)
            events.append(
                ScenarioEvent(
                    scenario=pattern.name,
                    track_id=track_id,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    parameters=params,
                )
            )

        return events

    # ------------------------------------------------------------------
    # hazard detection
    def _detect_hazardous_events(self, track_id: int, track: pd.DataFrame) -> List[HazardEvent]:
        lead_present = track["tag_lead_present"].to_numpy(dtype=bool)
        ttc = track.get("ttc", pd.Series(dtype=float)).astype(float)
        thw = track.get("thw", pd.Series(dtype=float)).astype(float)
        dhw = track.get("dhw", pd.Series(dtype=float)).astype(float)
        rel_speed = track.get("relative_speed", pd.Series(dtype=float)).astype(float)

        ttc_condition = (ttc > 0) & (ttc < self.hazard_ttc_threshold)
        thw_condition = (thw > 0) & (thw < self.hazard_thw_threshold)
        dhw_condition = (dhw > 0) & (dhw < self.hazard_dhw_threshold) & (rel_speed > 0)

        hazard_mask = lead_present & (ttc_condition | thw_condition | dhw_condition)

        if not hazard_mask.any():
            return []

        segments = find_boolean_segments(
            track["frame"].astype(int).tolist(), hazard_mask.tolist(), min_length=1
        )

        hazards: List[HazardEvent] = []
        for seg in segments:
            start_frame = int(seg.start_frame)
            end_frame = int(seg.end_frame)
            window = track[(track["frame"] >= start_frame) & (track["frame"] <= end_frame)]
            reasons: List[str] = []
            metrics: Dict[str, float] = {}

            if not window["ttc"].isna().all():
                metrics["min_ttc"] = float(window["ttc"].min(skipna=True))
                if metrics["min_ttc"] < self.hazard_ttc_threshold:
                    reasons.append("low_ttc")
            if not window["thw"].isna().all():
                metrics["min_thw"] = float(window["thw"].min(skipna=True))
                if metrics["min_thw"] < self.hazard_thw_threshold:
                    reasons.append("low_thw")
            if not window["dhw"].isna().all():
                metrics["min_dhw"] = float(window["dhw"].min(skipna=True))
                if metrics["min_dhw"] < self.hazard_dhw_threshold:
                    reasons.append("short_dhw")

            metrics["mean_relative_speed"] = float(window["relative_speed"].mean(skipna=True))

            hazards.append(
                HazardEvent(
                    track_id=track_id,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    reasons=tuple(sorted(set(reasons))) or ("threshold_exceeded",),
                    metrics=metrics,
                )
            )

        return hazards
