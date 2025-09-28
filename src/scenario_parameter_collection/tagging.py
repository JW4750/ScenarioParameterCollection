"""Frame-level tagging utilities for HighD tracks.

This module implements the first stage of the two-step scenario mining
approach described in *Real-World Scenario Mining for the Assessment of
Automated Vehicles* (Schuldt et al., 2018).  The functions compute
longitudinal and lateral action tags as well as interaction tags that
characterise the relation to surrounding traffic.  These tags can then be
combined to express higher level scenarios as suggested by the paper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

# Thresholds derived from Schuldt et al. and refined for the HighD data set.
LONGITUDINAL_ACCEL_THRESHOLD = 0.6  # m/s^2
LONGITUDINAL_DECEL_THRESHOLD = -0.6  # m/s^2
REL_SPEED_APPROACHING = 1.0  # m/s
REL_SPEED_OPENING = -1.0  # m/s
CRITICAL_THW = 1.0  # s
SHORT_THW = 2.0  # s
LONG_THW = 4.0  # s

LATERAL_WINDOW_SECONDS = 1.0  # seconds tagged as lane change around the transition


TAG_COLUMNS = (
    "longitudinal_tag",
    "lateral_tag",
    "interaction_tag",
    "gap_tag",
)


@dataclass(frozen=True)
class TagSpec:
    """Container storing the categorical tags assigned to a frame."""

    longitudinal_tag: str
    lateral_tag: str
    interaction_tag: str
    gap_tag: str

    def as_dict(self) -> Dict[str, str]:
        return {
            "longitudinal": self.longitudinal_tag,
            "lateral": self.lateral_tag,
            "interaction": self.interaction_tag,
            "gap": self.gap_tag,
        }


def assign_action_tags(tracks: pd.DataFrame, frame_rate: float) -> pd.DataFrame:
    """Return a dataframe with action tags for every frame in ``tracks``.

    Parameters
    ----------
    tracks:
        Prepared HighD dataframe containing at least the columns ``id``,
        ``frame``, ``laneId``, ``xAcceleration``, ``precedingId``,
        ``relative_speed`` (ego speed minus lead speed), and ``thw``.
    frame_rate:
        Recording frame rate in frames per second.

    Returns
    -------
    pd.DataFrame
        A dataframe aligned with ``tracks`` containing the tag columns defined
        in :data:`TAG_COLUMNS`.
    """

    result = pd.DataFrame(index=tracks.index, columns=TAG_COLUMNS, dtype=object)

    # ------------------------------------------------------------------
    # Longitudinal actions
    acc = pd.to_numeric(tracks["xAcceleration"], errors="coerce").fillna(0.0)
    longitudinal = np.full(len(acc), "cruising", dtype=object)
    longitudinal[acc >= LONGITUDINAL_ACCEL_THRESHOLD] = "accelerating"
    longitudinal[acc <= LONGITUDINAL_DECEL_THRESHOLD] = "decelerating"
    result["longitudinal_tag"] = longitudinal

    # ------------------------------------------------------------------
    # Interaction with lead vehicle / free driving
    lead_present = tracks["precedingId"].fillna(0) > 0
    relative_speed = pd.to_numeric(tracks.get("relative_speed"), errors="coerce").fillna(0.0)
    interaction = np.full(len(tracks), "free_flow", dtype=object)
    interaction[lead_present] = "following"
    interaction[lead_present & (relative_speed >= REL_SPEED_APPROACHING)] = "approaching"
    interaction[lead_present & (relative_speed <= REL_SPEED_OPENING)] = "opening_gap"
    result["interaction_tag"] = interaction

    # ------------------------------------------------------------------
    # Gap quality derived from THW (time headway) according to Schuldt et al.
    thw = pd.to_numeric(tracks.get("thw"), errors="coerce")
    gap = np.full(len(tracks), "unknown_gap", dtype=object)
    gap[np.isnan(thw)] = "no_lead"
    gap[(~np.isnan(thw)) & (thw <= CRITICAL_THW)] = "critical_gap"
    gap[(~np.isnan(thw)) & (thw > CRITICAL_THW) & (thw <= SHORT_THW)] = "short_gap"
    gap[(~np.isnan(thw)) & (thw > SHORT_THW) & (thw <= LONG_THW)] = "comfortable_gap"
    gap[(~np.isnan(thw)) & (thw > LONG_THW)] = "wide_gap"
    result["gap_tag"] = gap

    # ------------------------------------------------------------------
    # Lateral actions based on laneId transitions
    lateral = np.full(len(tracks), "lane_keeping", dtype=object)
    window = int(max(1, round(frame_rate * LATERAL_WINDOW_SECONDS)))

    for track_id, track_df in tracks.groupby("id"):
        idx = track_df.index.to_numpy()
        lanes = track_df["laneId"].to_numpy()
        diff = np.diff(lanes, prepend=lanes[0])
        for pos, delta in enumerate(diff):
            if delta == 0:
                continue
            if delta < 0:
                tag_value = "lane_change_left"
            else:
                tag_value = "lane_change_right"
            start = max(0, pos - window)
            end = min(len(track_df) - 1, pos + window)
            lateral[idx[start : end + 1]] = tag_value
    result["lateral_tag"] = lateral

    return result


def summarise_tags(window: pd.DataFrame) -> Dict[str, str]:
    """Collapse frame-level tags of ``window`` into a compact dictionary."""

    summary: Dict[str, str] = {}
    for column, key in zip(TAG_COLUMNS, ("longitudinal", "lateral", "interaction", "gap")):
        if column not in window:
            continue
        series = window[column].dropna().astype(str)
        if series.empty:
            continue
        summary[key] = series.mode().iat[0]
    return summary


__all__ = [
    "assign_action_tags",
    "summarise_tags",
    "TAG_COLUMNS",
    "TagSpec",
]
