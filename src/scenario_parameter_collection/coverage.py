"""Mapping HighD scenarios to Erwin de Gelder's highway functional catalogue."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List

from .detection import ScenarioEvent


@dataclass(frozen=True)
class ErwinScenario:
    """Functional scenario category defined by Erwin de Gelder."""

    name: str
    description: str
    key_parameters: List[str]
    references: List[str] = field(default_factory=list)


ERWIN_SCENARIOS: Dict[str, ErwinScenario] = {
    "follow_vehicle_cruise": ErwinScenario(
        name="follow_vehicle_cruise",
        description="Ego follows a preceding vehicle with small relative speed on the same lane.",
        key_parameters=["time_headway", "relative_speed", "ego_speed"],
        references=[
            "E. de Gelder et al., Coverage Metrics for a Scenario Database for the Scenario-Based Assessment of Automated Driving Systems, 2017",
        ],
    ),
    "lead_vehicle_braking": ErwinScenario(
        name="lead_vehicle_braking",
        description="Preceding vehicle performs a braking manoeuvre relevant for AEB systems.",
        key_parameters=["lead_acceleration", "ttc", "time_headway"],
        references=[
            "E. de Gelder et al., Coverage Metrics for a Scenario Database for the Scenario-Based Assessment of Automated Driving Systems, 2017",
        ],
    ),
    "lead_vehicle_accelerating": ErwinScenario(
        name="lead_vehicle_accelerating",
        description="Preceding vehicle accelerates and increases the gap to ego.",
        key_parameters=["lead_acceleration", "relative_speed", "ego_speed"],
        references=[
            "E. de Gelder et al., Coverage Metrics for a Scenario Database for the Scenario-Based Assessment of Automated Driving Systems, 2017",
        ],
    ),
    "approach_low_speed_vehicle": ErwinScenario(
        name="approach_low_speed_vehicle",
        description="Ego closes in on a slow or stationary object within the same lane.",
        key_parameters=["ttc", "time_headway", "relative_speed"],
        references=[
            "E. de Gelder et al., Coverage Metrics for a Scenario Database for the Scenario-Based Assessment of Automated Driving Systems, 2017",
        ],
    ),
    "lead_vehicle_cut_in": ErwinScenario(
        name="lead_vehicle_cut_in",
        description="A vehicle merges into ego lane ahead, forcing ego to adapt.",
        key_parameters=["post_cut_gap", "post_cut_ttc", "relative_speed"],
        references=[
            "E. de Gelder et al., Coverage Metrics for a Scenario Database for the Scenario-Based Assessment of Automated Driving Systems, 2017",
        ],
    ),
    "lead_vehicle_cut_out": ErwinScenario(
        name="lead_vehicle_cut_out",
        description="The preceding vehicle leaves ego lane, exposing the next vehicle ahead.",
        key_parameters=["pre_cut_gap", "relative_speed", "ttc"],
        references=[
            "E. de Gelder et al., Coverage Metrics for a Scenario Database for the Scenario-Based Assessment of Automated Driving Systems, 2017",
        ],
    ),
    "ego_lane_change_with_trailing_vehicle": ErwinScenario(
        name="ego_lane_change_with_trailing_vehicle",
        description="Ego vehicle changes lane while another vehicle is present behind in the target lane.",
        key_parameters=["target_lane_gap", "target_lane_relative_speed", "lateral_velocity"],
        references=[
            "E. de Gelder et al., Coverage Metrics for a Scenario Database for the Scenario-Based Assessment of Automated Driving Systems, 2017",
        ],
    ),
    "ego_merge_with_trailing_vehicle": ErwinScenario(
        name="ego_merge_with_trailing_vehicle",
        description="Ego merges from a ramp into the main lane with a following vehicle in target lane.",
        key_parameters=["merge_gap", "target_lane_relative_speed", "merge_duration"],
        references=[
            "E. de Gelder et al., Coverage Metrics for a Scenario Database for the Scenario-Based Assessment of Automated Driving Systems, 2017",
        ],
    ),
    "ego_overtaking": ErwinScenario(
        name="ego_overtaking",
        description="Ego overtakes a slower vehicle by changing to a faster lane and returning.",
        key_parameters=["initial_gap", "completion_gap", "overtake_duration"],
        references=[
            "E. de Gelder et al., Coverage Metrics for a Scenario Database for the Scenario-Based Assessment of Automated Driving Systems, 2017",
        ],
    ),
    "ego_overtaken_by_vehicle": ErwinScenario(
        name="ego_overtaken_by_vehicle",
        description="A surrounding vehicle overtakes ego on an adjacent lane.",
        key_parameters=["overtaker_speed", "lateral_gap", "event_duration"],
        references=[
            "E. de Gelder et al., Coverage Metrics for a Scenario Database for the Scenario-Based Assessment of Automated Driving Systems, 2017",
        ],
    ),
}


SCENARIO_TO_ERWIN: Dict[str, str] = {
    "car_following": "follow_vehicle_cruise",
    "slow_traffic": "approach_low_speed_vehicle",
    "stationary_lead": "approach_low_speed_vehicle",
    "lead_vehicle_braking": "lead_vehicle_braking",
    "cut_in_from_left": "lead_vehicle_cut_in",
    "cut_in_from_right": "lead_vehicle_cut_in",
    "cut_out_to_left": "lead_vehicle_cut_out",
    "cut_out_to_right": "lead_vehicle_cut_out",
    "ego_lane_change_left": "ego_lane_change_with_trailing_vehicle",
    "ego_lane_change_right": "ego_lane_change_with_trailing_vehicle",
}


@dataclass(frozen=True)
class UnmatchedEvent:
    """Scenario event that cannot be mapped to the Erwin catalogue."""

    scenario: str
    track_id: int
    start_frame: int
    end_frame: int
    start_time_s: float
    end_time_s: float


@dataclass
class ErwinCoverageSummary:
    """Coverage of detected scenarios with respect to Erwin de Gelder catalogue."""

    total_events: int
    mapped_events: int
    matched_counts: Dict[str, int]
    unmatched_events: List[UnmatchedEvent]

    def coverage_ratio(self) -> float:
        if self.total_events == 0:
            return 0.0
        return self.mapped_events / self.total_events

    def to_counts_dict(self) -> Dict[str, int]:
        """Return counts for all Erwin categories (including zeros)."""

        return {name: self.matched_counts.get(name, 0) for name in ERWIN_SCENARIOS}


def compute_erwin_coverage(
    events: Iterable[ScenarioEvent], *, frame_rate: float
) -> ErwinCoverageSummary:
    """Compute coverage of detected scenarios against the Erwin catalogue."""

    matched_counts: Dict[str, int] = {}
    unmatched_events: List[UnmatchedEvent] = []
    total_events = 0
    mapped_events = 0

    for event in events:
        total_events += 1
        erwin_name = SCENARIO_TO_ERWIN.get(event.scenario)
        if erwin_name is None:
            start_time_s = event.start_frame / frame_rate
            end_time_s = event.end_frame / frame_rate
            unmatched_events.append(
                UnmatchedEvent(
                    scenario=event.scenario,
                    track_id=event.track_id,
                    start_frame=event.start_frame,
                    end_frame=event.end_frame,
                    start_time_s=start_time_s,
                    end_time_s=end_time_s,
                )
            )
            continue
        mapped_events += 1
        matched_counts[erwin_name] = matched_counts.get(erwin_name, 0) + 1

    return ErwinCoverageSummary(
        total_events=total_events,
        mapped_events=mapped_events,
        matched_counts=matched_counts,
        unmatched_events=unmatched_events,
    )
