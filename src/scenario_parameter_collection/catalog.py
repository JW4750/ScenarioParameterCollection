"""Scenario catalogue and metadata for highway driving analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class ScenarioParameter:
    """Metadata about a measurable scenario parameter."""

    name: str
    description: str
    unit: Optional[str] = None
    typical_range: Optional[Tuple[float, float]] = None


@dataclass(frozen=True)
class ScenarioDefinition:
    """Definition of a functional highway scenario."""

    name: str
    description: str
    triggers: List[str]
    key_parameters: List[ScenarioParameter]
    references: List[str] = field(default_factory=list)


SCENARIO_DEFINITIONS: Dict[str, ScenarioDefinition] = {
    "free_driving": ScenarioDefinition(
        name="free_driving",
        description="Ego vehicle cruises without a relevant preceding vehicle within the sensor range.",
        triggers=[
            "No precedingId reported or distance headway beyond perception range",
            "Longitudinal speed above the minimum cruise threshold",
        ],
        key_parameters=[
            ScenarioParameter(
                name="speed",
                description="Ego longitudinal speed",
                unit="m/s",
                typical_range=(20.0, 45.0),
            ),
            ScenarioParameter(
                name="acceleration",
                description="Ego longitudinal acceleration",
                unit="m/s^2",
                typical_range=(-1.5, 1.5),
            ),
            ScenarioParameter(
                name="duration_s",
                description="Length of the free-driving episode",
                unit="s",
                typical_range=(2.0, 60.0),
            ),
        ],
        references=[
            "ISO 34502 Annex A free driving",
            "HighD benchmark studies on free-flow traffic",
        ],
    ),
    "car_following": ScenarioDefinition(
        name="car_following",
        description="Stable car-following with constant lane and bounded relative speed.",
        triggers=[
            "Constant precedingId with small longitudinal speed difference",
            "Time headway and distance headway in a comfort band",
        ],
        key_parameters=[
            ScenarioParameter(
                name="mean_thw",
                description="Mean time headway to the lead vehicle",
                unit="s",
                typical_range=(0.8, 2.5),
            ),
            ScenarioParameter(
                name="mean_dhw",
                description="Mean distance headway to the lead vehicle",
                unit="m",
                typical_range=(10.0, 80.0),
            ),
            ScenarioParameter(
                name="mean_relative_speed",
                description="Average relative speed between ego and lead vehicle",
                unit="m/s",
                typical_range=(-3.0, 3.0),
            ),
            ScenarioParameter(
                name="duration_s",
                description="Scenario duration",
                unit="s",
                typical_range=(2.0, 120.0),
            ),
        ],
        references=[
            "Treiber et al. Intelligent Driver Model calibration",
            "HighD lane-wise following studies",
        ],
    ),
    "lead_vehicle_braking": ScenarioDefinition(
        name="lead_vehicle_braking",
        description="Preceding vehicle performs a noticeable braking manoeuvre while ego is following.",
        triggers=[
            "Lead-vehicle longitudinal acceleration below braking threshold",
            "Positive closing speed or small TTC",
        ],
        key_parameters=[
            ScenarioParameter(
                name="min_lead_acc",
                description="Minimum longitudinal acceleration of the lead vehicle",
                unit="m/s^2",
                typical_range=(-6.0, -1.5),
            ),
            ScenarioParameter(
                name="min_ttc",
                description="Minimum time-to-collision during the event",
                unit="s",
                typical_range=(0.5, 6.0),
            ),
            ScenarioParameter(
                name="min_thw",
                description="Minimum time headway",
                unit="s",
                typical_range=(0.5, 2.5),
            ),
            ScenarioParameter(
                name="duration_s",
                description="Event duration",
                unit="s",
                typical_range=(0.5, 6.0),
            ),
        ],
        references=[
            "EuroNCAP AEB car-to-car rear stationary/moving scenarios",
            "SOTIF examples for lead vehicle braking",
        ],
    ),
    "ego_braking": ScenarioDefinition(
        name="ego_braking",
        description="Ego vehicle executes strong longitudinal deceleration irrespective of lead vehicle behaviour.",
        triggers=[
            "Ego longitudinal acceleration below braking threshold",
            "Speed reduction above minimum delta",
        ],
        key_parameters=[
            ScenarioParameter(
                name="min_acc",
                description="Minimum ego longitudinal acceleration",
                unit="m/s^2",
                typical_range=(-6.0, -2.0),
            ),
            ScenarioParameter(
                name="speed_drop",
                description="Speed reduction achieved during the event",
                unit="m/s",
                typical_range=(2.0, 20.0),
            ),
            ScenarioParameter(
                name="duration_s",
                description="Event duration",
                unit="s",
                typical_range=(0.5, 5.0),
            ),
        ],
        references=[
            "UNECE emergency braking test descriptions",
        ],
    ),
    "cut_in_from_left": ScenarioDefinition(
        name="cut_in_from_left",
        description="A vehicle from the left adjacent lane merges in front of ego and becomes the new leader.",
        triggers=[
            "PrecedingId switch accompanied by left-neighbour identifiers",
            "Sharp decrease of distance headway",
        ],
        key_parameters=[
            ScenarioParameter(
                name="gap_post_cut",
                description="Distance headway immediately after the cut-in",
                unit="m",
                typical_range=(5.0, 40.0),
            ),
            ScenarioParameter(
                name="relative_speed_post",
                description="Relative speed to the cutting-in vehicle",
                unit="m/s",
                typical_range=(-5.0, 5.0),
            ),
            ScenarioParameter(
                name="ttc_post",
                description="Time-to-collision after the cut-in",
                unit="s",
                typical_range=(0.5, 6.0),
            ),
        ],
        references=[
            "NHTSA lane change scenario definitions",
        ],
    ),
    "cut_in_from_right": ScenarioDefinition(
        name="cut_in_from_right",
        description="A vehicle from the right adjacent lane merges in front of ego and becomes the new leader.",
        triggers=[
            "PrecedingId switch accompanied by right-neighbour identifiers",
            "Gap reduction in right-to-left manoeuvre",
        ],
        key_parameters=[
            ScenarioParameter(
                name="gap_post_cut",
                description="Distance headway immediately after the cut-in",
                unit="m",
                typical_range=(5.0, 40.0),
            ),
            ScenarioParameter(
                name="relative_speed_post",
                description="Relative speed to the cutting-in vehicle",
                unit="m/s",
                typical_range=(-5.0, 5.0),
            ),
            ScenarioParameter(
                name="ttc_post",
                description="Time-to-collision after the cut-in",
                unit="s",
                typical_range=(0.5, 6.0),
            ),
        ],
        references=[
            "NHTSA lane change scenario definitions",
        ],
    ),
    "cut_out_to_left": ScenarioDefinition(
        name="cut_out_to_left",
        description="The current lead vehicle leaves the lane to the left, exposing a new lead vehicle or free space.",
        triggers=[
            "PrecedingId disappears and appears among left-lane neighbours",
            "Increase in distance headway or TTC",
        ],
        key_parameters=[
            ScenarioParameter(
                name="gap_before_cut",
                description="Distance headway shortly before the cut-out",
                unit="m",
                typical_range=(5.0, 60.0),
            ),
            ScenarioParameter(
                name="relative_speed_before",
                description="Relative speed prior to the cut-out",
                unit="m/s",
                typical_range=(-5.0, 5.0),
            ),
        ],
        references=[
            "ISO 34502 cut-out scenario taxonomy",
        ],
    ),
    "cut_out_to_right": ScenarioDefinition(
        name="cut_out_to_right",
        description="The current lead vehicle leaves the lane to the right, exposing a new lead vehicle or free space.",
        triggers=[
            "PrecedingId disappears and appears among right-lane neighbours",
            "Increase in distance headway or TTC",
        ],
        key_parameters=[
            ScenarioParameter(
                name="gap_before_cut",
                description="Distance headway shortly before the cut-out",
                unit="m",
                typical_range=(5.0, 60.0),
            ),
            ScenarioParameter(
                name="relative_speed_before",
                description="Relative speed prior to the cut-out",
                unit="m/s",
                typical_range=(-5.0, 5.0),
            ),
        ],
        references=[
            "ISO 34502 cut-out scenario taxonomy",
        ],
    ),
    "ego_lane_change_left": ScenarioDefinition(
        name="ego_lane_change_left",
        description="Ego vehicle changes to a lane with a lower numerical identifier (assumed to be left).",
        triggers=[
            "LaneId decrease sustained for more than one frame",
            "Non-zero lateral velocity during the transition",
        ],
        key_parameters=[
            ScenarioParameter(
                name="duration_s",
                description="Transition duration",
                unit="s",
                typical_range=(1.0, 6.0),
            ),
            ScenarioParameter(
                name="max_abs_y_velocity",
                description="Maximum absolute lateral speed during the manoeuvre",
                unit="m/s",
                typical_range=(0.1, 3.0),
            ),
            ScenarioParameter(
                name="speed_mean",
                description="Mean longitudinal speed during the manoeuvre",
                unit="m/s",
                typical_range=(15.0, 40.0),
            ),
        ],
        references=[
            "HighD lane-change benchmark",
        ],
    ),
    "ego_lane_change_right": ScenarioDefinition(
        name="ego_lane_change_right",
        description="Ego vehicle changes to a lane with a higher numerical identifier (assumed to be right).",
        triggers=[
            "LaneId increase sustained for more than one frame",
            "Non-zero lateral velocity during the transition",
        ],
        key_parameters=[
            ScenarioParameter(
                name="duration_s",
                description="Transition duration",
                unit="s",
                typical_range=(1.0, 6.0),
            ),
            ScenarioParameter(
                name="max_abs_y_velocity",
                description="Maximum absolute lateral speed during the manoeuvre",
                unit="m/s",
                typical_range=(0.1, 3.0),
            ),
            ScenarioParameter(
                name="speed_mean",
                description="Mean longitudinal speed during the manoeuvre",
                unit="m/s",
                typical_range=(15.0, 40.0),
            ),
        ],
        references=[
            "HighD lane-change benchmark",
        ],
    ),
    "slow_traffic": ScenarioDefinition(
        name="slow_traffic",
        description="Stop-and-go traffic with speeds below congestion threshold while following another vehicle.",
        triggers=[
            "Ego speed below congestion threshold",
            "Preceding vehicle present with short headway",
        ],
        key_parameters=[
            ScenarioParameter(
                name="mean_speed",
                description="Mean speed during the congested period",
                unit="m/s",
                typical_range=(0.0, 12.0),
            ),
            ScenarioParameter(
                name="mean_thw",
                description="Average headway in congestion",
                unit="s",
                typical_range=(0.5, 1.5),
            ),
            ScenarioParameter(
                name="duration_s",
                description="Event duration",
                unit="s",
                typical_range=(3.0, 120.0),
            ),
        ],
        references=[
            "Stop-and-go traffic characterisation in NGSim/HighD",
        ],
    ),
    "stationary_lead": ScenarioDefinition(
        name="stationary_lead",
        description="Approach towards a nearly stationary lead vehicle or obstacle in the same lane.",
        triggers=[
            "Lead speed below threshold",
            "Time-to-collision below caution limit",
        ],
        key_parameters=[
            ScenarioParameter(
                name="min_ttc",
                description="Minimum time-to-collision",
                unit="s",
                typical_range=(0.5, 4.0),
            ),
            ScenarioParameter(
                name="lead_speed",
                description="Mean lead-vehicle speed",
                unit="m/s",
                typical_range=(0.0, 5.0),
            ),
            ScenarioParameter(
                name="duration_s",
                description="Duration of the encounter",
                unit="s",
                typical_range=(0.5, 10.0),
            ),
        ],
        references=[
            "AEB stationary target scenarios",
        ],
    ),
}


def list_scenarios() -> Iterable[ScenarioDefinition]:
    """Return scenario definitions ordered by name."""

    return SCENARIO_DEFINITIONS.values()
