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
    tag_combination: Dict[str, List[str]] = field(default_factory=dict)
    min_duration_s: float = 1.0
    expansion_s: float = 0.0


SCENARIO_DEFINITIONS: Dict[str, ScenarioDefinition] = {
    "follow_vehicle_cruise": ScenarioDefinition(
        name="follow_vehicle_cruise",
        description="Ego vehicle follows a preceding vehicle on the same lane with small relative speed.",
        triggers=[
            "Lead vehicle present with stable identifier",
            "Lane keeping without cut-in/out events",
            "Moderate headway and near-zero relative speed",
        ],
        key_parameters=[
            ScenarioParameter(
                name="mean_thw",
                description="Mean time headway to the lead vehicle",
                unit="s",
                typical_range=(0.8, 2.5),
            ),
            ScenarioParameter(
                name="mean_relative_speed",
                description="Average longitudinal relative speed",
                unit="m/s",
                typical_range=(-2.0, 2.0),
            ),
            ScenarioParameter(
                name="duration_s",
                description="Scenario duration",
                unit="s",
                typical_range=(2.0, 120.0),
            ),
        ],
        references=[
            "E. de Gelder et al., Coverage Metrics for a Scenario Database for the Scenario-Based Assessment of Automated Driving Systems, 2017",
            "ISO 34502 car-following definitions",
        ],
        tag_combination={
            "required": ["tag_lead_present", "tag_lane_keep", "tag_lon_cruising", "tag_following_medium"],
            "forbidden": ["tag_following_close", "tag_lead_braking"],
        },
        min_duration_s=0.5,
    ),
    "lead_vehicle_braking": ScenarioDefinition(
        name="lead_vehicle_braking",
        description="Preceding vehicle performs a braking manoeuvre relevant for longitudinal safety systems.",
        triggers=[
            "Lead vehicle longitudinal acceleration below threshold",
            "Lead vehicle remains in ego lane",
        ],
        key_parameters=[
            ScenarioParameter(
                name="min_lead_acc",
                description="Minimum acceleration of the lead vehicle",
                unit="m/s^2",
                typical_range=(-6.0, -2.0),
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
                typical_range=(0.5, 3.0),
            ),
            ScenarioParameter(
                name="duration_s",
                description="Event duration",
                unit="s",
                typical_range=(0.5, 6.0),
            ),
        ],
        references=[
            "E. de Gelder et al., Coverage Metrics for a Scenario Database for the Scenario-Based Assessment of Automated Driving Systems, 2017",
            "Euro NCAP AEB test catalogue",
        ],
        tag_combination={"required": ["tag_lead_present", "tag_lead_braking"]},
        min_duration_s=0.8,
    ),
    "lead_vehicle_accelerating": ScenarioDefinition(
        name="lead_vehicle_accelerating",
        description="Preceding vehicle accelerates and increases the distance to the ego vehicle.",
        triggers=[
            "Lead vehicle longitudinal acceleration above threshold",
            "Lead vehicle remains in ego lane",
        ],
        key_parameters=[
            ScenarioParameter(
                name="max_lead_acc",
                description="Maximum acceleration of the lead vehicle",
                unit="m/s^2",
                typical_range=(0.5, 3.0),
            ),
            ScenarioParameter(
                name="mean_relative_speed",
                description="Average relative speed while the lead pulls away",
                unit="m/s",
                typical_range=(-5.0, 3.0),
            ),
            ScenarioParameter(
                name="duration_s",
                description="Event duration",
                unit="s",
                typical_range=(0.8, 10.0),
            ),
        ],
        references=[
            "E. de Gelder et al., Coverage Metrics for a Scenario Database for the Scenario-Based Assessment of Automated Driving Systems, 2017",
            "HighD longitudinal manoeuvre studies",
        ],
        tag_combination={
            "required": ["tag_lead_present", "tag_lead_accelerating", "tag_lane_keep"],
            "forbidden": ["tag_lead_braking"],
        },
        min_duration_s=0.8,
    ),
    "approach_low_speed_vehicle": ScenarioDefinition(
        name="approach_low_speed_vehicle",
        description="Ego vehicle approaches a significantly slower or stationary object in its lane.",
        triggers=[
            "Lead vehicle present with low speed or high closing rate",
            "Lane keeping without cut-in/out",
        ],
        key_parameters=[
            ScenarioParameter(
                name="mean_relative_speed",
                description="Average relative speed towards the lead vehicle",
                unit="m/s",
                typical_range=(0.5, 10.0),
            ),
            ScenarioParameter(
                name="min_ttc",
                description="Minimum time-to-collision",
                unit="s",
                typical_range=(0.5, 6.0),
            ),
            ScenarioParameter(
                name="min_thw",
                description="Minimum time headway",
                unit="s",
                typical_range=(0.5, 2.0),
            ),
            ScenarioParameter(
                name="duration_s",
                description="Scenario duration",
                unit="s",
                typical_range=(1.0, 15.0),
            ),
        ],
        references=[
            "E. de Gelder et al., Coverage Metrics for a Scenario Database for the Scenario-Based Assessment of Automated Driving Systems, 2017",
            "ISO 34502 approaching slow vehicle",
        ],
        tag_combination={
            "required": ["tag_lead_present", "tag_lane_keep"],
            "any": ["tag_lead_stationary", "tag_approaching_lead", "tag_slow_speed"],
            "forbidden": ["tag_lead_braking"],
        },
        min_duration_s=1.2,
    ),
    "lead_vehicle_cut_in": ScenarioDefinition(
        name="lead_vehicle_cut_in",
        description="A vehicle from an adjacent lane merges into ego lane and becomes the new lead vehicle.",
        triggers=[
            "Preceding identifier switches to a neighbour vehicle",
            "Gap reduction immediately after cut-in",
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
                typical_range=(-6.0, 5.0),
            ),
            ScenarioParameter(
                name="ttc_post",
                description="Time-to-collision after the cut-in",
                unit="s",
                typical_range=(0.5, 6.0),
            ),
            ScenarioParameter(
                name="duration_s",
                description="Observation window around the cut-in",
                unit="s",
                typical_range=(0.4, 4.0),
            ),
        ],
        references=[
            "E. de Gelder et al., Coverage Metrics for a Scenario Database for the Scenario-Based Assessment of Automated Driving Systems, 2017",
            "NHTSA cut-in scenario catalogue",
        ],
        tag_combination={
            "required": ["tag_lead_present"],
            "any": ["tag_cut_in_left", "tag_cut_in_right"],
        },
        min_duration_s=0.6,
        expansion_s=0.4,
    ),
    "lead_vehicle_cut_out": ScenarioDefinition(
        name="lead_vehicle_cut_out",
        description="The current lead vehicle leaves ego lane, exposing the next vehicle ahead or free space.",
        triggers=[
            "Preceding identifier disappears towards adjacent lane",
            "Temporary increase of headway",
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
                typical_range=(-6.0, 6.0),
            ),
            ScenarioParameter(
                name="duration_s",
                description="Observation window around the cut-out",
                unit="s",
                typical_range=(0.4, 4.0),
            ),
        ],
        references=[
            "E. de Gelder et al., Coverage Metrics for a Scenario Database for the Scenario-Based Assessment of Automated Driving Systems, 2017",
            "ISO 34502 cut-out taxonomy",
        ],
        tag_combination={"any": ["tag_cut_out_left", "tag_cut_out_right"]},
        min_duration_s=0.6,
        expansion_s=0.4,
    ),
    "ego_lane_change_with_trailing_vehicle": ScenarioDefinition(
        name="ego_lane_change_with_trailing_vehicle",
        description="Ego vehicle changes lane while a trailing vehicle is present in the target lane.",
        triggers=[
            "Lane change detected from laneId sequence",
            "Target lane contains a following or alongside vehicle",
        ],
        key_parameters=[
            ScenarioParameter(
                name="duration_s",
                description="Lane change duration",
                unit="s",
                typical_range=(1.0, 6.0),
            ),
            ScenarioParameter(
                name="max_abs_y_velocity",
                description="Maximum lateral speed during the manoeuvre",
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
            "E. de Gelder et al., Coverage Metrics for a Scenario Database for the Scenario-Based Assessment of Automated Driving Systems, 2017",
            "HighD lane change analyses",
        ],
        tag_combination={"any": ["tag_lane_change_left_trailing", "tag_lane_change_right_trailing"]},
        min_duration_s=1.0,
        expansion_s=0.4,
    ),
    "ego_merge_with_trailing_vehicle": ScenarioDefinition(
        name="ego_merge_with_trailing_vehicle",
        description="Ego merges from an on-ramp or shoulder into the main lane with a trailing vehicle present in the target lane.",
        triggers=[
            "Lane change with previously free lane conditions",
            "Target lane contains a following vehicle during merge",
        ],
        key_parameters=[
            ScenarioParameter(
                name="duration_s",
                description="Merge duration",
                unit="s",
                typical_range=(1.0, 6.0),
            ),
            ScenarioParameter(
                name="max_abs_y_velocity",
                description="Maximum lateral speed during the merge",
                unit="m/s",
                typical_range=(0.1, 3.0),
            ),
            ScenarioParameter(
                name="speed_mean",
                description="Mean longitudinal speed during the merge",
                unit="m/s",
                typical_range=(10.0, 35.0),
            ),
        ],
        references=[
            "E. de Gelder et al., Coverage Metrics for a Scenario Database for the Scenario-Based Assessment of Automated Driving Systems, 2017",
            "Ramp merging behaviour studies",
        ],
        tag_combination={"any": ["tag_merge_left", "tag_merge_right"]},
        min_duration_s=1.0,
        expansion_s=0.6,
    ),
    "ego_overtaking": ScenarioDefinition(
        name="ego_overtaking",
        description="Ego vehicle overtakes a slower vehicle by performing a lane change out and returning to the original lane.",
        triggers=[
            "Sequence of lane changes forming an overtake",
            "Positive relative speed while changing lanes",
        ],
        key_parameters=[
            ScenarioParameter(
                name="duration_s",
                description="Overtaking manoeuvre duration",
                unit="s",
                typical_range=(3.0, 15.0),
            ),
            ScenarioParameter(
                name="mean_speed",
                description="Mean ego speed during the manoeuvre",
                unit="m/s",
                typical_range=(20.0, 45.0),
            ),
            ScenarioParameter(
                name="max_relative_speed",
                description="Maximum relative speed to the overtaken vehicle",
                unit="m/s",
                typical_range=(0.5, 8.0),
            ),
        ],
        references=[
            "E. de Gelder et al., Coverage Metrics for a Scenario Database for the Scenario-Based Assessment of Automated Driving Systems, 2017",
            "German motorway overtaking analyses",
        ],
        tag_combination={"required": ["tag_overtaking"]},
        min_duration_s=2.0,
    ),
    "ego_overtaken_by_vehicle": ScenarioDefinition(
        name="ego_overtaken_by_vehicle",
        description="A surrounding vehicle overtakes ego on an adjacent lane without entering ego lane.",
        triggers=[
            "Adjacent-lane vehicle transitions from following to preceding",
            "Ego remains in its lane",
        ],
        key_parameters=[
            ScenarioParameter(
                name="duration_s",
                description="Duration of the overtaken episode",
                unit="s",
                typical_range=(2.0, 12.0),
            ),
            ScenarioParameter(
                name="mean_speed",
                description="Mean ego speed during the event",
                unit="m/s",
                typical_range=(15.0, 40.0),
            ),
            ScenarioParameter(
                name="passing_side",
                description="Side on which the overtaking vehicle passed (left/right)",
            ),
        ],
        references=[
            "E. de Gelder et al., Coverage Metrics for a Scenario Database for the Scenario-Based Assessment of Automated Driving Systems, 2017",
            "Traffic flow studies on passing manoeuvres",
        ],
        tag_combination={"required": ["tag_overtaken"]},
        min_duration_s=1.5,
    ),
}


def list_scenarios() -> Iterable[ScenarioDefinition]:
    """Return scenario definitions ordered by name."""

    return SCENARIO_DEFINITIONS.values()
