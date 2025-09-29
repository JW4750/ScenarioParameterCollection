"""Scenario parameter extraction toolkit for the HighD dataset."""

from .catalog import SCENARIO_DEFINITIONS, ScenarioDefinition

from .coverage import (
    ERWIN_SCENARIOS,
    SCENARIO_TO_ERWIN,
    ErwinCoverageSummary,
    UnmatchedEvent,
    compute_erwin_coverage,
)

from .convergence import (
    ConvergenceSummary,
    DistanceTriple,
    ScenarioConvergenceAnalyzer,
)

from .detection import DetectionResult, HighDScenarioDetector, ScenarioEvent
from .statistics import ScenarioStatistics, estimate_parameter_distributions

__all__ = [
    "SCENARIO_DEFINITIONS",
    "ScenarioDefinition",

    "ERWIN_SCENARIOS",
    "SCENARIO_TO_ERWIN",
    "HighDScenarioDetector",
    "DetectionResult",
    "ScenarioEvent",
    "ErwinCoverageSummary",
    "UnmatchedEvent",
    "ScenarioStatistics",
    "estimate_parameter_distributions",
    "compute_erwin_coverage",

    "ScenarioConvergenceAnalyzer",
    "ConvergenceSummary",
    "DistanceTriple",

]
