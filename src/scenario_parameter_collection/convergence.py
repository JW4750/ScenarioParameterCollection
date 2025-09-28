"""Convergence analysis for incremental HighD scenario ingestion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

import numpy as np
import pandas as pd

from .catalog import SCENARIO_DEFINITIONS, ScenarioDefinition
from .detection import HighDScenarioDetector
from .highd_loader import load_tracks
from .statistics import ParameterDistribution, ScenarioStatistics, estimate_parameter_distributions


@dataclass(frozen=True)
class DistanceTriple:
    """Container holding the three distance measures used in the analysis."""

    mise: float
    kl_divergence: float
    hellinger: float

    def is_within(self, thresholds: "DistanceTriple") -> bool:
        return (
            self.mise <= thresholds.mise
            and self.kl_divergence <= thresholds.kl_divergence
            and self.hellinger <= thresholds.hellinger
        )


@dataclass
class FrequencyShift:
    """Distance between successive scenario frequency distributions."""

    distribution: DistanceTriple
    scenarios: List[str]
    previous: Dict[str, float]
    current: Dict[str, float]


@dataclass
class ParameterShift:
    """Distance between two iterations for a single scenario parameter."""

    scenario: str
    parameter: str
    distance: DistanceTriple


@dataclass
class ConvergenceStep:
    """Summary of a single incremental ingestion step."""

    index: int
    added_files: List[str]
    statistics: ScenarioStatistics
    frequency_shift: Optional[FrequencyShift]
    parameter_shifts: List[ParameterShift]
    max_parameter_distance: Optional[DistanceTriple]


@dataclass
class ConvergenceSummary:
    """Overall convergence evaluation for a directory of recordings."""

    steps: List[ConvergenceStep]
    thresholds: DistanceTriple
    converged_step: Optional[int]


def _normalize_counts(counts: Mapping[str, int]) -> Dict[str, float]:
    total = float(sum(counts.values()))
    if total <= 0:
        return {key: 0.0 for key in counts}
    return {key: value / total for key, value in counts.items()}


def _union_keys(previous: Mapping[str, float], current: Mapping[str, float]) -> List[str]:
    keys = sorted(set(previous).union(current))
    return keys


def _compute_frequency_shift(
    previous: Mapping[str, int],
    current: Mapping[str, int],
    epsilon: float,
) -> FrequencyShift:
    prev_norm = _normalize_counts(previous)
    curr_norm = _normalize_counts(current)

    keys = _union_keys(prev_norm, curr_norm)
    prev_vec = np.array([prev_norm.get(key, 0.0) for key in keys], dtype=float)
    curr_vec = np.array([curr_norm.get(key, 0.0) for key in keys], dtype=float)

    distances = _discrete_distances(prev_vec, curr_vec, epsilon)

    return FrequencyShift(
        distribution=distances,
        scenarios=keys,
        previous={key: prev_norm.get(key, 0.0) for key in keys},
        current={key: curr_norm.get(key, 0.0) for key in keys},
    )


def _discrete_distances(
    previous: np.ndarray, current: np.ndarray, epsilon: float
) -> DistanceTriple:
    previous = np.asarray(previous, dtype=float)
    current = np.asarray(current, dtype=float)

    diff = current - previous
    mise = float(np.sum(diff**2))

    prev_safe = np.clip(previous, epsilon, None)
    curr_safe = np.clip(current, epsilon, None)
    forward = np.sum(curr_safe * np.log(curr_safe / prev_safe))
    backward = np.sum(prev_safe * np.log(prev_safe / curr_safe))
    kl_div = float(0.5 * (forward + backward))
    hellinger = float(
        np.sqrt(0.5 * np.sum((np.sqrt(curr_safe) - np.sqrt(prev_safe)) ** 2))
    )

    return DistanceTriple(mise=mise, kl_divergence=kl_div, hellinger=hellinger)


def _build_common_grid(
    previous: ParameterDistribution, current: ParameterDistribution, points: int
) -> np.ndarray:
    lower = min(float(previous.grid[0]), float(current.grid[0]))
    upper = max(float(previous.grid[-1]), float(current.grid[-1]))
    if upper <= lower:
        upper = lower + 1.0
    return np.linspace(lower, upper, points)


def _resample_distribution(dist: ParameterDistribution, grid: np.ndarray) -> np.ndarray:
    values = np.interp(grid, dist.grid.astype(float), dist.pdf.astype(float), left=0.0, right=0.0)
    values = np.clip(values, 0.0, None)
    normalization = np.trapz(values, grid)
    if normalization > 0:
        values = values / normalization
    return values


def _continuous_distances(
    previous: ParameterDistribution,
    current: ParameterDistribution,
    epsilon: float,
    grid_points: int,
) -> DistanceTriple:
    grid = _build_common_grid(previous, current, grid_points)
    prev_pdf = _resample_distribution(previous, grid)
    curr_pdf = _resample_distribution(current, grid)

    diff = curr_pdf - prev_pdf
    mise = float(np.trapz(diff**2, grid))

    prev_safe = np.clip(prev_pdf, epsilon, None)
    curr_safe = np.clip(curr_pdf, epsilon, None)
    kl_forward = np.trapz(curr_safe * np.log(curr_safe / prev_safe), grid)
    kl_backward = np.trapz(prev_safe * np.log(prev_safe / curr_safe), grid)
    kl_div = float(0.5 * (kl_forward + kl_backward))
    hellinger_integrand = (np.sqrt(curr_safe) - np.sqrt(prev_safe)) ** 2
    hellinger = float(np.sqrt(0.5 * np.trapz(hellinger_integrand, grid)))

    return DistanceTriple(mise=mise, kl_divergence=kl_div, hellinger=hellinger)


def _compute_parameter_shifts(
    previous: ScenarioStatistics,
    current: ScenarioStatistics,
    epsilon: float,
    grid_points: int,
) -> List[ParameterShift]:
    results: List[ParameterShift] = []

    curr_params = current.parameter_distributions
    prev_params = previous.parameter_distributions
    scenarios = set(curr_params) & set(prev_params)
    for scenario in sorted(scenarios):
        curr_parameters = curr_params[scenario]
        prev_parameters = prev_params[scenario]
        parameter_names = set(curr_parameters) & set(prev_parameters)
        for name in sorted(parameter_names):
            curr_dist = curr_parameters[name]
            prev_dist = prev_parameters[name]
            distance = _continuous_distances(prev_dist, curr_dist, epsilon, grid_points)
            results.append(
                ParameterShift(
                    scenario=scenario,
                    parameter=name,
                    distance=distance,
                )
            )

    return results


def _max_distance(distances: Iterable[DistanceTriple]) -> Optional[DistanceTriple]:
    max_mise = 0.0
    max_kl = 0.0
    max_hell = 0.0
    seen = False
    for dist in distances:
        seen = True
        max_mise = max(max_mise, dist.mise)
        max_kl = max(max_kl, dist.kl_divergence)
        max_hell = max(max_hell, dist.hellinger)
    if not seen:
        return None
    return DistanceTriple(max_mise, max_kl, max_hell)


class ScenarioConvergenceAnalyzer:
    """Incrementally ingest recordings and measure distribution stability."""

    def __init__(
        self,
        frame_rate: float = 25.0,
        scenario_definitions: Mapping[str, ScenarioDefinition] | None = None,
        bandwidth: Optional[float] = None,
        grid_size: int = 128,
        epsilon: float = 1e-6,
        parameter_grid_points: int = 256,
    ) -> None:
        self.frame_rate = frame_rate
        self.scenario_definitions = scenario_definitions or SCENARIO_DEFINITIONS
        self.bandwidth = bandwidth
        self.grid_size = grid_size
        self.epsilon = float(epsilon)
        self.parameter_grid_points = int(parameter_grid_points)

    def analyze(
        self,
        tracks_path: str | Path,
        thresholds: DistanceTriple,
    ) -> ConvergenceSummary:
        path = Path(tracks_path)
        if path.is_dir():
            files = sorted(p for p in path.glob("*_tracks.csv"))
            if not files:
                raise FileNotFoundError("No *_tracks.csv files found in directory")
        else:
            if not path.exists():
                raise FileNotFoundError(path)
            files = [path]

        detector = HighDScenarioDetector(frame_rate=self.frame_rate)

        aggregated_tracks: List[pd.DataFrame] = []
        previous_stats: Optional[ScenarioStatistics] = None
        steps: List[ConvergenceStep] = []
        converged_step: Optional[int] = None

        for index, file in enumerate(files, start=1):
            new_tracks = load_tracks(file)
            aggregated_tracks.append(new_tracks)
            merged_tracks = pd.concat(aggregated_tracks, ignore_index=True, sort=False)

            detection_result = detector.detect(merged_tracks)
            events = detection_result.events
            stats = estimate_parameter_distributions(
                events,
                scenario_definitions=self.scenario_definitions,
                bandwidth=self.bandwidth,
                grid_size=self.grid_size,
            )

            frequency_shift: Optional[FrequencyShift] = None
            parameter_shifts: List[ParameterShift] = []
            max_param_distance: Optional[DistanceTriple] = None

            if previous_stats is not None:
                frequency_shift = _compute_frequency_shift(
                    previous_stats.counts,
                    stats.counts,
                    epsilon=self.epsilon,
                )
                parameter_shifts = _compute_parameter_shifts(
                    previous_stats,
                    stats,
                    epsilon=self.epsilon,
                    grid_points=self.parameter_grid_points,
                )
                max_param_distance = _max_distance(
                    shift.distance for shift in parameter_shifts
                )

            step = ConvergenceStep(
                index=index,
                added_files=[file.name],
                statistics=stats,
                frequency_shift=frequency_shift,
                parameter_shifts=parameter_shifts,
                max_parameter_distance=max_param_distance,
            )
            steps.append(step)

            if (
                converged_step is None
                and frequency_shift is not None
                and frequency_shift.distribution.is_within(thresholds)
            ):
                candidate_distances = [frequency_shift.distribution]
                if max_param_distance is not None:
                    candidate_distances.append(max_param_distance)
                if all(dist.is_within(thresholds) for dist in candidate_distances):
                    converged_step = index

            previous_stats = stats

        return ConvergenceSummary(
            steps=steps,
            thresholds=thresholds,
            converged_step=converged_step,
        )


def convergence_steps_to_dataframe(summary: ConvergenceSummary) -> pd.DataFrame:
    """Flatten the convergence summary into a dataframe for reporting."""

    records: List[Dict[str, object]] = []
    for step in summary.steps:
        base_record = {
            "step": step.index,
            "added_files": ",".join(step.added_files),
            "total_events": int(sum(step.statistics.counts.values())),
        }

        if step.frequency_shift is not None:
            freq = step.frequency_shift.distribution
            base_record.update(
                {
                    "frequency_mise": freq.mise,
                    "frequency_kl": freq.kl_divergence,
                    "frequency_hellinger": freq.hellinger,
                }
            )

        if step.max_parameter_distance is not None:
            base_record.update(
                {
                    "parameter_mise": step.max_parameter_distance.mise,
                    "parameter_kl": step.max_parameter_distance.kl_divergence,
                    "parameter_hellinger": step.max_parameter_distance.hellinger,
                }
            )

        records.append(base_record)

    return pd.DataFrame.from_records(records)


def parameter_shifts_to_dataframe(summary: ConvergenceSummary) -> pd.DataFrame:
    """Expand per-parameter distances for detailed inspection."""

    rows: List[Dict[str, object]] = []
    for step in summary.steps:
        if step.index == 1:
            continue
        for shift in step.parameter_shifts:
            rows.append(
                {
                    "step": step.index,
                    "added_files": ",".join(step.added_files),
                    "scenario": shift.scenario,
                    "parameter": shift.parameter,
                    "mise": shift.distance.mise,
                    "kl_divergence": shift.distance.kl_divergence,
                    "hellinger": shift.distance.hellinger,
                }
            )
    return pd.DataFrame.from_records(rows)

