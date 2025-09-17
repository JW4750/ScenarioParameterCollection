"""Statistics utilities for scenario events."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional

import numpy as np
import pandas as pd
try:  # pragma: no cover - optional dependency import
    from scipy.stats import gaussian_kde
except Exception:  # pragma: no cover - executed when SciPy is unavailable
    gaussian_kde = None

from .catalog import SCENARIO_DEFINITIONS, ScenarioDefinition
from .detection import ScenarioEvent


@dataclass
class ParameterDistribution:
    """Probability distribution estimate for a scenario parameter."""

    parameter: str
    grid: np.ndarray
    pdf: np.ndarray

    def to_dict(self) -> Dict[str, List[float]]:
        return {
            "parameter": self.parameter,
            "grid": self.grid.astype(float).tolist(),
            "pdf": self.pdf.astype(float).tolist(),
        }


@dataclass
class ScenarioStatistics:
    """Container for aggregated scenario statistics."""

    counts: Dict[str, int]
    parameter_distributions: Dict[str, Dict[str, ParameterDistribution]]
    events: pd.DataFrame

    def to_dict(self) -> Dict[str, object]:
        distributions: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
        for scenario, params in self.parameter_distributions.items():
            distributions[scenario] = {}
            for name, dist in params.items():
                distributions[scenario][name] = dist.to_dict()
        return {
            "counts": self.counts,
            "parameter_distributions": distributions,
            "events": self.events.to_dict(orient="records"),
        }


def events_to_dataframe(events: Iterable[ScenarioEvent]) -> pd.DataFrame:
    """Convert a list of events to a pandas dataframe."""

    records: List[Dict[str, object]] = []
    for event in events:
        base = {
            "scenario": event.scenario,
            "track_id": event.track_id,
            "start_frame": event.start_frame,
            "end_frame": event.end_frame,
        }
        base.update(event.parameters)
        records.append(base)
    if not records:
        return pd.DataFrame(columns=["scenario", "track_id", "start_frame", "end_frame"])
    return pd.DataFrame.from_records(records)


def _gaussian_kernel_pdf(
    values: np.ndarray, grid: np.ndarray, bandwidth: Optional[float]
) -> np.ndarray:
    """Estimate a smooth probability density using Gaussian kernels.

    This is used as a lightweight fallback when SciPy is not available. The
    implementation follows the classic Gaussian KDE formulation with a
    Silverman-style bandwidth heuristic.
    """

    values = np.asarray(values, dtype=float)
    if bandwidth is None:
        if values.size <= 1:
            bandwidth = 1.0
        else:
            std = float(np.nanstd(values, ddof=1))
            if not np.isfinite(std) or std <= 0:
                std = float(np.nanstd(values, ddof=0))
            if not np.isfinite(std) or std <= 0:
                std = 1.0
            bandwidth = 1.06 * std * (values.size ** (-1 / 5))
            if bandwidth <= 0:
                bandwidth = 1.0
    else:
        bandwidth = float(bandwidth)
        if bandwidth <= 0:
            bandwidth = 1.0

    diffs = (grid[:, None] - values[None, :]) / bandwidth
    kernel = np.exp(-0.5 * diffs**2) / (bandwidth * np.sqrt(2 * np.pi))
    pdf = kernel.mean(axis=1)
    normalization = np.trapz(pdf, grid)
    if normalization > 0:
        pdf = pdf / normalization
    return pdf


def _compute_pdf(
    values: np.ndarray, grid: np.ndarray, bandwidth: Optional[float]
) -> np.ndarray:
    if gaussian_kde is not None:
        kde = gaussian_kde(values, bw_method=bandwidth)
        pdf = kde(grid)
        normalization = np.trapz(pdf, grid)
        if normalization > 0:
            pdf = pdf / normalization
        return pdf
    return _gaussian_kernel_pdf(values, grid, bandwidth)


def estimate_parameter_distributions(
    events: Iterable[ScenarioEvent],
    scenario_definitions: Mapping[str, ScenarioDefinition] | None = None,
    bandwidth: Optional[float] = None,
    grid_size: int = 128,
) -> ScenarioStatistics:
    """Estimate scenario counts and parameter probability density functions."""

    if scenario_definitions is None:
        scenario_definitions = SCENARIO_DEFINITIONS

    df = events_to_dataframe(events)
    if df.empty:
        return ScenarioStatistics(counts={}, parameter_distributions={}, events=df)

    counts = df.groupby("scenario").size().to_dict()
    distributions: Dict[str, Dict[str, ParameterDistribution]] = {}

    for scenario, group in df.groupby("scenario"):
        definition = scenario_definitions.get(scenario)
        if definition is not None:
            parameter_names = [param.name for param in definition.key_parameters]
        else:
            excluded = {"scenario", "track_id", "start_frame", "end_frame"}
            parameter_names = [col for col in group.columns if col not in excluded]

        scenario_distributions: Dict[str, ParameterDistribution] = {}
        for name in parameter_names:
            if name not in group.columns:
                continue
            series = group[name].dropna()
            if series.empty:
                continue
            values = series.astype(float).to_numpy()
            if np.all(values == values[0]):
                center = values[0]
                grid = np.linspace(center - 1.0, center + 1.0, grid_size)
                pdf = np.zeros_like(grid)
                pdf[np.argmin(np.abs(grid - center))] = 1.0
            else:
                v_min = float(np.nanmin(values))
                v_max = float(np.nanmax(values))
                span = max(v_max - v_min, 1e-3)
                padding = 0.1 * span
                grid = np.linspace(v_min - padding, v_max + padding, grid_size)
                pdf = _compute_pdf(values, grid, bandwidth)
            scenario_distributions[name] = ParameterDistribution(
                parameter=name,
                grid=grid,
                pdf=pdf,
            )
        if scenario_distributions:
            distributions[scenario] = scenario_distributions

    return ScenarioStatistics(counts=counts, parameter_distributions=distributions, events=df)
