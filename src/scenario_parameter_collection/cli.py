"""Command line entry point for scenario detection and statistics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from .catalog import SCENARIO_DEFINITIONS
from .coverage import ERWIN_SCENARIOS, compute_erwin_coverage
from .detection import HighDScenarioDetector
from .highd_loader import load_tracks
from .statistics import estimate_parameter_distributions


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detect HighD scenarios and compute statistics.")
    parser.add_argument(
        "--tracks",
        required=True,
        help="Path to a HighD *_tracks.csv file or a directory containing multiple files.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory where results (CSV/JSON) will be written.",
    )
    parser.add_argument(
        "--bandwidth",
        type=float,
        default=None,
        help="Optional bandwidth override for the Gaussian KDE estimator.",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=256,
        help="Number of evaluation points for each probability density function.",
    )
    parser.add_argument(
        "--frame-rate",
        type=float,
        default=25.0,
        help="Frame rate of the recording (frames per second).",
    )
    return parser


def write_outputs(output_dir: Path, stats, coverage) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    events_path = output_dir / "scenario_events.csv"
    counts_path = output_dir / "scenario_counts.csv"
    distributions_path = output_dir / "parameter_distributions.json"
    erwin_counts_path = output_dir / "erwin_coverage.csv"
    erwin_summary_path = output_dir / "erwin_coverage_summary.json"
    unmatched_path = output_dir / "unmapped_events.csv"

    stats.events.to_csv(events_path, index=False)
    counts_df = pd.DataFrame(
        sorted(stats.counts.items()), columns=["scenario", "count"]
    )
    counts_df.to_csv(counts_path, index=False)
    with distributions_path.open("w", encoding="utf-8") as fp:
        json.dump(stats.to_dict()["parameter_distributions"], fp, indent=2, ensure_ascii=False)

    erwin_counts = coverage.to_counts_dict()
    erwin_rows = [
        {
            "erwin_scenario": name,
            "count": erwin_counts[name],
            "description": ERWIN_SCENARIOS[name].description,
        }
        for name in ERWIN_SCENARIOS
    ]
    pd.DataFrame(erwin_rows).to_csv(erwin_counts_path, index=False)

    summary_payload = {
        "total_events": coverage.total_events,
        "mapped_events": coverage.mapped_events,
        "coverage_ratio": coverage.coverage_ratio(),
        "unmatched_events": len(coverage.unmatched_events),
    }
    with erwin_summary_path.open("w", encoding="utf-8") as fp:
        json.dump(summary_payload, fp, indent=2, ensure_ascii=False)

    unmatched_rows = [
        {
            "scenario": event.scenario,
            "track_id": event.track_id,
            "start_frame": event.start_frame,
            "end_frame": event.end_frame,
            "start_time_s": event.start_time_s,
            "end_time_s": event.end_time_s,
        }
        for event in coverage.unmatched_events
    ]
    pd.DataFrame(
        unmatched_rows,
        columns=[
            "scenario",
            "track_id",
            "start_frame",
            "end_frame",
            "start_time_s",
            "end_time_s",
        ],
    ).to_csv(unmatched_path, index=False)


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    tracks = load_tracks(args.tracks)
    detector = HighDScenarioDetector(frame_rate=args.frame_rate)
    events = detector.detect(tracks)
    stats = estimate_parameter_distributions(
        events,
        scenario_definitions=SCENARIO_DEFINITIONS,
        bandwidth=args.bandwidth,
        grid_size=args.grid_size,
    )
    coverage = compute_erwin_coverage(events, frame_rate=args.frame_rate)

    if not stats.counts:
        print("No scenarios detected.")
        return

    print("Scenario frequencies:")
    for scenario, count in sorted(stats.counts.items(), key=lambda item: item[0]):
        print(f"  {scenario}: {count}")

    print(
        "Erwin coverage: "
        f"{coverage.mapped_events}/{coverage.total_events} "
        f"({coverage.coverage_ratio():.1%})"
    )
    if coverage.unmatched_events:
        unmatched_names = sorted({event.scenario for event in coverage.unmatched_events})
        print("Unmapped scenarios:")
        for name in unmatched_names:
            print(f"  {name}")

    output_dir = Path(args.output_dir)
    write_outputs(output_dir, stats, coverage)
    print(f"Results written to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
