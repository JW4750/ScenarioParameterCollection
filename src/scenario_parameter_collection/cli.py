"""Command line entry point for scenario detection and statistics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from .catalog import SCENARIO_DEFINITIONS

from .coverage import ERWIN_SCENARIOS, compute_erwin_coverage

from .detection import DetectionResult, HighDScenarioDetector
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



def write_outputs(
    output_dir: Path,
    stats,
    coverage,
    detection_result: DetectionResult,
) -> None:

    output_dir.mkdir(parents=True, exist_ok=True)
    events_path = output_dir / "scenario_events.csv"
    counts_path = output_dir / "scenario_counts.csv"
    distributions_path = output_dir / "parameter_distributions.json"

    erwin_counts_path = output_dir / "erwin_coverage.csv"
    erwin_summary_path = output_dir / "erwin_coverage_summary.json"
    unmatched_path = output_dir / "unmapped_events.csv"
    unmatched_frames_path = output_dir / "unmatched_frames.csv"
    frame_coverage_path = output_dir / "frame_coverage_summary.json"
    hazard_events_path = output_dir / "hazard_events.csv"
    unknown_hazards_path = output_dir / "unknown_hazard_events.csv"
    unknown_hazard_frames_path = output_dir / "unknown_hazard_frames.csv"

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

    detection_result.unmatched_frames.to_csv(unmatched_frames_path, index=False)

    hazard_rows = []
    for hazard in detection_result.hazard_events:
        row = {
            "track_id": hazard.track_id,
            "start_frame": hazard.start_frame,
            "end_frame": hazard.end_frame,
            "reasons": ",".join(hazard.reasons),
        }
        row.update(hazard.metrics)
        for key in ("min_ttc", "min_thw", "min_dhw", "mean_relative_speed"):
            row.setdefault(key, float("nan"))
        hazard_rows.append(row)

    hazard_columns = [
        "track_id",
        "start_frame",
        "end_frame",
        "reasons",
        "min_ttc",
        "min_thw",
        "min_dhw",
        "mean_relative_speed",
    ]

    pd.DataFrame(hazard_rows, columns=hazard_columns).to_csv(hazard_events_path, index=False)
    unknown_rows = []
    for hazard in detection_result.unknown_hazard_events:
        row = {
            "track_id": hazard.track_id,
            "start_frame": hazard.start_frame,
            "end_frame": hazard.end_frame,
            "reasons": ",".join(hazard.reasons),
        }
        row.update(hazard.metrics)
        for key in ("min_ttc", "min_thw", "min_dhw", "mean_relative_speed"):
            row.setdefault(key, float("nan"))
        unknown_rows.append(row)

    pd.DataFrame(unknown_rows, columns=hazard_columns).to_csv(unknown_hazards_path, index=False)
    detection_result.unknown_hazard_frames.to_csv(unknown_hazard_frames_path, index=False)
    coverage_payload = {
        "total_frames": detection_result.total_frames,
        "unmatched_frames": int(len(detection_result.unmatched_frames)),
        "coverage_ratio": detection_result.coverage_ratio(),
        "hazard_events": len(detection_result.hazard_events),
        "unknown_hazard_events": len(detection_result.unknown_hazard_events),
        "kilometers_per_unknown_hazard": detection_result.kilometers_per_unknown_hazard(),
    }
    with frame_coverage_path.open("w", encoding="utf-8") as fp:
        json.dump(coverage_payload, fp, indent=2, ensure_ascii=False)



def main(argv: Iterable[str] | None = None) -> None:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    tracks = load_tracks(args.tracks)
    detector = HighDScenarioDetector(frame_rate=args.frame_rate)
    detection_result = detector.detect(tracks)
    events = detection_result.events
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

    unmatched_count = len(detection_result.unmatched_frames)
    total_frames = detection_result.total_frames
    coverage_ratio = detection_result.coverage_ratio()
    print(
        "Frame coverage: "
        f"{total_frames - unmatched_count}/{total_frames} "
        f"({coverage_ratio:.1%})"
    )

    if detection_result.hazard_events:
        print(
            "Hazardous situations detected: "
            f"{len(detection_result.hazard_events)} total / "
            f"{len(detection_result.unknown_hazard_events)} unknown"
        )
        km_between = detection_result.kilometers_per_unknown_hazard()
        if km_between is not None:
            print(
                "Average kilometres between unknown hazards: "
                f"{km_between:.3f} km"
            )

    output_dir = Path(args.output_dir)
    write_outputs(output_dir, stats, coverage, detection_result)

    print(f"Results written to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
