"""CLI entry point for the convergence analysis workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from .catalog import SCENARIO_DEFINITIONS
from .convergence import (
    ConvergenceSummary,
    DistanceTriple,
    ScenarioConvergenceAnalyzer,
    convergence_steps_to_dataframe,
    parameter_shifts_to_dataframe,
)


DEFAULT_THRESHOLDS = DistanceTriple(mise=2e-3, kl_divergence=5e-3, hellinger=5e-2)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Assess convergence of scenario frequencies and parameter distributions"
    )
    parser.add_argument(
        "--tracks",
        required=True,
        help="Directory containing HighD *_tracks.csv files or a single file.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/convergence",
        help="Directory where convergence results will be written.",
    )
    parser.add_argument(
        "--frame-rate",
        type=float,
        default=25.0,
        help="Frame rate of the HighD recordings (frames per second).",
    )
    parser.add_argument(
        "--bandwidth",
        type=float,
        default=None,
        help="Optional KDE bandwidth override passed to the statistics module.",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=128,
        help="Number of samples for each probability density estimate.",
    )
    parser.add_argument(
        "--parameter-grid",
        type=int,
        default=256,
        help="Evaluation points for pairwise PDF comparisons.",
    )
    parser.add_argument(
        "--mise-threshold",
        type=float,
        default=DEFAULT_THRESHOLDS.mise,
        help="Convergence threshold for the MISE metric.",
    )
    parser.add_argument(
        "--kl-threshold",
        type=float,
        default=DEFAULT_THRESHOLDS.kl_divergence,
        help="Convergence threshold for the symmetric KL divergence.",
    )
    parser.add_argument(
        "--hellinger-threshold",
        type=float,
        default=DEFAULT_THRESHOLDS.hellinger,
        help="Convergence threshold for the Hellinger distance.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-6,
        help="Floor added to probabilities during divergence calculations.",
    )
    return parser


def write_summary(output_dir: Path, summary: ConvergenceSummary) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    steps_path = output_dir / "convergence_steps.csv"
    parameters_path = output_dir / "parameter_shifts.csv"
    summary_path = output_dir / "convergence_summary.json"

    steps_df = convergence_steps_to_dataframe(summary)
    if not steps_df.empty:
        steps_df.to_csv(steps_path, index=False)

    parameters_df = parameter_shifts_to_dataframe(summary)
    if not parameters_df.empty:
        parameters_df.to_csv(parameters_path, index=False)

    payload = {
        "thresholds": {
            "mise": summary.thresholds.mise,
            "kl_divergence": summary.thresholds.kl_divergence,
            "hellinger": summary.thresholds.hellinger,
        },
        "converged_step": summary.converged_step,
        "total_steps": len(summary.steps),
        "final_counts": summary.steps[-1].statistics.counts if summary.steps else {},
    }

    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    thresholds = DistanceTriple(
        mise=args.mise_threshold,
        kl_divergence=args.kl_threshold,
        hellinger=args.hellinger_threshold,
    )

    analyzer = ScenarioConvergenceAnalyzer(
        frame_rate=args.frame_rate,
        scenario_definitions=SCENARIO_DEFINITIONS,
        bandwidth=args.bandwidth,
        grid_size=args.grid_size,
        epsilon=args.epsilon,
        parameter_grid_points=args.parameter_grid,
    )

    summary = analyzer.analyze(args.tracks, thresholds=thresholds)

    print("Processed steps:", len(summary.steps))
    if summary.converged_step is not None:
        print(
            "Converged at step",
            summary.converged_step,
            "out of",
            len(summary.steps),
        )
    else:
        print("Convergence thresholds not reached within available recordings.")

    if summary.steps:
        last_counts = summary.steps[-1].statistics.counts
        print("Final scenario frequency distribution:")
        total = float(sum(last_counts.values())) or 1.0
        for scenario, count in sorted(last_counts.items()):
            share = count / total
            print(f"  {scenario}: {count} ({share:.1%})")

    output_dir = Path(args.output_dir)
    write_summary(output_dir, summary)
    print(f"Convergence artefacts written to {output_dir.resolve()}")


if __name__ == "__main__":
    main()

