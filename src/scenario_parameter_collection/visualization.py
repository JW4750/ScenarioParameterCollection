"""HTML report generation for scenario detection outputs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

OUTPUT_EVENTS = "scenario_events.csv"
OUTPUT_COUNTS = "scenario_counts.csv"
OUTPUT_DISTRIBUTIONS = "parameter_distributions.json"
DEFAULT_REPORT_NAME = "scenario_report.html"


def _read_counts(output_dir: Path) -> pd.DataFrame:
    counts_path = output_dir / OUTPUT_COUNTS
    if not counts_path.exists():
        return pd.DataFrame(columns=["scenario", "count"])
    df = pd.read_csv(counts_path)
    if "scenario" in df.columns:
        df["scenario"] = df["scenario"].astype(str)
    if "count" in df.columns:
        df = df.sort_values("count", ascending=False)
    return df


def _read_distributions(output_dir: Path) -> Dict[str, Dict[str, Dict[str, Iterable[float]]]]:
    distributions_path = output_dir / OUTPUT_DISTRIBUTIONS
    if not distributions_path.exists():
        return {}
    with distributions_path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    return {str(key): value for key, value in data.items()}


def _read_events(output_dir: Path) -> pd.DataFrame:
    events_path = output_dir / OUTPUT_EVENTS
    if not events_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(events_path)
    if "scenario" in df.columns:
        df["scenario"] = df["scenario"].astype(str)
    return df


def _render_counts_section(df: pd.DataFrame) -> Tuple[str, str, bool]:
    if df.empty or "count" not in df.columns or "scenario" not in df.columns:
        message = "<p class=\"empty\">No scenario counts were found.</p>"
        return message, message, False

    fig = go.Figure(
        data=[
            go.Bar(
                x=df["scenario"],
                y=df["count"],
                marker=dict(color="#2563eb"),
            )
        ]
    )
    fig.update_layout(
        title="Scenario occurrences",
        xaxis_title="Scenario",
        yaxis_title="Count",
        template="plotly_white",
        height=420,
        margin=dict(l=40, r=20, t=60, b=80),
    )

    chart_html = pio.to_html(
        fig,
        include_plotlyjs=True,
        full_html=False,
        default_width="100%",
        default_height=420,
    )
    table_html = df.reset_index(drop=True).to_html(index=False, classes="data-table")
    return chart_html, table_html, True


def _compute_parameter_summary(
    events: pd.DataFrame, distributions: Dict[str, Dict[str, Dict[str, Iterable[float]]]]
) -> Dict[str, pd.DataFrame]:
    if events.empty or "scenario" not in events.columns:
        return {}
    numeric_cols = [
        col
        for col in events.select_dtypes(include=["number"]).columns
        if col not in {"track_id", "start_frame", "end_frame"}
    ]
    if not numeric_cols:
        return {}

    summaries: Dict[str, pd.DataFrame] = {}
    for scenario, group in events.groupby("scenario"):
        allowed = set(distributions.get(scenario, {}).keys())
        if allowed:
            columns = [col for col in numeric_cols if col in allowed]
        else:
            columns = numeric_cols
        if not columns:
            continue
        numeric = group[columns]
        summary = numeric.agg(["mean", "std", "min", "max"]).transpose().reset_index()
        summary = summary.rename(
            columns={
                "index": "parameter",
                "mean": "mean",
                "std": "std",
                "min": "min",
                "max": "max",
            }
        )
        summary = summary.replace({np.nan: ""})
        if not summary.empty:
            summaries[scenario] = summary.round(3)
    return summaries


def _render_distribution_sections(
    distributions: Dict[str, Dict[str, Dict[str, Iterable[float]]]],
    events: pd.DataFrame,
    plotly_loaded: bool,
) -> Tuple[str, bool]:
    if not distributions:
        return "<p class=\"empty\">No parameter distributions were found.</p>", plotly_loaded

    summary_tables = _compute_parameter_summary(events, distributions)
    sections = []
    js_loaded = plotly_loaded

    for scenario in sorted(distributions.keys()):
        params = distributions[scenario]
        if not params:
            continue
        fig = go.Figure()
        for name, payload in sorted(params.items()):
            grid = payload.get("grid", [])
            pdf = payload.get("pdf", [])
            fig.add_trace(
                go.Scatter(
                    x=grid,
                    y=pdf,
                    mode="lines",
                    name=name,
                    hovertemplate="Value: %{x:.4f}<br>Density: %{y:.4f}<extra>" + name + "</extra>",
                )
            )
        fig.update_layout(
            title=f"{scenario} parameter distributions",
            xaxis_title="Value",
            yaxis_title="Probability density",
            template="plotly_white",
            height=420,
            legend_title="Parameter",
            hovermode="x unified",
            margin=dict(l=40, r=20, t=60, b=60),
        )
        include_js = not js_loaded
        chart_html = pio.to_html(
            fig,
            include_plotlyjs=include_js,
            full_html=False,
            default_width="100%",
            default_height=420,
        )
        js_loaded = js_loaded or include_js

        summary = summary_tables.get(scenario)
        if summary is not None and not summary.empty:
            filtered = summary[summary["parameter"].isin(params.keys())]
            if filtered.empty:
                summary_html = "<p class=\"empty\">No numeric summary available.</p>"
            else:
                summary_html = filtered.to_html(index=False, classes="data-table compact")
        else:
            summary_html = "<p class=\"empty\">No numeric summary available.</p>"

        section_html = f"""
        <article class=\"scenario-card\">
            <h3>{escape(scenario)}</h3>
            <div class=\"chart\">{chart_html}</div>
            <div class=\"table-wrapper\">
                <h4>Parameter summary</h4>
                {summary_html}
            </div>
        </article>
        """
        sections.append(section_html)

    if not sections:
        return "<p class=\"empty\">No parameter distributions were found.</p>", js_loaded

    return "\n".join(sections), js_loaded


def _render_events_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "<p class=\"empty\">No scenario events available.</p>"

    preview = df.sort_values(by=["scenario", "start_frame"], kind="stable", na_position="last").head(20)
    preview = preview.replace({np.nan: ""})
    return preview.to_html(index=False, classes="data-table wide")


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\" />
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
<title>{title}</title>
<style>
    body {{
        font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif;
        margin: 0;
        background: #f3f4f6;
        color: #1f2933;
    }}
    header {{
        background: linear-gradient(135deg, #1e3a8a, #2563eb);
        color: white;
        padding: 32px 20px;
    }}
    header h1 {{
        margin: 0 0 8px 0;
        font-size: 2rem;
    }}
    header p {{
        margin: 4px 0;
        font-size: 1rem;
    }}
    main {{
        max-width: 1200px;
        margin: 0 auto;
        padding: 24px;
    }}
    section {{
        background: white;
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px rgba(15, 23, 42, 0.08);
    }}
    section h2 {{
        margin-top: 0;
    }}
    .scenario-card {{
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 24px;
        background: #f9fafb;
    }}
    .scenario-card h3 {{
        margin-top: 0;
        margin-bottom: 12px;
    }}
    .chart {{
        margin-bottom: 16px;
    }}
    .table-wrapper {{
        overflow-x: auto;
    }}
    table {{
        border-collapse: collapse;
        width: 100%;
    }}
    th, td {{
        border: 1px solid #e5e7eb;
        padding: 8px 12px;
        text-align: right;
    }}
    th {{
        background: #e0e7ff;
        text-align: center;
    }}
    td:first-child, th:first-child {{
        text-align: left;
    }}
    .data-table.compact td {{
        padding: 6px 8px;
    }}
    .empty {{
        font-style: italic;
        color: #6b7280;
    }}
    footer {{
        padding: 16px 20px 32px;
        text-align: center;
        color: #6b7280;
        font-size: 0.9rem;
    }}
</style>
</head>
<body>
<header>
    <h1>{title}</h1>
    <p>{summary}</p>
    <p class=\"generated\">Generated on {generated_at}</p>
</header>
<main>
    <section>
        <h2>Scenario Frequency Overview</h2>
        <div class=\"chart\">{counts_chart}</div>
        <div class=\"table-wrapper\">{counts_table}</div>
    </section>
    <section>
        <h2>Parameter Distributions</h2>
        {scenario_sections}
    </section>
    <section>
        <h2>Sample Events</h2>
        <div class=\"table-wrapper\">{events_table}</div>
    </section>
</main>
<footer>
    Source directory: {output_directory}
</footer>
</body>
</html>
"""


def generate_report(
    output_dir: Path,
    *,
    html_path: Path | None = None,
    title: str = "HighD Scenario Report",
) -> Path:
    """Generate an interactive HTML report from CLI output files."""

    output_dir = Path(output_dir)
    counts_df = _read_counts(output_dir)
    distributions = _read_distributions(output_dir)
    events_df = _read_events(output_dir)

    counts_chart, counts_table, js_loaded = _render_counts_section(counts_df)
    scenario_sections, js_loaded = _render_distribution_sections(
        distributions, events_df, js_loaded
    )
    events_table = _render_events_table(events_df)

    if "count" in counts_df.columns:
        total_events = int(counts_df["count"].sum())
    else:
        total_events = len(events_df)
    if "scenario" in counts_df.columns and not counts_df.empty:
        scenario_count = int(counts_df["scenario"].nunique())
    else:
        scenario_count = len(distributions)
    summary_text = (
        f"Detected <strong>{total_events}</strong> events spanning "
        f"<strong>{scenario_count}</strong> scenarios."
    )

    html_output = HTML_TEMPLATE.format(
        title=escape(title),
        summary=summary_text,
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        counts_chart=counts_chart,
        counts_table=counts_table,
        scenario_sections=scenario_sections,
        events_table=events_table,
        output_directory=escape(str(output_dir.resolve())),
    )

    destination = Path(html_path) if html_path is not None else (output_dir / DEFAULT_REPORT_NAME)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(html_output, encoding="utf-8")
    return destination


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate an interactive HTML report from scenario outputs."
    )
    parser.add_argument(
        "--outputs",
        default="outputs",
        help="Directory containing scenario_events.csv, scenario_counts.csv, and parameter_distributions.json.",
    )
    parser.add_argument(
        "--html",
        default=None,
        help="Path to the generated HTML file (defaults to <outputs>/scenario_report.html).",
    )
    parser.add_argument(
        "--title",
        default="HighD Scenario Report",
        help="Title displayed at the top of the report.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    output_dir = Path(args.outputs)
    if not output_dir.exists():
        parser.error(f"Output directory {output_dir} does not exist.")

    html_path = Path(args.html) if args.html else None
    report_path = generate_report(output_dir, html_path=html_path, title=args.title)
    print(f"Report written to {report_path}")


__all__ = ["generate_report", "build_argument_parser", "main"]
