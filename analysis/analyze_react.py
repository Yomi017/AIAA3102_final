#!/usr/bin/env python3
"""Aggregate ALFWorld / React benchmark runs and produce Markdown + charts.

The script scans benchmark result folders (for example `benchmark_results/React`
and `benchmark_results/React_baseline`), computes success metrics, renders
comparison charts, and writes a Markdown summary file.  It is intentionally
parameterised so additional runs can be analysed without touching the code.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt


@dataclass
class RunStats:
    name: str
    total_games: int
    successes: int
    avg_steps: float
    avg_success_steps: float
    avg_action_success: float
    failure_reasons: Counter

    @property
    def success_rate(self) -> float:
        if self.total_games == 0:
            return 0.0
        return self.successes / self.total_games

    @property
    def failure_count(self) -> int:
        return sum(self.failure_reasons.values())


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_results = repo_root / "benchmark_results"
    parser = argparse.ArgumentParser(
        description="Generate Markdown + charts for React/ALFWorld runs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=default_results,
        help="Folder that contains per-run subdirectories",
    )
    parser.add_argument(
        "--runs",
        nargs="*",
        default=None,
        help="Explicit sub-folders to analyse (relative to --results-root).",
    )
    parser.add_argument(
        "--prefix",
        default="React",
        help="Fallback prefix for auto-discovery when --runs is omitted.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=default_results / "react_analysis.md",
        help="Markdown file to write.",
    )
    parser.add_argument(
        "--chart-dir",
        type=Path,
        default=default_results / "charts",
        help="Directory where PNG charts will be stored.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Resolution for generated charts.",
    )
    return parser.parse_args()


def discover_runs(results_root: Path, runs: List[str] | None, prefix: str) -> List[Path]:
    if runs:
        run_dirs = [results_root / r for r in runs]
    else:
        run_dirs = [
            p for p in sorted(results_root.iterdir()) if p.is_dir() and p.name.lower().startswith(prefix.lower())
        ]
    missing = [str(p) for p in run_dirs if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Could not find run folders: {missing}")
    if not run_dirs:
        raise ValueError("No run directories discovered. Pass --runs or adjust --prefix.")
    return run_dirs


def load_csv_rows(csv_path: Path) -> list[dict]:
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return list(reader)


def normalise_bool(value: str) -> bool:
    return value.strip().lower() in {"✅", "true", "1", "yes", "success"}


def normalise_percent(value: str) -> float:
    value = value.strip().replace("%", "")
    return float(value) if value else 0.0


def normalise_reason(text: str) -> str:
    if not text:
        return "Unspecified failure"
    cleaned = text.strip()
    if cleaned.lower().startswith("error:"):
        cleaned = cleaned.split(":", 1)[1].strip() or "Error"
    return cleaned[0].upper() + cleaned[1:] if cleaned else "Unspecified failure"


def summarise_run(run_dir: Path) -> RunStats:
    csv_path = run_dir / "detailed_results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Expected file missing: {csv_path}")
    rows = load_csv_rows(csv_path)
    total_games = len(rows)
    successes = 0
    steps: list[int] = []
    successful_steps: list[int] = []
    action_success_rates: list[float] = []
    failures = Counter()

    for row in rows:
        success = normalise_bool(row.get("success", ""))
        successes += int(success)
        steps.append(int(row.get("steps", 0)))
        successful_steps.append(int(row.get("successful_steps", 0)))
        action_success_rates.append(normalise_percent(row.get("action_success_rate", "0")))
        if not success:
            failures[normalise_reason(row.get("task_preview", ""))] += 1

    avg_steps = statistics.fmean(steps) if steps else 0.0
    avg_success_steps = statistics.fmean(successful_steps) if successful_steps else 0.0
    avg_action_success = statistics.fmean(action_success_rates) if action_success_rates else 0.0

    return RunStats(
        name=run_dir.name,
        total_games=total_games,
        successes=successes,
        avg_steps=avg_steps,
        avg_success_steps=avg_success_steps,
        avg_action_success=avg_action_success,
        failure_reasons=failures,
    )


def plot_success_rates(stats: Iterable[RunStats], chart_path: Path, dpi: int) -> None:
    stats = list(stats)
    fig, ax = plt.subplots(figsize=(max(6, len(stats) * 1.8), 4))
    names = [s.name for s in stats]
    values = [s.success_rate * 100 for s in stats]
    bars = ax.bar(names, values, color="#4c72b0")
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(0, 100)
    ax.set_title("ALFWorld success rate by run")
    for bar, value in zip(bars, values, strict=False):
        ax.annotate(
            f"{value:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(chart_path, dpi=dpi)
    plt.close(fig)


def plot_failure_reasons(stats: Iterable[RunStats], chart_path: Path, dpi: int) -> None:
    total_failures = Counter()
    for run_stats in stats:
        total_failures.update(run_stats.failure_reasons)
    if not total_failures:
        chart_path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "No failures recorded", ha="center", va="center", fontsize=12)
        ax.axis("off")
        fig.savefig(chart_path, dpi=dpi)
        plt.close(fig)
        return

    reasons, counts = zip(*total_failures.most_common())
    fig_height = max(3, 0.4 * len(reasons))
    fig, ax = plt.subplots(figsize=(8, fig_height))
    ax.barh(range(len(reasons)), counts, color="#dd8452")
    ax.set_yticks(range(len(reasons)), labels=reasons)
    ax.set_xlabel("Failure count")
    ax.set_title("Failure reasons across runs")
    fig.tight_layout()
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(chart_path, dpi=dpi)
    plt.close(fig)


def build_markdown(stats: List[RunStats], success_chart: Path, failure_chart: Path) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = ["# React / ALFWorld Benchmark Analysis", ""]
    lines.append(f"_Generated {timestamp}_")
    lines.append("")

    header = "| Run | Total Games | Success Rate | Avg Steps | Avg Successful Steps | Avg Action Accuracy | Failures |"
    sep = "| --- | --- | --- | --- | --- | --- | --- |"
    rows = []
    for s in stats:
        rows.append(
            f"| {s.name} | {s.total_games} | {s.success_rate*100:.2f}% | {s.avg_steps:.2f} | "
            f"{s.avg_success_steps:.2f} | {s.avg_action_success:.2f}% | {s.failure_count} |"
        )
    lines.extend(["## Run overview", "", header, sep, *rows, ""])

    best = max(stats, key=lambda s: s.success_rate)
    worst = min(stats, key=lambda s: s.success_rate)
    lines.append("## Highlights")
    lines.append("")
    lines.append(f"- **Best run**: `{best.name}` with {best.success_rate*100:.1f}% success rate.")
    lines.append(f"- **Most reliable actions**: `{max(stats, key=lambda s: s.avg_action_success).name}` averaging "
                 f"{max(stats, key=lambda s: s.avg_action_success).avg_action_success:.2f}% action accuracy.")
    lines.append(f"- **Needs attention**: `{worst.name}` success rate {worst.success_rate*100:.1f}%.")
    lines.append("")

    lines.append("## Success rate comparison")
    lines.append("")
    lines.append(f"![Success rate chart]({success_chart.as_posix()})")
    lines.append("")

    lines.append("## Failure reasons")
    lines.append("")
    lines.append(f"![Failure reason chart]({failure_chart.as_posix()})")
    lines.append("")

    total_failure_counter = Counter()
    for s in stats:
        total_failure_counter.update(s.failure_reasons)
    if total_failure_counter:
        lines.append("### Breakdown")
        lines.append("")
        lines.append("| Reason | Count |")
        lines.append("| --- | --- |")
        for reason, count in total_failure_counter.most_common():
            lines.append(f"| {reason} | {count} |")
        lines.append("")
    else:
        lines.append("No failures recorded across the analysed runs.")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    run_dirs = discover_runs(args.results_root, args.runs, args.prefix)
    run_stats = [summarise_run(run_dir) for run_dir in run_dirs]

    # Charts
    success_chart = args.chart_dir / "react_success_rates.png"
    failure_chart = args.chart_dir / "react_failures.png"
    plot_success_rates(run_stats, success_chart, args.dpi)
    plot_failure_reasons(run_stats, failure_chart, args.dpi)

    def _path_for_markdown(asset: Path) -> Path:
        try:
            return asset.relative_to(args.output_md.parent)
        except ValueError:
            rel = os.path.relpath(asset, args.output_md.parent)
            return Path(rel)

    # Markdown
    markdown = build_markdown(
        run_stats,
        _path_for_markdown(success_chart),
        _path_for_markdown(failure_chart),
    )
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(markdown, encoding="utf-8")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
