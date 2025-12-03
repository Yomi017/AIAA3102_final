#!/usr/bin/env python3
"""Analyse web / tool-evaluation benchmarks and output Markdown reports."""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt


AI_SCORE_RE = re.compile(r"AI评分\((?P<score>[0-9.]+)\)")
HUMAN_SCORE_RE = re.compile(r"人工评分[:：]\s*(?P<score>[0-9.]+)")


@dataclass
class WebRunStats:
    name: str
    num_cases: int
    avg_score: float
    avg_ai_score: Optional[float]
    avg_human_score: Optional[float]
    avg_elapsed: float
    fail_reasons: Counter
    scores: List[float]


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_results = repo_root / "benchmark_results"
    parser = argparse.ArgumentParser(
        description="Generate Markdown + charts for web benchmark runs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results-root", type=Path, default=default_results)
    parser.add_argument("--runs", nargs="*", default=None)
    parser.add_argument("--prefix", default="web")
    parser.add_argument(
        "--output-md",
        type=Path,
        default=default_results / "web_analysis.md",
        help="Markdown file destination.",
    )
    parser.add_argument(
        "--chart-dir",
        type=Path,
        default=default_results / "charts",
        help="Where PNG charts are stored.",
    )
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


def discover_runs(results_root: Path, runs: List[str] | None, prefix: str) -> List[Path]:
    if runs:
        run_dirs = [results_root / r for r in runs]
    else:
        run_dirs = [
            p for p in sorted(results_root.iterdir()) if p.is_dir() and p.name.lower().startswith(prefix.lower())
        ]
    if not run_dirs:
        raise ValueError("No run directories discovered. Pass --runs or change --prefix.")
    missing = [str(p) for p in run_dirs if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing run folders: {missing}")
    return run_dirs


def load_cases(run_dir: Path) -> list[dict]:
    for candidate in ("detailed_results.json", "ablation_results.json"):
        json_path = run_dir / candidate
        if json_path.exists():
            with json_path.open(encoding="utf-8") as fh:
                return json.load(fh)
    raise FileNotFoundError(f"No JSON results found in {run_dir}")


def parse_score_reason(text: str) -> tuple[Optional[float], Optional[float]]:
    if not text:
        return None, None
    ai_match = AI_SCORE_RE.search(text)
    human_match = HUMAN_SCORE_RE.search(text)
    ai_score = float(ai_match.group("score")) if ai_match else None
    human_score = float(human_match.group("score")) if human_match else None
    return ai_score, human_score


def summarise_run(run_dir: Path) -> WebRunStats:
    cases = load_cases(run_dir)
    scores = [float(case.get("score", 0.0)) for case in cases]
    elapsed = [float(case.get("elapsed_time", 0.0)) for case in cases]
    ai_scores: list[float] = []
    human_scores: list[float] = []
    fail_reasons: Counter = Counter()

    for case in cases:
        ai_score, human_score = parse_score_reason(case.get("score_reason", ""))
        if ai_score is not None:
            ai_scores.append(ai_score)
        if human_score is not None:
            human_scores.append(human_score)
        if case.get("score", 0.0) < 0.7:
            reason_text = (case.get("score_reason") or case.get("error") or "").splitlines()
            fail_reasons[reason_text[0] if reason_text else "Unspecified issue"] += 1

    avg_ai = statistics.fmean(ai_scores) if ai_scores else None
    avg_human = statistics.fmean(human_scores) if human_scores else None

    return WebRunStats(
        name=run_dir.name,
        num_cases=len(cases),
        avg_score=statistics.fmean(scores) if scores else 0.0,
        avg_ai_score=avg_ai,
        avg_human_score=avg_human,
        avg_elapsed=statistics.fmean(elapsed) if elapsed else 0.0,
        fail_reasons=fail_reasons,
        scores=scores,
    )


def plot_average_scores(stats: Iterable[WebRunStats], chart_path: Path, dpi: int) -> None:
    stats = list(stats)
    fig, ax = plt.subplots(figsize=(max(6, len(stats) * 1.6), 4))
    names = [s.name for s in stats]
    values = [s.avg_score for s in stats]
    ax.bar(names, values, color="#c44e52")
    ax.set_ylabel("Average score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Average web benchmark score")
    for idx, value in enumerate(values):
        ax.annotate(f"{value:.2f}", xy=(idx, value), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(chart_path, dpi=dpi)
    plt.close(fig)


def plot_ai_vs_human(stats: Iterable[WebRunStats], chart_path: Path, dpi: int) -> None:
    stats = list(stats)
    fig, ax = plt.subplots(figsize=(max(6, len(stats) * 1.6), 4))
    x = range(len(stats))
    width = 0.35
    ai_values = [s.avg_ai_score or 0.0 for s in stats]
    human_values = [s.avg_human_score or 0.0 for s in stats]
    ax.bar([i - width / 2 for i in x], ai_values, width, label="AI score", color="#4c72b0")
    ax.bar([i + width / 2 for i in x], human_values, width, label="Human score", color="#55a868")
    ax.set_xticks(list(x), [s.name for s in stats])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("AI vs human average scores")
    ax.legend()
    fig.tight_layout()
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(chart_path, dpi=dpi)
    plt.close(fig)


def plot_elapsed_times(stats: Iterable[WebRunStats], chart_path: Path, dpi: int) -> None:
    stats = list(stats)
    fig, ax = plt.subplots(figsize=(max(6, len(stats) * 1.6), 4))
    names = [s.name for s in stats]
    values = [s.avg_elapsed for s in stats]
    ax.bar(names, values, color="#8172b3")
    ax.set_ylabel("Avg elapsed time (s)")
    ax.set_title("Response latency")
    fig.tight_layout()
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(chart_path, dpi=dpi)
    plt.close(fig)


def _path_for_markdown(asset: Path, md_parent: Path) -> Path:
    try:
        return asset.relative_to(md_parent)
    except ValueError:
        rel = os.path.relpath(asset, md_parent)
        return Path(rel)


def build_markdown(stats: List[WebRunStats], charts: Dict[str, Path]) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = ["# Web Benchmark Analysis", "", f"_Generated {timestamp}_", ""]

    header = "| Run | Cases | Avg score | Avg AI score | Avg human score | Avg time (s) |"
    sep = "| --- | --- | --- | --- | --- | --- |"
    rows = []
    for s in stats:
        ai_str = f"{s.avg_ai_score:.2f}" if s.avg_ai_score is not None else "n/a"
        human_str = f"{s.avg_human_score:.2f}" if s.avg_human_score is not None else "n/a"
        rows.append(
            f"| {s.name} | {s.num_cases} | {s.avg_score:.2f} | {ai_str} | {human_str} | {s.avg_elapsed:.1f} |"
        )
    lines.extend(["## Run overview", "", header, sep, *rows, ""])

    best = max(stats, key=lambda s: s.avg_score)
    most_consistent = min(stats, key=lambda s: statistics.pstdev(s.scores) if len(s.scores) > 1 else 0.0)
    weakest = min(stats, key=lambda s: s.avg_score)
    lines.append("## Highlights")
    lines.append("")
    lines.append(f"- **Highest average score**: `{best.name}` at {best.avg_score:.2f}.")
    lines.append(f"- **Most consistent**: `{most_consistent.name}` had the lowest score variance.")
    lines.append(f"- **Needs attention**: `{weakest.name}` average {weakest.avg_score:.2f}.")
    lines.append("")

    for title, path in charts.items():
        lines.append(f"## {title}")
        lines.append("")
        lines.append(f"![{title}]({path.as_posix()})")
        lines.append("")

    aggregated_failures = Counter()
    for s in stats:
        aggregated_failures.update(s.fail_reasons)
    if aggregated_failures:
        lines.append("## Frequent failure reasons (score < 0.7)")
        lines.append("")
        lines.append("| Reason | Count |")
        lines.append("| --- | --- |")
        for reason, count in aggregated_failures.most_common():
            lines.append(f"| {reason[:90]} | {count} |")
        lines.append("")
    else:
        lines.append("No failures below the 0.7 threshold were recorded.")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    run_dirs = discover_runs(args.results_root, args.runs, args.prefix)
    stats = [summarise_run(run_dir) for run_dir in run_dirs]

    charts: Dict[str, Path] = {}
    avg_chart = args.chart_dir / "web_average_scores.png"
    plot_average_scores(stats, avg_chart, args.dpi)
    charts["Average scores"] = _path_for_markdown(avg_chart, args.output_md.parent)

    ai_chart = args.chart_dir / "web_ai_vs_human.png"
    plot_ai_vs_human(stats, ai_chart, args.dpi)
    charts["AI vs human"] = _path_for_markdown(ai_chart, args.output_md.parent)

    elapsed_chart = args.chart_dir / "web_elapsed.png"
    plot_elapsed_times(stats, elapsed_chart, args.dpi)
    charts["Latency"] = _path_for_markdown(elapsed_chart, args.output_md.parent)

    markdown = build_markdown(stats, charts)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(markdown, encoding="utf-8")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
