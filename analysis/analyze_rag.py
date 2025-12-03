#!/usr/bin/env python3
"""Analyse RAG benchmark outputs (scores, timings) and emit Markdown + charts."""

from __future__ import annotations

import argparse
import json
import os
import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt


@dataclass
class RAGRunStats:
    name: str
    num_cases: int
    avg_score: float
    median_score: float
    avg_elapsed: float
    case_type_avgs: Dict[str, float]
    case_type_counts: Dict[str, int]
    scores: List[float]
    cases: List[dict]


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_results = repo_root / "benchmark_results"
    parser = argparse.ArgumentParser(
        description="Generate Markdown + charts for RAG evaluations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results-root", type=Path, default=default_results)
    parser.add_argument("--runs", nargs="*", default=None)
    parser.add_argument("--prefix", default="RAG")
    parser.add_argument(
        "--output-md",
        type=Path,
        default=default_results / "rag_analysis.md",
        help="Markdown file to write.",
    )
    parser.add_argument(
        "--chart-dir",
        type=Path,
        default=default_results / "charts",
        help="Where generated PNG charts will be written.",
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
        raise ValueError("No run directories discovered. Pass --runs or adjust --prefix.")
    missing = [str(p) for p in run_dirs if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing run directories: {missing}")
    return run_dirs


def load_cases(run_dir: Path) -> list[dict]:
    for candidate in ("detailed_results.json", "ablation_results.json"):
        json_path = run_dir / candidate
        if json_path.exists():
            with json_path.open(encoding="utf-8") as fh:
                return json.load(fh)
    raise FileNotFoundError(f"No JSON results found in {run_dir}")


def summarise_run(run_dir: Path) -> RAGRunStats:
    cases = load_cases(run_dir)
    scores = [float(case.get("score", 0.0)) for case in cases]
    elapsed = [float(case.get("elapsed_time", 0.0)) for case in cases]
    case_type_values: dict[str, list[float]] = defaultdict(list)
    case_type_counts: dict[str, int] = defaultdict(int)
    for case, score in zip(cases, scores, strict=False):
        case_type = case.get("case_type", "Unspecified")
        case_type_values[case_type].append(score)
        case_type_counts[case_type] += 1

    case_type_avgs = {ctype: statistics.fmean(values) for ctype, values in case_type_values.items()}

    return RAGRunStats(
        name=run_dir.name,
        num_cases=len(cases),
        avg_score=statistics.fmean(scores) if scores else 0.0,
        median_score=statistics.median(scores) if scores else 0.0,
        avg_elapsed=statistics.fmean(elapsed) if elapsed else 0.0,
        case_type_avgs=case_type_avgs,
        case_type_counts=case_type_counts,
        scores=scores,
        cases=cases,
    )


def plot_average_scores(stats: Iterable[RAGRunStats], chart_path: Path, dpi: int) -> None:
    stats = list(stats)
    fig, ax = plt.subplots(figsize=(max(6, len(stats) * 1.6), 4))
    names = [s.name for s in stats]
    values = [s.avg_score for s in stats]
    bars = ax.bar(names, values, color="#55a868")
    ax.set_ylabel("Average score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Average RAG score by run")
    for bar, value in zip(bars, values, strict=False):
        ax.annotate(f"{value:.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, value),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(chart_path, dpi=dpi)
    plt.close(fig)


def plot_case_type_scores(stats: Iterable[RAGRunStats], chart_path: Path, dpi: int) -> None:
    stats = list(stats)
    all_case_types = sorted({ctype for s in stats for ctype in s.case_type_avgs.keys()})
    if not all_case_types:
        chart_path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.text(0.5, 0.5, "No case type data", ha="center", va="center")
        ax.axis("off")
        fig.savefig(chart_path, dpi=dpi)
        plt.close(fig)
        return
    fig, ax = plt.subplots(figsize=(max(6, len(stats) * 1.6), 4 + 0.2 * len(all_case_types)))
    x = range(len(stats))
    width = 0.8 / max(1, len(all_case_types))
    for idx, ctype in enumerate(all_case_types):
        offsets = [val + idx * width for val in x]
        values = [s.case_type_avgs.get(ctype, 0.0) for s in stats]
        ax.bar([o - 0.4 + width / 2 for o in offsets], values, width, label=ctype)
    ax.set_xticks(list(x), [s.name for s in stats])
    ax.set_ylabel("Average score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Average score per case type")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(chart_path, dpi=dpi)
    plt.close(fig)


def plot_score_boxplot(stats: Iterable[RAGRunStats], chart_path: Path, dpi: int) -> None:
    stats = list(stats)
    fig, ax = plt.subplots(figsize=(max(6, len(stats) * 1.6), 4))
    data = [s.scores for s in stats]
    ax.boxplot(data, labels=[s.name for s in stats], showmeans=True)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Score distribution per run")
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


def build_markdown(stats: List[RAGRunStats], charts: dict[str, Path]) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = ["# RAG Evaluation Analysis", "", f"_Generated {timestamp}_", ""]

    header = "| Run | Cases | Avg Score | Median | Avg Time (s) | Case type coverage |"
    sep = "| --- | --- | --- | --- | --- | --- |"
    rows = []
    for s in stats:
        coverage = ", ".join(
            f"{ctype}: {s.case_type_avgs.get(ctype, 0):.2f} ({s.case_type_counts.get(ctype, 0)} cases)"
            for ctype in sorted(s.case_type_avgs.keys())
        ) or "n/a"
        rows.append(
            f"| {s.name} | {s.num_cases} | {s.avg_score:.2f} | {s.median_score:.2f} | {s.avg_elapsed:.1f} | {coverage} |"
        )
    lines.extend(["## Run overview", "", header, sep, *rows, ""])

    best = max(stats, key=lambda s: s.avg_score)
    slowest = max(stats, key=lambda s: s.avg_elapsed)
    fastest = min(stats, key=lambda s: s.avg_elapsed)
    lines.append("## Highlights")
    lines.append("")
    lines.append(f"- **Highest average score**: `{best.name}` ({best.avg_score:.2f}).")
    lines.append(f"- **Fastest responses**: `{fastest.name}` average {fastest.avg_elapsed:.1f}s.")
    lines.append(f"- **Slowest responses**: `{slowest.name}` average {slowest.avg_elapsed:.1f}s.")
    lines.append("")

    for title, path in charts.items():
        lines.append(f"## {title}")
        lines.append("")
        lines.append(f"![{title}]({path.as_posix()})")
        lines.append("")

    all_cases = [case | {"__run": stat.name} for stat in stats for case in stat.cases]
    worst_cases = sorted(all_cases, key=lambda c: c.get("score", 0))[:5]
    if worst_cases:
        lines.append("## Lowest scoring prompts")
        lines.append("")
        lines.append("| Run | Case ID | Type | Score | Summary |")
        lines.append("| --- | --- | --- | --- | --- |")
        for case in worst_cases:
            reason = (case.get('score_reason') or '').splitlines()
            snippet = reason[0][:80] if reason else ""
            lines.append(
                f"| {case['__run']} | {case.get('case_id','?')} | {case.get('case_type','?')} | {case.get('score',0):.2f} | "
                f"{snippet} |"
            )
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    run_dirs = discover_runs(args.results_root, args.runs, args.prefix)
    stats = [summarise_run(run_dir) for run_dir in run_dirs]

    charts = {}
    avg_chart = args.chart_dir / "rag_average_scores.png"
    plot_average_scores(stats, avg_chart, args.dpi)
    charts["Average scores"] = _path_for_markdown(avg_chart, args.output_md.parent)

    case_type_chart = args.chart_dir / "rag_case_type_scores.png"
    plot_case_type_scores(stats, case_type_chart, args.dpi)
    charts["Case type averages"] = _path_for_markdown(case_type_chart, args.output_md.parent)

    box_chart = args.chart_dir / "rag_score_distribution.png"
    plot_score_boxplot(stats, box_chart, args.dpi)
    charts["Score distribution"] = _path_for_markdown(box_chart, args.output_md.parent)

    markdown = build_markdown(stats, charts)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(markdown, encoding="utf-8")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
