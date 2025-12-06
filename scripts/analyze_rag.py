#!/usr/bin/env python3
"""Compare RAG vs baseline benchmark results and emit markdown/CSV reports.

Expected layout (per model):

benchmark_results/
    RAG/
        session_xxx/detailed_results.json
        session_yyy/detailed_results.json
    RAG_baseline/
        session_aaa/detailed_results.json
        session_bbb/detailed_results.json

All sessions under a model are concatenated to form the model's aggregate
results before comparison. If a direct JSON file is passed, it will be used as
is.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:  # matplotlib is optional; charts are skipped if unavailable
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


@dataclass
class Case:
    case_id: str
    case_type: str
    score: float
    elapsed_time: float

    @staticmethod
    def from_json(obj: Dict) -> "Case":
        return Case(
            case_id=str(obj["case_id"]),
            case_type=str(obj.get("case_type", "unknown")),
            score=float(obj.get("score", math.nan)),
            elapsed_time=float(obj.get("elapsed_time", math.nan)),
        )


def load_cases(path: Path) -> List[Case]:
    """Load cases from a JSON file or by aggregating all session files in a dir."""

    def load_file(json_path: Path) -> List[Case]:
        data = json.loads(json_path.read_text())
        return [Case.from_json(item) for item in data]

    if path.is_file():
        return load_file(path)

    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")

    # Collect all detailed/ablation result files under the directory.
    candidates = list(path.rglob("detailed_results.json")) + list(path.rglob("ablation_results.json"))
    candidates = sorted({p.resolve() for p in candidates})
    if not candidates:
        raise FileNotFoundError(f"No detailed_results.json or ablation_results.json found under {path}")

    cases: List[Case] = []
    for json_path in candidates:
        cases.extend(load_file(json_path))
    return cases


def summary_stats(cases: Sequence[Case]) -> Dict[str, float]:
    scores = [c.score for c in cases if not math.isnan(c.score)]
    times = [c.elapsed_time for c in cases if not math.isnan(c.elapsed_time)]
    return {
        "count": len(cases),
        "score_mean": statistics.fmean(scores) if scores else math.nan,
        "score_median": statistics.median(scores) if scores else math.nan,
        "time_mean": statistics.fmean(times) if times else math.nan,
    }


def group_by_type(cases: Sequence[Case]) -> Dict[str, List[Case]]:
    grouped: Dict[str, List[Case]] = {}
    for case in cases:
        grouped.setdefault(case.case_type, []).append(case)
    return grouped


def best_worst(cases: Sequence[Case], n: int = 3) -> Tuple[List[Case], List[Case]]:
    ordered = sorted(cases, key=lambda c: c.score)
    return ordered[:n], ordered[-n:][::-1]


def align_cases(a: Sequence[Case], b: Sequence[Case]) -> List[Tuple[Case, Case]]:
    left = {c.case_id: c for c in a}
    right = {c.case_id: c for c in b}
    common_ids = sorted(set(left) & set(right))
    return [(left[i], right[i]) for i in common_ids]


def fmt_float(val: float) -> str:
    return "nan" if math.isnan(val) else f"{val:.2f}"


def build_markdown(
    rag_cases: Sequence[Case],
    base_cases: Sequence[Case],
    out_csv: Optional[Path],
) -> str:
    rag_stats = summary_stats(rag_cases)
    base_stats = summary_stats(base_cases)

    lines: List[str] = []
    lines.append("# RAG vs Baseline Analysis")
    lines.append("")
    lines.append("## Overall")
    lines.append("| Model | Cases | Mean Score | Median Score | Mean Time (s) |")
    lines.append("| --- | --- | --- | --- | --- |")
    lines.append(
        f"| RAG | {rag_stats['count']} | {fmt_float(rag_stats['score_mean'])} | "
        f"{fmt_float(rag_stats['score_median'])} | {fmt_float(rag_stats['time_mean'])} |"
    )
    lines.append(
        f"| Baseline | {base_stats['count']} | {fmt_float(base_stats['score_mean'])} | "
        f"{fmt_float(base_stats['score_median'])} | {fmt_float(base_stats['time_mean'])} |"
    )
    lines.append("")

    lines.append("## By Case Type")
    lines.append("| Model | Case Type | Cases | Mean Score | Mean Time (s) |")
    lines.append("| --- | --- | --- | --- | --- |")
    for label, cases in [("RAG", rag_cases), ("Baseline", base_cases)]:
        for ctype, subset in group_by_type(cases).items():
            stats = summary_stats(subset)
            lines.append(
                f"| {label} | {ctype} | {stats['count']} | {fmt_float(stats['score_mean'])} | "
                f"{fmt_float(stats['time_mean'])} |"
            )
    lines.append("")

    rag_worst, rag_best = best_worst(rag_cases)
    lines.append("## RAG Extremes")
    lines.append("### Lowest Scores")
    lines.extend(render_case_table(rag_worst))
    lines.append("\n### Highest Scores")
    lines.extend(render_case_table(rag_best))
    lines.append("")

    pairs = align_cases(rag_cases, base_cases)
    lines.append("## Per-Case Comparison (RAG vs Baseline)")
    lines.append("| Case ID | Case Type | RAG Score | Baseline Score | Δ Score | RAG Time (s) | Baseline Time (s) | Δ Time |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    csv_rows: List[str] = [
        "case_id,case_type,rag_score,baseline_score,delta_score,rag_time,baseline_time,delta_time"
    ]
    for rag, base in pairs:
        d_score = rag.score - base.score
        d_time = rag.elapsed_time - base.elapsed_time
        lines.append(
            f"| {rag.case_id} | {rag.case_type} | {fmt_float(rag.score)} | {fmt_float(base.score)} | "
            f"{fmt_float(d_score)} | {fmt_float(rag.elapsed_time)} | {fmt_float(base.elapsed_time)} | {fmt_float(d_time)} |"
        )
        csv_rows.append(
            ",".join(
                [
                    rag.case_id,
                    rag.case_type,
                    f"{rag.score}",
                    f"{base.score}",
                    f"{d_score}",
                    f"{rag.elapsed_time}",
                    f"{base.elapsed_time}",
                    f"{d_time}",
                ]
            )
        )
    lines.append("")

    if out_csv:
        out_csv.write_text("\n".join(csv_rows))

    return "\n".join(lines)


def render_case_table(cases: Sequence[Case]) -> List[str]:
    rows: List[str] = ["| Case ID | Case Type | Score | Time (s) |", "| --- | --- | --- | --- |"]
    for c in cases:
        rows.append(
            f"| {c.case_id} | {c.case_type} | {fmt_float(c.score)} | {fmt_float(c.elapsed_time)} |"
        )
    return rows


def plot_charts(
    rag_cases: Sequence[Case],
    base_cases: Sequence[Case],
    charts_dir: Path,
) -> None:
    if plt is None:  # pragma: no cover
        return
    charts_dir.mkdir(parents=True, exist_ok=True)

    def savefig(name: str) -> None:
        plt.tight_layout()
        plt.savefig(charts_dir / name, dpi=150)
        plt.close()

    # Average score per model
    plt.bar(["RAG", "Baseline"], [summary_stats(rag_cases)["score_mean"], summary_stats(base_cases)["score_mean"]])
    plt.ylabel("Mean score")
    plt.autoscale(enable=True, axis="y", tight=True)
    savefig("rag_avg_scores.png")

    # Score delta per case
    pairs = align_cases(rag_cases, base_cases)
    ids = [p[0].case_id for p in pairs]
    deltas = [p[0].score - p[1].score for p in pairs]
    plt.bar(ids, deltas)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Score delta (RAG - Baseline)")
    plt.autoscale(enable=True, axis="y", tight=True)
    savefig("rag_score_deltas.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rag", type=Path, default=Path("benchmark_results/RAG"), help="RAG results dir (or single JSON file)")
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("benchmark_results/RAG_baseline"),
        help="Baseline results dir (or single JSON file)",
    )
    parser.add_argument("--out-md", type=Path, default=Path("benchmark_results/rag_reanalysis.md"))
    parser.add_argument("--out-csv", type=Path, default=Path("benchmark_results/rag_case_comparison.csv"))
    parser.add_argument("--charts-dir", type=Path, default=Path("benchmark_results/charts"))
    parser.add_argument("--no-charts", action="store_true", help="Skip chart generation even if matplotlib is available")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rag_cases = load_cases(args.rag)
    base_cases = load_cases(args.baseline)

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)

    markdown = build_markdown(rag_cases, base_cases, args.out_csv)
    args.out_md.write_text(markdown)

    if not args.no_charts:
        plot_charts(rag_cases, base_cases, args.charts_dir)

    print(f"Wrote markdown to {args.out_md}")
    if args.out_csv:
        print(f"Wrote CSV to {args.out_csv}")
    if not args.no_charts and plt is None:
        print("matplotlib not installed; charts skipped")


if __name__ == "__main__":
    main()
