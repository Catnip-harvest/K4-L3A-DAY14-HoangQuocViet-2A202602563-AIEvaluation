"""Noise floor and lab-heuristic comparison for the Exercise 3.4 bonus.

Reads two runs of ``compare_frameworks.py`` (the same judge, the same dataset,
run twice) and the lab's word-overlap scores in ``benchmark_results.json``, and
writes ``artifacts/framework_noise.json`` with:

* the run-to-run noise floor per framework and metric: the mean absolute
  difference between the two runs, and the share of cases whose pass/fail flag
  (score < 0.5) flips between them;
* the between-framework Pearson/Spearman correlation per metric for run 1,
  run 2 and the mean of both runs, to read against that noise floor;
* each framework's correlation with the lab's heuristic scores, and how the
  cases each framework flags compare with the lab's own failures.

Standard library only; it never calls an API. Run from the repository root:

    python bonus/compare_runs.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = REPO_ROOT / "artifacts"

FRAMEWORKS = ("ragas", "deepeval")
METRICS = ("faithfulness", "answer_relevancy", "context_recall", "context_precision")
# The lab's score name for each framework metric.
LAB_METRIC_NAMES = {
    "faithfulness": "faithfulness",
    "answer_relevancy": "relevance",
    "context_recall": "context_recall",
    "context_precision": "context_precision",
}
FAIL_BELOW = 0.5
MIN_PAIRS = 3

Scores = dict[str, float | None]  # case id -> score


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"not valid JSON: {path} ({exc.msg})") from exc
    if not isinstance(value, dict):
        raise ValueError(f"root must be a JSON object: {path}")
    return value


def run_scores(run: dict[str, Any]) -> dict[str, dict[str, Scores]]:
    """Return scores[framework][metric][case_id] from one comparison run."""

    scores: dict[str, dict[str, Scores]] = {
        framework: {metric: {} for metric in METRICS} for framework in FRAMEWORKS
    }
    for row in run["cases"]:
        for framework in FRAMEWORKS:
            for metric in METRICS:
                scores[framework][metric][row["id"]] = row[framework][metric]["score"]
    return scores


def lab_scores(benchmark: dict[str, Any]) -> dict[str, Scores]:
    """Return the lab heuristic scores as lab[framework_metric][case_id]."""

    lab: dict[str, Scores] = {metric: {} for metric in METRICS}
    for row in benchmark["results"]:
        for metric, lab_name in LAB_METRIC_NAMES.items():
            lab[metric][row["id"]] = row.get(lab_name)
    return lab


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------


def _mean(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _rounded(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(value, digits)


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) < MIN_PAIRS:
        return None
    try:
        return statistics.correlation(xs, ys)
    except statistics.StatisticsError:  # one side is constant
        return None


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and values[order[end + 1]] == values[order[start]]:
            end += 1
        for tied in order[start : end + 1]:
            ranks[tied] = (start + end) / 2 + 1
        start = end + 1
    return ranks


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) < MIN_PAIRS:
        return None
    return pearson(_average_ranks(xs), _average_ranks(ys))


def _pairs(first: Scores, second: Scores) -> tuple[list[str], list[float], list[float]]:
    ids: list[str] = []
    xs: list[float] = []
    ys: list[float] = []
    for case_id, x in first.items():
        y = second.get(case_id)
        if x is not None and y is not None:
            ids.append(case_id)
            xs.append(x)
            ys.append(y)
    return ids, xs, ys


def _correlations(first: Scores, second: Scores) -> dict[str, Any]:
    _, xs, ys = _pairs(first, second)
    return {
        "n": len(xs),
        "pearson": _rounded(pearson(xs, ys)),
        "spearman": _rounded(spearman(xs, ys)),
    }


def mean_of_runs(first: Scores, second: Scores) -> Scores:
    """Per-case mean over whichever of the two runs produced a score."""

    merged: Scores = {}
    for case_id in first:
        present = [s for s in (first[case_id], second.get(case_id)) if s is not None]
        merged[case_id] = statistics.fmean(present) if present else None
    return merged


def is_failing(score: float | None) -> bool:
    return score is not None and score < FAIL_BELOW


# --------------------------------------------------------------------------
# Sections of the report
# --------------------------------------------------------------------------


def noise_floor(first: Scores, second: Scores) -> dict[str, Any]:
    ids, xs, ys = _pairs(first, second)
    differences = [abs(a - b) for a, b in zip(xs, ys, strict=True)]
    flips = [
        case_id
        for case_id, a, b in zip(ids, xs, ys, strict=True)
        if is_failing(a) != is_failing(b)
    ]
    return {
        "paired_n": len(ids),
        "mean_run1": _rounded(_mean(xs)),
        "mean_run2": _rounded(_mean(ys)),
        "mean_absolute_difference": _rounded(_mean(differences)),
        "max_absolute_difference": _rounded(max(differences, default=None)),
        "pass_fail_flip_count": len(flips),
        "pass_fail_flip_share": _rounded(len(flips) / len(ids) if ids else None),
        "pass_fail_flip_ids": flips,
        "run_to_run_pearson": _rounded(pearson(xs, ys)),
        "run_to_run_spearman": _rounded(spearman(xs, ys)),
    }


def flagged_ids(scores: dict[str, Scores]) -> list[str]:
    """Cases with at least one metric below the failure line, in case order."""

    case_ids = list(next(iter(scores.values())))
    return [c for c in case_ids if any(is_failing(scores[m][c]) for m in scores)]


def overlap(
    first_name: str, first: list[str], second_name: str, second: list[str]
) -> dict[str, list[str]]:
    return {
        "both": [i for i in first if i in second],
        f"{first_name}_only": [i for i in first if i not in second],
        f"{second_name}_only": [i for i in second if i not in first],
    }


def failing_cases(
    runs: dict[str, dict[str, dict[str, Scores]]],
    lab_failed: list[str],
    lab: dict[str, Scores],
) -> dict[str, Any]:
    flagged = {
        run_name: {fw: flagged_ids(scores[fw]) for fw in FRAMEWORKS}
        for run_name, scores in runs.items()
    }
    report: dict[str, Any] = {
        "rule": f"a case is flagged when any of its 4 metrics is < {FAIL_BELOW}",
        "lab_heuristic_failures": {"count": len(lab_failed), "ids": lab_failed},
        "by_run": {},
        "stable_across_runs": {},
        "per_metric_vs_lab": {},
    }
    for run_name, by_framework in flagged.items():
        report["by_run"][run_name] = {
            "ragas": by_framework["ragas"],
            "deepeval": by_framework["deepeval"],
            "ragas_vs_deepeval": overlap(
                "ragas", by_framework["ragas"], "deepeval", by_framework["deepeval"]
            ),
            "ragas_vs_lab": overlap(
                "ragas", by_framework["ragas"], "lab", lab_failed
            ),
            "deepeval_vs_lab": overlap(
                "deepeval", by_framework["deepeval"], "lab", lab_failed
            ),
        }
    for framework in FRAMEWORKS:
        run1_ids, run2_ids = flagged["run1"][framework], flagged["run2"][framework]
        report["stable_across_runs"][framework] = {
            "flagged_in_both_runs": [i for i in run1_ids if i in run2_ids],
            "flagged_in_one_run_only": sorted(set(run1_ids) ^ set(run2_ids)),
        }
    for metric in METRICS:
        lab_ids = [c for c, s in lab[metric].items() if is_failing(s)]
        entry: dict[str, Any] = {"lab": lab_ids}
        for framework in FRAMEWORKS:
            for run_name, scores in runs.items():
                ids = [c for c, s in scores[framework][metric].items() if is_failing(s)]
                entry[f"{framework}_{run_name}"] = ids
        report["per_metric_vs_lab"][metric] = entry
    return report


def build_report(
    run1: dict[str, Any],
    run2: dict[str, Any],
    benchmark: dict[str, Any],
    answers: dict[str, Any],
    paths: dict[str, str],
) -> dict[str, Any]:
    runs = {"run1": run_scores(run1), "run2": run_scores(run2)}
    runs_mean = {
        fw: {m: mean_of_runs(runs["run1"][fw][m], runs["run2"][fw][m]) for m in METRICS}
        for fw in FRAMEWORKS
    }
    views = {"run1": runs["run1"], "run2": runs["run2"], "mean_of_runs": runs_mean}
    lab = lab_scores(benchmark)
    lab_failed = [r["id"] for r in benchmark["results"] if not r["passed"]]

    between = {
        metric: {
            view: _correlations(scores["ragas"][metric], scores["deepeval"][metric])
            for view, scores in views.items()
        }
        for metric in METRICS
    }
    versus_lab = {
        framework: {
            metric: {
                view: _correlations(scores[framework][metric], lab[metric])
                for view, scores in views.items()
            }
            for metric in METRICS
        }
        for framework in FRAMEWORKS
    }
    generator = answers.get("agent", {})
    judge = run1["judge"]
    return {
        "schema_version": "1.0",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "inputs": {
            **paths,
            "case_count": run1["config"]["case_count"],
            "run1_generated_at": run1["generated_at"],
            "run2_generated_at": run2["generated_at"],
            "judge_model": judge["model"],
            "judge_reasoning_effort": judge["reasoning_effort"],
            "generator_model": generator.get("model"),
            "generator_reasoning_effort": generator.get("reasoning_effort"),
            "judge_is_generator_model": judge["model"] == generator.get("model"),
            "lab_summary": benchmark["summary"],
        },
        "noise_floor": {
            framework: {
                metric: noise_floor(
                    runs["run1"][framework][metric], runs["run2"][framework][metric]
                )
                for metric in METRICS
            }
            for framework in FRAMEWORKS
        },
        "between_frameworks": between,
        "vs_lab_heuristic": versus_lab,
        "failing_cases": failing_cases(runs, lab_failed, lab),
        "caveats": [
            "The judge cannot run at temperature 0 (only the default is accepted), "
            "so two identical runs differ; noise_floor measures that.",
            "The lab faithfulness scores the answer against the GOLD context, while "
            "both frameworks judge it against the RETRIEVED chunks: not like-for-like.",
            "The lab relevance is token overlap between answer and question; "
            "answer_relevancy in the frameworks is an LLM judgement: different method.",
            "The lab context_recall/context_precision use token overlap with the "
            "expected answer over the retrieved chunks: same inputs, different method.",
            "The lab pass rule also uses completeness, which has no counterpart in "
            "either framework's four metrics.",
            "The judge is the same model as the generator: self-preference risk.",
        ],
    }


# --------------------------------------------------------------------------
# Console output and command line
# --------------------------------------------------------------------------


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def print_summary(report: dict[str, Any]) -> None:
    print("Run-to-run noise floor (run1 vs run2, same judge):")
    print(
        f"  {'framework':<10}{'metric':<18}{'mean |diff|':>12}"
        f"{'flips':>8}{'pearson':>9}"
    )
    for framework in FRAMEWORKS:
        for metric in METRICS:
            cell = report["noise_floor"][framework][metric]
            flips = f"{cell['pass_fail_flip_count']}/{cell['paired_n']}"
            mean_diff = _fmt(cell["mean_absolute_difference"])
            print(
                f"  {framework:<10}{metric:<18}{mean_diff:>12}"
                f"{flips:>8}{_fmt(cell['run_to_run_pearson']):>9}"
            )
    print("\nBetween frameworks (ragas vs deepeval), pearson / spearman:")
    for metric in METRICS:
        parts = []
        for view in ("run1", "run2", "mean_of_runs"):
            cell = report["between_frameworks"][metric][view]
            parts.append(f"{view} {_fmt(cell['pearson'])}/{_fmt(cell['spearman'])}")
        print(f"  {metric:<18}" + "   ".join(parts))
    print("\nVersus the lab heuristic (mean of runs), pearson / spearman:")
    for framework in FRAMEWORKS:
        for metric in METRICS:
            cell = report["vs_lab_heuristic"][framework][metric]["mean_of_runs"]
            print(
                f"  {framework:<9}{metric:<18}"
                f"{_fmt(cell['pearson'])}/{_fmt(cell['spearman'])}  (n={cell['n']})"
            )
    failing = report["failing_cases"]
    print(f"\nLab heuristic failures: {failing['lab_heuristic_failures']['count']}")
    for run_name, entry in failing["by_run"].items():
        print(
            f"  {run_name}: ragas flags {len(entry['ragas'])}, "
            f"deepeval flags {len(entry['deepeval'])}, "
            f"both {len(entry['ragas_vs_deepeval']['both'])}"
        )


def _display(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--run1", type=Path, default=ARTIFACTS / "framework_comparison.json"
    )
    parser.add_argument(
        "--run2", type=Path, default=ARTIFACTS / "framework_comparison_run2.json"
    )
    parser.add_argument(
        "--benchmark", type=Path, default=ARTIFACTS / "benchmark_results.json"
    )
    parser.add_argument(
        "--answers", type=Path, default=ARTIFACTS / "actual_answers.json"
    )
    parser.add_argument(
        "--output", type=Path, default=ARTIFACTS / "framework_noise.json"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run1, run2 = _read_json(args.run1), _read_json(args.run2)
        benchmark, answers = _read_json(args.benchmark), _read_json(args.answers)
        if [r["id"] for r in run1["cases"]] != [r["id"] for r in run2["cases"]]:
            raise ValueError("the two runs cover different cases")
    except (ValueError, KeyError) as exc:
        print(f"ERROR: {exc}")
        return 2
    paths = {
        "run1": _display(args.run1),
        "run2": _display(args.run2),
        "benchmark_results": _display(args.benchmark),
        "actual_answers": _display(args.answers),
    }
    report = build_report(run1, run2, benchmark, answers, paths)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print_summary(report)
    print(f"\nSaved noise report: {_display(output)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
