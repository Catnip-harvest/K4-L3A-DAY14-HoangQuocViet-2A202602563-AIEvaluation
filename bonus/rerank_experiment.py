"""Exercise 3.5 (bonus): does reranking the SAME retrieved chunks raise Context Precision?

For every case in ``artifacts/actual_answers.json`` this script

1. takes the retrieved chunks in the retriever's rank order,
2. scores Context Recall and Context Precision against the golden expected answer,
3. reranks the same chunks with ``template.rerank_by_overlap(chunks, question)``,
4. asserts the reranked list is a permutation of the original (nothing added or dropped),
5. scores both metrics again.

The reranker only sees the QUESTION. At inference time the expected answer does not
exist, so reranking by it would leak the reference and make the experiment meaningless.
The metrics, on the other hand, are allowed to use the expected answer: they are the
measuring stick, not part of the system under test.

No LLM is called and no saved artifact is modified. Usage, from the repo root:

    python bonus/rerank_experiment.py
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from template import RAGASEvaluator, rerank_by_overlap  # noqa: E402

GOLDEN_PATH = REPO_ROOT / "golden_dataset.json"
ANSWERS_PATH = REPO_ROOT / "artifacts" / "actual_answers.json"
OUTPUT_PATH = REPO_ROOT / "artifacts" / "rerank_results.json"
FLOAT_TOLERANCE = 1e-9


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _squash_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _gold_chunk_ranks(chunks: list[str], gold_texts: list[str]) -> list[int | None]:
    """1-based rank of the chunk that contains each gold excerpt (None if absent)."""
    squashed_chunks = [_squash_whitespace(chunk) for chunk in chunks]
    ranks: list[int | None] = []
    for gold_text in gold_texts:
        needle = _squash_whitespace(gold_text)
        ranks.append(
            next(
                (rank for rank, chunk in enumerate(squashed_chunks, start=1) if needle in chunk),
                None,
            )
        )
    return ranks


def _label_chunks(chunks: list[str], text_to_id: dict[str, str]) -> list[str]:
    return [text_to_id[chunk] for chunk in chunks]


def run_case(
    golden_record: dict[str, Any],
    answer_record: dict[str, Any],
    evaluator: RAGASEvaluator,
) -> dict[str, Any]:
    question = golden_record["question"]
    expected = golden_record["expected_answer"]
    retrieved = answer_record["retrieved_contexts"]
    chunks = [record["text"].strip() for record in retrieved]
    text_to_id = {record["text"].strip(): record["chunk_id"] for record in retrieved}
    if len(text_to_id) != len(chunks):
        raise ValueError(f"{golden_record['id']}: duplicate chunk text in retrieved list")

    recall_before = evaluator.evaluate_context_recall(chunks, expected)
    precision_before = evaluator.evaluate_context_precision(chunks, expected)

    reranked = rerank_by_overlap(chunks, question)

    if Counter(reranked) != Counter(chunks) or len(reranked) != len(chunks):
        raise AssertionError(f"{golden_record['id']}: reranked list is not a permutation")
    recall_after = evaluator.evaluate_context_recall(reranked, expected)
    precision_after = evaluator.evaluate_context_precision(reranked, expected)
    if abs(recall_after - recall_before) > FLOAT_TOLERANCE:
        raise AssertionError(f"{golden_record['id']}: recall changed after reranking")

    gold_texts = [context["text"] for context in golden_record["contexts"]]
    return {
        "id": golden_record["id"],
        "difficulty": golden_record["difficulty"],
        "question": question,
        "recall_before": recall_before,
        "recall_after": recall_after,
        "precision_before": precision_before,
        "precision_after": precision_after,
        "delta_precision": precision_after - precision_before,
        "order_changed": reranked != chunks,
        "chunk_order_before": _label_chunks(chunks, text_to_id),
        "chunk_order_after": _label_chunks(reranked, text_to_id),
        "gold_excerpt_ranks_before": _gold_chunk_ranks(chunks, gold_texts),
        "gold_excerpt_ranks_after": _gold_chunk_ranks(reranked, gold_texts),
    }


def _average(rows: list[dict[str, Any]], key: str) -> float:
    return sum(row[key] for row in rows) / len(rows) if rows else 0.0


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "cases": len(rows),
        "avg_recall_before": _average(rows, "recall_before"),
        "avg_recall_after": _average(rows, "recall_after"),
        "avg_precision_before": _average(rows, "precision_before"),
        "avg_precision_after": _average(rows, "precision_after"),
        "avg_delta_precision": _average(rows, "delta_precision"),
        "order_changed": sum(1 for row in rows if row["order_changed"]),
        "precision_up": sum(1 for row in rows if row["delta_precision"] > FLOAT_TOLERANCE),
        "precision_down": sum(1 for row in rows if row["delta_precision"] < -FLOAT_TOLERANCE),
        "precision_same": sum(1 for row in rows if abs(row["delta_precision"]) <= FLOAT_TOLERANCE),
    }


def print_table(rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    print("| ID | Recall before | Recall after | Precision before | Precision after | Delta | Order changed |")
    print("|---|---:|---:|---:|---:|---:|---|")
    for row in rows:
        print(
            f"| {row['id']} | {row['recall_before']:.3f} | {row['recall_after']:.3f} | "
            f"{row['precision_before']:.3f} | {row['precision_after']:.3f} | "
            f"{row['delta_precision']:+.3f} | {'yes' if row['order_changed'] else 'no'} |"
        )
    print(
        f"| **Avg ({summary['cases']} cases)** | {summary['avg_recall_before']:.3f} | "
        f"{summary['avg_recall_after']:.3f} | {summary['avg_precision_before']:.3f} | "
        f"{summary['avg_precision_after']:.3f} | {summary['avg_delta_precision']:+.3f} | "
        f"{summary['order_changed']} changed |"
    )
    print(
        f"\nPrecision up: {summary['precision_up']}, down: {summary['precision_down']}, "
        f"same: {summary['precision_same']}"
    )


def main() -> int:
    golden = _read_json(GOLDEN_PATH)
    answers = _read_json(ANSWERS_PATH)
    if golden.get("corpus_id") != answers.get("corpus_id"):
        raise ValueError("golden dataset and actual answers use different corpus_id")
    answers_by_id = {record["id"]: record for record in answers["answers"]}

    evaluator = RAGASEvaluator()
    rows = [
        run_case(record, answers_by_id[record["id"]], evaluator)
        for record in golden["qa_pairs"]
    ]
    summary = summarize(rows)

    below_one = [row for row in rows if row["precision_before"] < 1.0 - FLOAT_TOLERANCE]
    artifact = {
        "description": (
            "Context Recall/Precision before and after rerank_by_overlap(chunks, question) "
            "on the same retrieved chunks. The reranker sees the question only; metrics use "
            "the golden expected answer."
        ),
        "source_generated_at": answers.get("generated_at"),
        "agent": answers.get("agent"),
        "reranker": "template.rerank_by_overlap",
        "reranker_query": "question",
        "average_all_cases": summary,
        "average_cases_with_precision_below_1": {
            "ids": [row["id"] for row in below_one],
            **summarize(below_one),
        },
        "cases": rows,
    }
    OUTPUT_PATH.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print_table(rows, summary)
    print(
        "\nCases with precision before < 1.0: "
        + ", ".join(row["id"] for row in below_one)
        + f"\nAverage over those {len(below_one)} cases: precision "
        f"{_average(below_one, 'precision_before'):.3f} -> {_average(below_one, 'precision_after'):.3f}"
    )
    print(f"\nSaved: {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
