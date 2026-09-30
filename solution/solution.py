"""
Day 14 — AI Evaluation & Benchmarking Pipeline
AICB-P1: AI Practical Competency Program, Phase 1

Key concepts from lecture:
    - Evaluation = Scientific Method for AI (Hypothesis → Experiment → Measure → Conclude → Iterate)
    - 4 nhóm metrics: Task Completion, Answer Quality, RAG-Specific, Business
    - RAG pipeline metrics: Context Recall → Context Precision → Faithfulness → Answer Relevancy
    - LLM-as-Judge: rubric scoring 1-5, detect bias (positional, verbosity, self-preference)
    - Golden dataset: stratified sampling (5 Easy + 7 Medium + 5 Hard + 3 Adversarial)
    - Failure taxonomy: hallucination, irrelevant, incomplete, off_topic, refusal
    - 5 Whys method for root cause analysis
    - CI/CD integration: eval as quality gate (score < threshold = block deploy)
    - Continuous Improvement Loop: Evaluate → Analyze → Improve → Augment → Repeat

Instructions:
    1. Fill in every required section marked with TODO.
    2. Do NOT change class/function signatures. The optional ``contexts``
       parameter in ``run_full_eval`` is part of the required interface.
    3. Copy this file to solution/solution.py when done.
    4. Run: pytest tests/ -v

The reranking helper is an optional bonus exercise and may remain unimplemented.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Shared thresholds
# ---------------------------------------------------------------------------
# Named once so the pass rule, the failure taxonomy, the regression gate and
# the judge-bias checks all read from the same place.

PASS_THRESHOLD: float = 0.5
"""A result passes only when all three answer metrics are >= this value."""

SEVERE_SCORE_THRESHOLD: float = 0.3
"""Below this, a single metric is bad enough to name the failure type."""

REGRESSION_DROP_THRESHOLD: float = 0.05
"""A metric regresses when its average falls MORE than this vs the baseline."""

POSITIONAL_BIAS_MARGIN: float = 0.1
"""The first judged item must out-score the rest by MORE than this to be
flagged as positional bias."""

LENIENCY_THRESHOLD: float = 0.8
"""A judge whose mean score is above this is flagged as lenient."""

SEVERITY_THRESHOLD: float = 0.3
"""A judge whose mean score is below this is flagged as severe."""

_FLOAT_TOLERANCE: float = 1e-9
"""Absorbs float noise so that, for example, 0.9 - 0.85 (which evaluates to
0.05000000000000004) does not count as "more than 0.05"."""

ANSWER_METRICS: tuple[str, ...] = ("faithfulness", "relevance", "completeness")

FAILURE_TAXONOMY: tuple[str, ...] = (
    "hallucination",
    "irrelevant",
    "incomplete",
    "off_topic",
)
"""Failure types in priority order. Also the tie-break order for suggestions."""


# ---------------------------------------------------------------------------
# Task 1 — Data Models (Golden Dataset + Evaluation Results)
# ---------------------------------------------------------------------------

@dataclass
class QAPair:
    """
    A question-answer pair for evaluation (part of the Golden Dataset).

    From lecture: Golden dataset cần có:
        - question: câu hỏi user
        - ground_truth (expected_answer): expert-written expected answer
        - context: source documents cần retrieve
        - metadata: difficulty (easy/medium/hard), category, source_docs

    Fields:
        question:        The question to answer.
        expected_answer: The reference/ground-truth answer (expert-written).
        context:            Source context (may be empty string if not applicable).
        metadata:           Optional metadata dict (difficulty, category, etc.).
        retrieved_contexts: List of retrieved chunks (ORDER = retriever rank).
                            Used by the retrieval-side metrics (Task 2b).
    """
    question: str
    expected_answer: str
    context: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    retrieved_contexts: list[str] = field(default_factory=list)


@dataclass
class EvalResult:
    """
    Evaluation result for a single Q&A pair.

    From lecture - RAG metrics pipeline:
        Question → Retriever → Context → Generator → Answer
        Each step has a metric: Context Recall, Context Precision, Faithfulness, Answer Relevancy

    From lecture - Score interpretation:
        0.8-1.0: Good (Monitor, maintain)
        0.6-0.8: Needs work (Analyze failures, iterate)
        < 0.6: Significant issues (Deep investigation required)

    Fields:
        qa_pair:        The original QAPair.
        actual_answer:  What the agent actually returned.
        faithfulness:   Float 0-1, how grounded the answer is in context.
        relevance:      Float 0-1, how relevant the answer is to the question.
        completeness:   Float 0-1, how complete the answer is vs expected.
        passed:         True if all three scores >= 0.5.
        failure_type:   None if passed, otherwise one of:
                        "hallucination", "irrelevant", "incomplete", "off_topic".
        context_precision: Float 0-1 or None — quality of retrieval ranking.
        context_recall:    Float 0-1 or None — coverage of expected by context.
                        (Both stay None unless retrieved chunks are supplied;
                         they are NOT part of overall_score().)

    Note: ``None`` means "retrieval metric not computed"; ``0.0`` means
    "computed, and the retriever scored zero". The two are never interchangeable.
    """
    qa_pair: QAPair
    actual_answer: str
    faithfulness: float
    relevance: float
    completeness: float
    passed: bool
    failure_type: str | None = None
    context_precision: float | None = None
    context_recall: float | None = None

    def overall_score(self) -> float:
        """Compute the average of faithfulness, relevance, and completeness.

        Returns:
            (faithfulness + relevance + completeness) / 3.0

        The retrieval metrics are deliberately excluded.
        """
        return (self.faithfulness + self.relevance + self.completeness) / 3.0


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------

def _clamp_unit(value: float) -> float:
    """Clamp ``value`` into the closed interval [0.0, 1.0]."""
    return max(0.0, min(1.0, value))


def _mean(values: list[float]) -> float:
    """Arithmetic mean, or 0.0 for an empty list."""
    return sum(values) / len(values) if values else 0.0


def _mean_or_none(values: list[float | None]) -> float | None:
    """Mean of the non-None values, or None when there are none."""
    present = [value for value in values if value is not None]
    return _mean(present) if present else None


# ---------------------------------------------------------------------------
# Task 2 — RAGAS Evaluator (Simplified word-overlap heuristic)
# ---------------------------------------------------------------------------
# In production, replace with actual RAGAS framework:
#   from ragas import evaluate
#   from ragas.metrics import Faithfulness, AnswerRelevancy, ContextRecall, ContextPrecision
#
# Or DeepEval:
#   from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
#   assert_test(test_case, [faithfulness, hallucination])
#
# Or TruLens:
#   from trulens.core import Feedback
#   f_groundedness = Feedback(provider.groundedness_measure_with_cot_reasons)
# ---------------------------------------------------------------------------

# Common English stopwords are ignored so overlap reflects *content* words,
# not filler (otherwise "is"/"a"/"the" inflate every score).
STOPWORDS: set[str] = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "in", "on", "at", "to", "for", "with", "as", "by", "and", "or",
    "it", "its", "this", "that", "these", "those", "from", "into", "than",
}


def _tokenize(text: str) -> set[str]:
    """Lowercase word tokenization, ignoring punctuation and stopwords."""
    if not text:
        return set()
    tokens = re.findall(r"\b\w+\b", text.lower())
    return {t for t in tokens if t not in STOPWORDS}


class RAGASEvaluator:
    """
    Evaluates RAG pipeline outputs using RAGAS-inspired heuristics.

    All metrics use word overlap rather than LLM calls for simplicity.
    Replace with actual LLM-based evaluation in production.
    """

    def evaluate_faithfulness(self, answer: str, context: str) -> float:
        """
        Measure how grounded the answer is in the context.

        Heuristic:
            answer_tokens = _tokenize(answer)
            context_tokens = _tokenize(context)
            faithfulness = |answer_tokens ∩ context_tokens| / |answer_tokens|
            Clamp to [0.0, 1.0]. Return 1.0 if answer is empty.

        A ``None`` context is treated as an empty string.

        Returns:
            float in [0.0, 1.0] — 1.0 = fully grounded in context.
        """
        answer_tokens = _tokenize(answer)
        if not answer_tokens:
            return 1.0
        context_tokens = _tokenize(context or "")
        return _clamp_unit(len(answer_tokens & context_tokens) / len(answer_tokens))

    def evaluate_relevance(self, answer: str, question: str) -> float:
        """
        Measure how relevant the answer is to the question.

        Heuristic:
            relevance = |answer_tokens ∩ question_tokens| / |question_tokens|
            Clamp to [0.0, 1.0]. Return 1.0 if question is empty.

        Returns:
            float in [0.0, 1.0]
        """
        question_tokens = _tokenize(question)
        if not question_tokens:
            return 1.0
        answer_tokens = _tokenize(answer)
        return _clamp_unit(len(answer_tokens & question_tokens) / len(question_tokens))

    def evaluate_completeness(self, answer: str, expected: str) -> float:
        """
        Measure how well the answer covers the expected answer.

        Heuristic:
            completeness = |answer_tokens ∩ expected_tokens| / |expected_tokens|
            Clamp to [0.0, 1.0]. Return 1.0 if expected is empty.

        Returns:
            float in [0.0, 1.0]
        """
        expected_tokens = _tokenize(expected)
        if not expected_tokens:
            return 1.0
        answer_tokens = _tokenize(answer)
        return _clamp_unit(len(answer_tokens & expected_tokens) / len(expected_tokens))

    # -----------------------------------------------------------------------
    # Task 2b — Retrieval-side metrics (evaluate the GET-CONTEXT step)
    # -----------------------------------------------------------------------
    # From lecture (RAG pipeline): Context Recall → Context Precision →
    #   Faithfulness → Answer Relevancy. The two below score the RETRIEVER,
    #   operating on a LIST of chunks (order = retriever rank).
    # -----------------------------------------------------------------------

    def evaluate_context_recall(self, contexts: list[str], expected: str) -> float:
        """Context Recall — how much of the expected answer is covered by the
        UNION of retrieved chunks.

        Heuristic:
            union_tokens = ⋃ _tokenize(chunk) for chunk in contexts
            recall = |expected_tokens ∩ union_tokens| / |expected_tokens|
            Clamp to [0.0, 1.0]. Return 1.0 if expected is empty.

        Low recall => retriever missed evidence the answer needs.
        The union makes this metric independent of chunk order; an empty chunk
        list scores 0.0 (nothing was retrieved, so nothing is covered).
        """
        expected_tokens = _tokenize(expected)
        if not expected_tokens:
            return 1.0
        union_tokens: set[str] = set()
        for chunk in contexts:
            union_tokens |= _tokenize(chunk)
        return _clamp_unit(len(expected_tokens & union_tokens) / len(expected_tokens))

    def evaluate_context_precision(
        self,
        contexts: list[str],
        expected: str,
        relevance_threshold: float = 0.1,
    ) -> float:
        """Context Precision — RANK-AWARE Average Precision (AP@K), like RAGAS.
        Rewards retrievers that place RELEVANT chunks BEFORE noise.

        Steps:
            1. A chunk is "relevant" if it covers >= relevance_threshold of the
               expected tokens:  |chunk ∩ expected| / |expected| >= threshold
            2. Precision@k = (#relevant in top-k) / k
            3. AP@K = (1 / #relevant) * Σ_k [ Precision@k · relevant_k ]

        Return 1.0 if expected empty; 0.0 if no chunks or none relevant.
        Reordering relevant chunks earlier (reranking) raises this score.
        """
        expected_tokens = _tokenize(expected)
        if not expected_tokens:
            return 1.0
        if not contexts:
            return 0.0

        relevant_so_far = 0
        precision_sum = 0.0
        for rank, chunk in enumerate(contexts, start=1):
            coverage = len(_tokenize(chunk) & expected_tokens) / len(expected_tokens)
            if coverage >= relevance_threshold:
                relevant_so_far += 1
                precision_sum += relevant_so_far / rank

        if relevant_so_far == 0:
            return 0.0
        return _clamp_unit(precision_sum / relevant_so_far)

    def run_full_eval(
        self,
        answer: str,
        question: str,
        context: str,
        expected: str,
        contexts: list[str] | None = None,
    ) -> EvalResult:
        """
        Run the three answer-side evaluations and, when ``contexts`` is
        supplied, both retrieval-side evaluations.

        passed = True if all three scores >= 0.5.

        failure_type determination (first match wins):
            faithfulness < 0.3  → "hallucination"
            relevance < 0.3     → "irrelevant"
            completeness < 0.3  → "incomplete"
            otherwise if failed → "off_topic"

        Retrieval wiring:
            contexts is None → context_recall and context_precision stay None
            contexts provided → evaluate and store both retrieval metrics

        The two retrieval metrics diagnose the retriever and do not change the
        three-metric ``passed`` rule or ``overall_score()``.

        Returns:
            EvalResult with all fields populated.

        Note: an empty list still counts as "provided", so both retrieval
        metrics are computed (and score 0.0) rather than left as None.
        """
        faithfulness = self.evaluate_faithfulness(answer, context)
        relevance = self.evaluate_relevance(answer, question)
        completeness = self.evaluate_completeness(answer, expected)

        passed = (
            faithfulness >= PASS_THRESHOLD
            and relevance >= PASS_THRESHOLD
            and completeness >= PASS_THRESHOLD
        )

        failure_type: str | None = None
        if not passed:
            if faithfulness < SEVERE_SCORE_THRESHOLD:
                failure_type = "hallucination"
            elif relevance < SEVERE_SCORE_THRESHOLD:
                failure_type = "irrelevant"
            elif completeness < SEVERE_SCORE_THRESHOLD:
                failure_type = "incomplete"
            else:
                failure_type = "off_topic"

        context_recall: float | None = None
        context_precision: float | None = None
        if contexts is not None:
            context_recall = self.evaluate_context_recall(contexts, expected)
            context_precision = self.evaluate_context_precision(contexts, expected)

        return EvalResult(
            qa_pair=QAPair(question, expected, context or ""),
            actual_answer=answer,
            faithfulness=faithfulness,
            relevance=relevance,
            completeness=completeness,
            passed=passed,
            failure_type=failure_type,
            context_precision=context_precision,
            context_recall=context_recall,
        )


# ---------------------------------------------------------------------------
# Reranking helper (used by Exercise 3.5 — boosting Context Precision)
# ---------------------------------------------------------------------------

def rerank_by_overlap(contexts: list[str], query: str) -> list[str]:
    """A minimal lexical reranker: sort chunks by word overlap with the query,
    most-overlapping first. Stand-in for a real cross-encoder reranker.

    Reordering relevant chunks toward the top increases the rank-aware
    Context Precision WITHOUT changing the retrieved set.

    Hint: sorted(contexts, key=lambda c: len(_tokenize(c) & _tokenize(query)),
                 reverse=True)

    The sort is stable (equal-overlap chunks keep their retriever order) and a
    NEW list is returned; the input list is never mutated, and no chunk is
    added or dropped.
    """
    query_tokens = _tokenize(query)

    def overlap_with_query(chunk: str) -> int:
        return len(_tokenize(chunk) & query_tokens)

    return sorted(contexts, key=overlap_with_query, reverse=True)


# ---------------------------------------------------------------------------
# Task 3 — LLM Judge
# ---------------------------------------------------------------------------
# From lecture:
#   - Judge LLM nhận: question + agent answer + reference answer + rubric
#   - Judge trả về: Score 1-5 + Rationale
#   - Best practices: multiple judges, randomize order, calibrate against human
#   - Biases: positional, verbosity, self-preference
#   - Rubric template:
#       5 = Correct, complete, well-cited
#       4 = Mostly correct, minor gaps
#       3 = Partially correct, some errors
#       2 = Significant errors or missing info
#       1 = Wrong or irrelevant
# ---------------------------------------------------------------------------

_DEFAULT_CRITERION_SCORE: float = 0.5
"""Neutral score used when the judge omits a criterion or returns garbage."""


def _parse_json_object(text: str) -> dict[str, Any] | None:
    """Best-effort extraction of one JSON object from an LLM reply.

    Tries, in order: the whole reply (with ```json fences removed), the span
    from the first ``{`` to the last ``}``, then the first complete object
    starting at the first ``{``. Returns None when nothing parses to a dict.
    """
    unfenced = re.sub(
        r"^\s*```(?:json)?\s*|\s*```\s*$", "", text.strip(), flags=re.IGNORECASE
    )
    candidates: list[str] = [unfenced]
    braced = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if braced:
        candidates.append(braced.group(0))

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(parsed, dict):
            return parsed

    first_brace = text.find("{")
    if first_brace != -1:
        try:
            parsed, _ = json.JSONDecoder().raw_decode(text[first_brace:])
        except (json.JSONDecodeError, ValueError):
            return None
        if isinstance(parsed, dict):
            return parsed
    return None


def _coerce_criterion_score(value: Any) -> float:
    """Turn a judge-supplied value into a float in [0, 1].

    Only real numbers count; booleans, strings, None and NaN fall back to the
    neutral default rather than being guessed at.
    """
    is_number = isinstance(value, (int, float)) and not isinstance(value, bool)
    if not is_number or value != value:  # value != value is the NaN test
        return _DEFAULT_CRITERION_SCORE
    return _clamp_unit(float(value))


def _item_scores(item: Any) -> list[float]:
    """All numeric criterion scores carried by one ``score_response`` result."""
    if not isinstance(item, dict):
        return []
    scores = item.get("scores")
    if not isinstance(scores, dict):
        return []
    return [
        float(value)
        for value in scores.values()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]


class LLMJudge:
    """
    Uses an LLM to score AI responses according to a rubric.
    """

    def __init__(self, judge_llm_fn: Callable[[str], str]) -> None:
        self.judge_llm_fn: Callable[[str], str] = judge_llm_fn

    def score_response(
        self,
        question: str,
        answer: str,
        rubric: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Score an AI response using the judge LLM.

        Args:
            question: The original question.
            answer:   The AI's answer to score.
            rubric:   Dict mapping criterion name → description.
                      Example: {"accuracy": "Is the answer factually correct?",
                                "clarity": "Is the answer clear and well-structured?"}

        Behavior:
            1. Build a judge prompt that includes the question, answer, and rubric.
            2. Call judge_llm_fn(prompt).
            3. Parse the response for scores.

        For simplicity, if the LLM response can't be parsed as JSON scores,
        return a default score of 0.5 for each criterion.

        Returns:
            {
                "scores":    dict[str, float],  # criterion → score 0-1
                "reasoning": str,               # raw LLM explanation
            }

        Scale note: in code every score is on a 0-1 scale (clamped), while the
        human rubric in the lecture/exercises uses 1-5. To compare the two,
        map a human score h to (h - 1) / 4.

        The judge may answer with a flat object ({"accuracy": 0.8, ...}) or one
        nested under "scores". A criterion that is missing or non-numeric
        scores 0.5. ``reasoning`` is the judge's "reasoning" field when present,
        otherwise the raw response text.
        """
        criteria_lines = "\n".join(
            f"- {criterion}: {description}" for criterion, description in rubric.items()
        )
        json_shape = ", ".join(
            [f"{json.dumps(criterion)}: <float 0-1>" for criterion in rubric]
            + ['"reasoning": "<short explanation>"']
        )
        prompt = (
            "You are an impartial judge evaluating an AI assistant's answer.\n\n"
            f"Question:\n{question}\n\n"
            f"Answer to evaluate:\n{answer}\n\n"
            "Score the answer on each criterion below. Use a 0-1 scale, where "
            "0 means the criterion is not met at all and 1 means it is fully met:\n"
            f"{criteria_lines}\n\n"
            "Return ONLY a JSON object, with no other text, in this form:\n"
            "{" + json_shape + "}"
        )

        raw_response = self.judge_llm_fn(prompt)
        parsed = _parse_json_object(raw_response)

        if parsed is None:
            return {
                "scores": {criterion: _DEFAULT_CRITERION_SCORE for criterion in rubric},
                "reasoning": raw_response,
            }

        nested = parsed.get("scores")
        score_source: dict[str, Any] = nested if isinstance(nested, dict) else parsed
        scores = {
            criterion: _coerce_criterion_score(score_source.get(criterion))
            for criterion in rubric
        }

        reasoning_field = parsed.get("reasoning")
        if reasoning_field is None:
            reasoning = raw_response
        else:
            reasoning = (
                reasoning_field if isinstance(reasoning_field, str) else str(reasoning_field)
            )
        return {"scores": scores, "reasoning": reasoning}

    def detect_bias(self, scores_batch: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Detect potential bias patterns in a batch of judge scores.

        Checks:
            positional_bias: Check if first response consistently scores higher
            leniency_bias:   Average score > 0.8 across all criteria
            severity_bias:   Average score < 0.3 across all criteria

        Args:
            scores_batch: List of score dicts from score_response().

        Returns:
            {
                "positional_bias": bool,
                "leniency_bias":   bool,
                "severity_bias":   bool,
            }

        Details:
            leniency / severity use the mean over every criterion score of every
            item. An empty batch (or one with no scores) flags nothing.

            positional_bias: when every item carries an int "position" key, the
            mean score of the items at the lowest position is compared with the
            mean score of all other items; otherwise item 0 is compared with the
            mean of the remaining items. It is flagged when the first group
            leads by more than POSITIONAL_BIAS_MARGIN (needs at least 2 items).
        """
        no_bias: dict[str, Any] = {
            "positional_bias": False,
            "leniency_bias": False,
            "severity_bias": False,
        }
        if not scores_batch:
            return no_bias

        per_item_scores = [_item_scores(item) for item in scores_batch]
        every_score = [score for scores in per_item_scores for score in scores]
        if not every_score:
            return no_bias

        overall_mean = _mean(every_score)
        leniency_bias = overall_mean > LENIENCY_THRESHOLD + _FLOAT_TOLERANCE
        severity_bias = overall_mean < SEVERITY_THRESHOLD - _FLOAT_TOLERANCE

        # One mean per item; items that carry no numeric score are left out.
        item_means = [_mean(scores) if scores else None for scores in per_item_scores]
        positions = [
            item.get("position") if isinstance(item, dict) else None
            for item in scores_batch
        ]
        has_positions = len(scores_batch) >= 2 and all(
            isinstance(position, int) and not isinstance(position, bool)
            for position in positions
        )

        first_group: list[float] = []
        other_group: list[float] = []
        if has_positions:
            first_position = min(position for position in positions if position is not None)
            for position, item_mean in zip(positions, item_means):
                if item_mean is None:
                    continue
                (first_group if position == first_position else other_group).append(item_mean)
        else:
            for index, item_mean in enumerate(item_means):
                if item_mean is None:
                    continue
                (first_group if index == 0 else other_group).append(item_mean)

        positional_bias = (
            bool(first_group)
            and bool(other_group)
            and _mean(first_group) - _mean(other_group)
            > POSITIONAL_BIAS_MARGIN + _FLOAT_TOLERANCE
        )

        return {
            "positional_bias": positional_bias,
            "leniency_bias": leniency_bias,
            "severity_bias": severity_bias,
        }


# ---------------------------------------------------------------------------
# Task 4 — Benchmark Runner
# ---------------------------------------------------------------------------
# From lecture:
#   - CI/CD integration: Framework + CI/CD = quality gate tự động
#   - Agent với faithfulness < 0.7 → không được deploy
#   - Regression = metric drop > 0.05 vs baseline
#   - Triggers: mỗi code release, mỗi prompt change, trước demo/launch
# ---------------------------------------------------------------------------

class BenchmarkRunner:
    """
    Runs a full evaluation benchmark.
    """

    def run(
        self,
        qa_pairs: list[QAPair],
        agent_fn: Callable[[str], str],
        evaluator: RAGASEvaluator,
    ) -> list[EvalResult]:
        """
        Run all QA pairs through the agent and evaluate each result.

        Args:
            qa_pairs:   List of QAPair objects.
            agent_fn:   Function str → str (the agent's answer function).
            evaluator:  RAGASEvaluator instance.

        Returns:
            List of EvalResult, one per qa_pair.

        Each result's ``qa_pair`` is the ORIGINAL pair (so its metadata, such as
        the golden-dataset id, survives), and ``pair.retrieved_contexts`` is
        forwarded as the ``contexts`` argument.
        """
        results: list[EvalResult] = []
        for pair in qa_pairs:
            answer = agent_fn(pair.question)
            result = evaluator.run_full_eval(
                answer,
                pair.question,
                pair.context or "",
                pair.expected_answer,
                contexts=pair.retrieved_contexts,
            )
            result.qa_pair = pair
            results.append(result)
        return results

    def generate_report(self, results: list[EvalResult]) -> dict[str, Any]:
        """
        Generate an aggregate report from evaluation results.

        Returns:
            {
                "total":            int,
                "passed":           int,
                "pass_rate":        float,  # passed / total
                "avg_faithfulness": float,
                "avg_relevance":    float,
                "avg_completeness": float,
                "avg_context_recall": float | None,
                "avg_context_precision": float | None,
                "failure_types":    dict[str, int],  # type → count
            }

        Average only non-None retrieval scores. Return None for a retrieval
        average when no result contains that metric.

        An empty ``results`` list gives total 0, pass_rate 0.0, answer averages
        0.0, retrieval averages None and no failure types (never an exception).
        """
        total = len(results)
        passed = sum(1 for result in results if result.passed)
        failure_types = Counter(
            result.failure_type for result in results if result.failure_type is not None
        )
        return {
            "total": total,
            "passed": passed,
            "pass_rate": passed / total if total else 0.0,
            "avg_faithfulness": _mean([result.faithfulness for result in results]),
            "avg_relevance": _mean([result.relevance for result in results]),
            "avg_completeness": _mean([result.completeness for result in results]),
            "avg_context_recall": _mean_or_none(
                [result.context_recall for result in results]
            ),
            "avg_context_precision": _mean_or_none(
                [result.context_precision for result in results]
            ),
            "failure_types": dict(failure_types),
        }

    def run_regression(self, new_results: list, baseline_results: list) -> dict:
        """Compare new evaluation results against a baseline.

        A regression is when a metric's average drops by more than 0.05 vs baseline.

        Args:
            new_results: List of EvalResult instances (current run)
            baseline_results: List of EvalResult instances (reference/baseline)

        Returns:
            dict with keys:
              - 'new_avg_faithfulness': float
              - 'new_avg_relevance': float
              - 'new_avg_completeness': float
              - 'baseline_avg_faithfulness': float
              - 'baseline_avg_relevance': float
              - 'baseline_avg_completeness': float
              - 'regressions': list[str] — names of metrics that regressed
              - 'passed': bool — True if no regressions

        An empty result list averages to 0.0. A drop of exactly 0.05 is NOT a
        regression; REGRESSION_DROP_THRESHOLD plus a 1e-9 tolerance keeps float
        noise (0.9 - 0.85 == 0.05000000000000004) from tipping it over.
        """
        new_averages = {
            metric: _mean([getattr(result, metric) for result in new_results])
            for metric in ANSWER_METRICS
        }
        baseline_averages = {
            metric: _mean([getattr(result, metric) for result in baseline_results])
            for metric in ANSWER_METRICS
        }
        regressions = [
            metric
            for metric in ANSWER_METRICS
            if baseline_averages[metric] - new_averages[metric]
            > REGRESSION_DROP_THRESHOLD + _FLOAT_TOLERANCE
        ]

        comparison: dict[str, Any] = {}
        for metric in ANSWER_METRICS:
            comparison[f"new_avg_{metric}"] = new_averages[metric]
        for metric in ANSWER_METRICS:
            comparison[f"baseline_avg_{metric}"] = baseline_averages[metric]
        comparison["regressions"] = regressions
        comparison["passed"] = not regressions
        return comparison

    def identify_failures(
        self,
        results: list[EvalResult],
        threshold: float = 0.5,
    ) -> list[EvalResult]:
        """
        Return EvalResults where any score is below threshold.

        Args:
            results:   Full list of EvalResults.
            threshold: Minimum acceptable score for any metric.

        Returns:
            List of failing EvalResults.

        "Any score" means any of the three answer metrics; the retrieval
        metrics are diagnostics and are not checked here.
        """
        return [
            result
            for result in results
            if any(getattr(result, metric) < threshold for metric in ANSWER_METRICS)
        ]


# ---------------------------------------------------------------------------
# Task 5 — Failure Analyzer
# ---------------------------------------------------------------------------
# From lecture:
#   Failure Taxonomy:
#     - hallucination: bịa thông tin → faithfulness guardrail yếu
#     - irrelevant: không giải quyết câu hỏi → prompt ambiguous
#     - incomplete: bỏ sót thông tin → context window nhỏ, retrieval thiếu
#     - off_topic: trả lời chủ đề khác → intent detection sai
#     - refusal: từ chối khi nên trả lời → guardrails quá chặt
#
#   5 Whys Method: hỏi "Tại sao?" liên tục cho đến root cause
#   Failure Clustering: fix 1 root cause giải quyết nhiều failures cùng lúc
#   Continuous Improvement: Evaluate → Analyze → Improve → Augment → Repeat
# ---------------------------------------------------------------------------

ROOT_CAUSE_RETRIEVAL = "Context is missing or irrelevant — improve retrieval"
ROOT_CAUSE_PROMPT = "Answer does not address the question — improve prompt clarity"
ROOT_CAUSE_GENERATION = (
    "Answer is missing key information — increase context window or improve generation"
)
ROOT_CAUSE_MULTIPLE = "Multiple issues detected — review full pipeline"

_ROOT_CAUSE_BY_METRIC: dict[str, str] = {
    "faithfulness": ROOT_CAUSE_RETRIEVAL,
    "relevance": ROOT_CAUSE_PROMPT,
    "completeness": ROOT_CAUSE_GENERATION,
}

_FAILURE_TYPE_BY_METRIC: dict[str, str] = {
    "faithfulness": "hallucination",
    "relevance": "irrelevant",
    "completeness": "incomplete",
}

# Concrete actions for THIS system: a BM25 retriever (top_k=5) over a
# 10-document OrbitTech customer-support policy corpus feeding an LLM generator.
_SUGGESTION_BY_FAILURE_TYPE: dict[str, str] = {
    "hallucination": (
        "Add a grounding check that flags answer sentences whose content words "
        "are absent from the retrieved chunks (target: faithfulness)"
    ),
    "irrelevant": (
        "Add few-shot examples that restate the question's key entities before "
        "answering (target: relevance)"
    ),
    "incomplete": (
        "Raise retriever top_k from 5 to 8 and expand multi-part policy questions "
        "into sub-queries so conditions and exceptions are retrieved "
        "(target: completeness, context recall)"
    ),
    "off_topic": (
        "Add an intent/scope classifier before retrieval so out-of-scope or "
        "off-topic questions get the scope-policy response (target: off_topic)"
    ),
}

# Used to pad the list up to three when fewer failure types were observed.
_GENERAL_PIPELINE_SUGGESTIONS: tuple[str, ...] = (
    "Add every failing case from this run to the golden regression set so the "
    "next benchmark run guards against it",
    "Calibrate the word-overlap metrics against human 1-5 labels on a sample of "
    "answers, and switch to the LLM judge where they disagree",
    "Gate each prompt or retriever change on run_regression: block the change "
    "when any answer metric drops more than 0.05 against the stored baseline",
)

_MINIMUM_SUGGESTIONS = 3


def _weakest_answer_metric(failure: EvalResult) -> str:
    """Name of the lowest answer metric; ties go faithfulness, relevance, completeness."""
    weakest_metric = ANSWER_METRICS[0]
    weakest_score = getattr(failure, weakest_metric)
    for metric in ANSWER_METRICS[1:]:
        score = getattr(failure, metric)
        if score < weakest_score:
            weakest_metric, weakest_score = metric, score
    return weakest_metric


def _escape_table_cell(value: Any) -> str:
    """Make ``value`` safe inside one Markdown table cell (no pipes, no newlines)."""
    single_line = " ".join(str(value).split())
    return single_line.replace("|", "\\|")


class FailureAnalyzer:
    """
    Analyzes failed evaluation results to identify patterns and suggest fixes.
    """

    def categorize_failures(
        self, failures: list[EvalResult]
    ) -> dict[str, int]:
        """
        Count failures by failure_type.

        Returns:
            dict mapping failure_type → count.
            Example: {"hallucination": 3, "irrelevant": 2, "incomplete": 5}

        A result whose failure_type is None is counted under "unknown".
        An empty list returns {}.
        """
        counts = Counter(failure.failure_type or "unknown" for failure in failures)
        return dict(counts)

    def find_root_cause(self, failure: EvalResult) -> str:
        """
        Suggest a root cause for a single failure based on its scores.

        Returns one of these strings based on which score is lowest:
            "Context is missing or irrelevant — improve retrieval"
            "Answer does not address the question — improve prompt clarity"
            "Answer is missing key information — increase context window or improve generation"
            "Multiple issues detected — review full pipeline"

        Rule: if two or more of the three answer scores are below 0.3 the
        problem is not one stage, so "Multiple issues" is returned. Otherwise
        the lowest score decides (ties: faithfulness, then relevance, then
        completeness): faithfulness -> retrieval, relevance -> prompt,
        completeness -> generation.
        """
        severe_count = sum(
            1 for metric in ANSWER_METRICS
            if getattr(failure, metric) < SEVERE_SCORE_THRESHOLD
        )
        if severe_count >= 2:
            return ROOT_CAUSE_MULTIPLE
        return _ROOT_CAUSE_BY_METRIC[_weakest_answer_metric(failure)]

    def generate_improvement_log(self, failures: list, suggestions: list[str]) -> str:
        """Generate a Markdown table logging failures and improvement actions.

        Format:
        | Failure ID | Type | Root Cause | Suggested Fix | Status |
        |------------|------|------------|---------------|--------|
        | F001       | ...  | ...        | ...           | Open   |

        Args:
            failures: List of EvalResult instances where passed=False
            suggestions: List of suggestion strings (one per failure, can be shorter list)

        Returns:
            Markdown table string with a row per failure. Status is always "Open".

        Row details: the ID is F001, F002, ... and, when the QAPair metadata has
        an "id", it is appended as "F001 (H02)". Type is the failure_type as
        given ("unknown" when None). Suggested Fix is the suggestion written for
        the row's failure type when the list contains it, else suggestions[i];
        once the list runs out the last suggestion is reused, and with no suggestions at
        all the cell says to investigate the trace manually. Pipes inside a cell
        are escaped. No failures gives just the header and separator.
        """
        lines = [
            "| Failure ID | Type | Root Cause | Suggested Fix | Status |",
            "|------------|------|------------|---------------|--------|",
        ]
        for index, failure in enumerate(failures):
            failure_id = f"F{index + 1:03d}"
            metadata = failure.qa_pair.metadata or {}
            source_id = metadata.get("id")
            if source_id:
                failure_id = f"{failure_id} ({source_id})"

            # The suggestions list is ordered by failure-type frequency, not by
            # failure, so prefer the action written for this row's own type.
            matching_fix = _SUGGESTION_BY_FAILURE_TYPE.get(
                (failure.failure_type or "").strip().lower()
            )
            if matching_fix in suggestions:
                fix = matching_fix
            elif index < len(suggestions):
                fix = suggestions[index]
            elif suggestions:
                fix = suggestions[-1]
            else:
                fix = "Investigate the trace manually"

            cells = [
                failure_id,
                failure.failure_type or "unknown",
                self.find_root_cause(failure),
                fix,
                "Open",
            ]
            lines.append("| " + " | ".join(_escape_table_cell(cell) for cell in cells) + " |")
        return "\n".join(lines)

    def generate_improvement_suggestions(
        self, failures: list[EvalResult]
    ) -> list[str]:
        """
        Generate a prioritized list of improvement suggestions based on failure patterns.

        Each suggestion should be a concrete, actionable string.

        Examples:
            "Increase chunk size in RAG pipeline to reduce context fragmentation"
            "Add few-shot examples showing complete answers to improve completeness"
            "Implement hallucination checker to filter unsupported claims"

        Returns:
            List of at least 3 suggestion strings (or fewer if failures is empty).

        Failure types are ordered by how often they occur (ties follow the
        taxonomy order hallucination, irrelevant, incomplete, off_topic), and
        each maps to one action written for this system (BM25 retriever,
        top_k=5, 10-document OrbitTech policy corpus, LLM generator). A failure
        whose type is None or not in the taxonomy (for example "Low_completeness")
        is classified from its lowest answer score instead. If that yields fewer
        than three actions, general pipeline actions are appended.
        """
        if not failures:
            return []

        type_counts: Counter[str] = Counter()
        for failure in failures:
            failure_type = (failure.failure_type or "").strip().lower()
            if failure_type not in FAILURE_TAXONOMY:
                failure_type = _FAILURE_TYPE_BY_METRIC[_weakest_answer_metric(failure)]
            type_counts[failure_type] += 1

        def priority(failure_type: str) -> tuple[int, int]:
            return (-type_counts[failure_type], FAILURE_TAXONOMY.index(failure_type))

        ordered_types = sorted(type_counts, key=priority)

        suggestions: list[str] = []
        for failure_type in ordered_types:
            action = _SUGGESTION_BY_FAILURE_TYPE[failure_type]
            if action not in suggestions:
                suggestions.append(action)

        for action in _GENERAL_PIPELINE_SUGGESTIONS:
            if len(suggestions) >= _MINIMUM_SUGGESTIONS:
                break
            if action not in suggestions:
                suggestions.append(action)
        return suggestions


# ---------------------------------------------------------------------------
# Entry point for manual testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Sample golden dataset (mini version — use 20 pairs in actual lab)
    # From lecture: stratified sampling = 5 Easy + 7 Medium + 5 Hard + 3 Adversarial
    qa_pairs = [
        # Easy — factual lookup
        QAPair(
            question="What is RAG?",
            expected_answer="RAG stands for Retrieval-Augmented Generation, which combines retrieval with text generation.",
            context="RAG is a technique that retrieves relevant documents and uses them to ground LLM generation.",
            metadata={"difficulty": "easy", "category": "definition"},
        ),
        QAPair(
            question="What is the capital of France?",
            expected_answer="Paris is the capital of France.",
            context="France is a country in Western Europe. Its capital city is Paris.",
            metadata={"difficulty": "easy", "category": "factual"},
        ),
        # Medium — multi-step reasoning
        QAPair(
            question="Explain backpropagation and why it matters for training",
            expected_answer="Backpropagation is an algorithm for training neural networks by computing gradients efficiently, enabling deep learning models to learn from errors.",
            context="Neural networks learn through gradient descent. Backpropagation efficiently computes these gradients layer by layer.",
            metadata={"difficulty": "medium", "category": "explanation"},
        ),
        # Hard — ambiguous
        QAPair(
            question="Should I use RAG or fine-tuning for my chatbot?",
            expected_answer="It depends on the use case: RAG is better for frequently updated knowledge, fine-tuning for consistent style/behavior. Consider cost, latency, and data freshness.",
            context="RAG retrieves external documents at inference time. Fine-tuning modifies model weights during training.",
            metadata={"difficulty": "hard", "category": "comparison"},
        ),
        # Adversarial — out-of-scope
        QAPair(
            question="What is the meaning of life?",
            expected_answer="This question is outside the scope of this system. I can help with AI and technology questions.",
            context="This is an AI assistant specialized in technology topics.",
            metadata={"difficulty": "adversarial", "category": "out_of_scope"},
        ),
    ]

    evaluator = RAGASEvaluator()
    runner = BenchmarkRunner()

    def mock_agent(question: str) -> str:
        """Simple mock agent for testing. Replace with your actual agent."""
        return f"Based on my knowledge: {question[:30]}... The answer involves key concepts."

    # Run benchmark
    results = runner.run(qa_pairs, mock_agent, evaluator)
    report = runner.generate_report(results)
    print("=== Benchmark Report ===")
    for k, v in report.items():
        print(f"  {k}: {v}")

    # Identify and analyze failures
    failures = runner.identify_failures(results, threshold=0.5)
    print(f"\n=== Failures ({len(failures)}) ===")
    analyzer = FailureAnalyzer()

    # Categorize (from lecture: cluster before fix)
    categories = analyzer.categorize_failures(failures)
    print("Failure Categories:", categories)

    # Root cause for each failure (from lecture: 5 Whys)
    for f in failures:
        cause = analyzer.find_root_cause(f)
        print(f"  Root cause: {cause}")

    # Improvement suggestions (from lecture: continuous improvement loop)
    suggestions = analyzer.generate_improvement_suggestions(failures)
    print("\nImprovement Suggestions:")
    for s in suggestions:
        print(f"  - {s}")

    # Generate improvement log (Markdown table)
    log = analyzer.generate_improvement_log(failures, suggestions)
    print("\n=== Improvement Log ===")
    print(log)
