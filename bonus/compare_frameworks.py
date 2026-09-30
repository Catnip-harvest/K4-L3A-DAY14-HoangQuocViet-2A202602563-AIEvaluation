"""Exercise 3.4 bonus: compare RAGAS and DeepEval on the lab's golden dataset.

Both frameworks receive exactly the same input for every case (question,
actual answer, retrieved contexts in rank order, expected answer as the
reference) and are judged by exactly the same LLM, reached through the same
``OpenAIJudge`` class. The four metric pairs compared are:

    faithfulness       RAGAS Faithfulness
                       DeepEval FaithfulnessMetric
    answer_relevancy   RAGAS ResponseRelevancy
                       DeepEval AnswerRelevancyMetric
    context_recall     RAGAS LLMContextRecall
                       DeepEval ContextualRecallMetric
    context_precision  RAGAS LLMContextPrecisionWithReference
                       DeepEval ContextualPrecisionMetric

Why custom model wrappers: ``gpt-6-luna`` rejects the low ``temperature`` both
frameworks send by default, so each framework is handed a small adapter that
calls the OpenAI Responses API with a reasoning effort instead.

Usage (from the repository root, with the bonus interpreter):

    python bonus/compare_frameworks.py
    python bonus/compare_frameworks.py --ids E01 H03 --output artifacts/x.json

The OpenAI key is read from the repository ``.env`` and is never printed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import re
import statistics
import sys
import tempfile
import threading
import time
import warnings
from collections.abc import Awaitable, Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent


def _configure_environment() -> None:
    """Switch off telemetry and keep framework caches out of the repository.

    This must run before ``ragas`` or ``deepeval`` is imported, because both
    read these variables at import time.
    """

    os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "YES"
    os.environ["RAGAS_DO_NOT_TRACK"] = "true"
    # DeepEval would otherwise load ./.env on its own; this script loads it once.
    os.environ["DEEPEVAL_DISABLE_DOTENV"] = "1"
    # DeepEval creates ``.deepeval/`` in the working directory by default.
    os.environ.setdefault(
        "DEEPEVAL_CACHE_FOLDER",
        str(Path(tempfile.gettempdir()) / "deepeval-framework-comparison"),
    )


_configure_environment()

from deepeval.metrics import (  # noqa: E402
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
)
from deepeval.models import DeepEvalBaseLLM  # noqa: E402
from deepeval.test_case import LLMTestCase  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from langchain_core.outputs import Generation, LLMResult  # noqa: E402
from openai import AsyncOpenAI, OpenAI  # noqa: E402
from pydantic import BaseModel, ValidationError  # noqa: E402
from ragas.dataset_schema import SingleTurnSample  # noqa: E402
from ragas.embeddings import BaseRagasEmbeddings  # noqa: E402
from ragas.llms import BaseRagasLLM  # noqa: E402
from ragas.run_config import RunConfig  # noqa: E402

# RAGAS 0.4 keeps these classic metric classes importable from ``ragas.metrics``
# but warns that ``ragas.metrics.collections`` is preferred. The collections
# versions only take an instructor client and call Chat Completions with their
# own sampling settings, so DeepEval and RAGAS would not be judged under the
# same conditions. The classic classes accept a plain ``BaseRagasLLM``, which
# lets both frameworks share ``OpenAIJudge``.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from ragas.metrics import (  # noqa: E402
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        ResponseRelevancy,
    )

METRIC_NAMES = (
    "faithfulness",
    "answer_relevancy",
    "context_recall",
    "context_precision",
)
FRAMEWORK_NAMES = ("ragas", "deepeval")
FAILURE_THRESHOLD = 0.5
MIN_PAIRS_FOR_CORRELATION = 3
SCORE_DIGITS = 4
REQUEST_TIMEOUT_SECONDS = 120
MAX_ERROR_LENGTH = 400
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
# A cut-off judge answer is retried once with double the output-token budget.
TOKEN_BUDGET_MULTIPLIERS = (1, 2)
STRUCTURED_OUTPUT_ATTEMPTS = 2


# --------------------------------------------------------------------------
# Input: golden dataset joined with the saved actual answers
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalCase:
    """One question with everything both frameworks need to score it."""

    case_id: str
    difficulty: str | None
    question: str
    actual_answer: str
    retrieved_contexts: tuple[str, ...]
    reference: str


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"{label} not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {path} ({exc.msg})") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} root must be a JSON object")
    return value


def _require_text(record: dict[str, Any], key: str, where: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where}: {key} must be a non-empty string")
    return value.strip()


def _answers_by_id(answers_document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    records = answers_document.get("answers")
    if not isinstance(records, list) or not records:
        raise ValueError("Actual answers must contain a non-empty answers list")
    by_id: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"answers[{index}] must be an object")
        record_id = _require_text(record, "id", f"answers[{index}]")
        if record_id in by_id:
            raise ValueError(f"Duplicate actual-answer ID: {record_id}")
        by_id[record_id] = record
    return by_id


def _retrieved_texts(actual_record: dict[str, Any], case_id: str) -> tuple[str, ...]:
    retrieved = actual_record.get("retrieved_contexts")
    if not isinstance(retrieved, list):
        raise ValueError(f"{case_id}: retrieved_contexts must be a list")
    texts: list[str] = []
    for index, chunk in enumerate(retrieved):
        where = f"{case_id}: retrieved_contexts[{index}]"
        if not isinstance(chunk, dict):
            raise ValueError(f"{where} must be an object")
        texts.append(_require_text(chunk, "text", where))
    return tuple(texts)


def load_cases(
    golden_path: Path,
    answers_path: Path,
    selected_ids: Sequence[str] | None = None,
) -> tuple[str, list[EvalCase]]:
    """Join golden records and saved answers by id, in golden-file order."""

    golden = _read_json_object(golden_path, "Golden dataset")
    answers = _read_json_object(answers_path, "Actual answers")
    corpus_id = golden.get("corpus_id")
    if corpus_id != answers.get("corpus_id"):
        raise ValueError("Golden dataset and actual answers use different corpus_id")
    golden_records = golden.get("qa_pairs")
    if not isinstance(golden_records, list) or not golden_records:
        raise ValueError("Golden dataset qa_pairs must be a non-empty list")

    answers_lookup = _answers_by_id(answers)
    wanted = set(selected_ids) if selected_ids else None
    cases: list[EvalCase] = []
    seen_ids: set[str] = set()
    for index, record in enumerate(golden_records):
        if not isinstance(record, dict):
            raise ValueError(f"qa_pairs[{index}] must be an object")
        case_id = _require_text(record, "id", f"qa_pairs[{index}]")
        seen_ids.add(case_id)
        if wanted is not None and case_id not in wanted:
            continue
        actual_record = answers_lookup.get(case_id)
        if actual_record is None:
            raise ValueError(f"Missing actual answer for {case_id}")
        if actual_record.get("error") is not None:
            raise ValueError(f"{case_id}: inference contains an error")
        question = _require_text(record, "question", case_id)
        if actual_record.get("question") != record.get("question"):
            raise ValueError(f"{case_id}: question differs between the two files")
        cases.append(
            EvalCase(
                case_id=case_id,
                difficulty=record.get("difficulty"),
                question=question,
                actual_answer=_require_text(actual_record, "actual_answer", case_id),
                retrieved_contexts=_retrieved_texts(actual_record, case_id),
                reference=_require_text(record, "expected_answer", case_id),
            )
        )

    if wanted is not None:
        unknown = sorted(wanted - seen_ids)
        if unknown:
            raise ValueError("Unknown --ids: " + ", ".join(unknown))
    return str(corpus_id), cases


# --------------------------------------------------------------------------
# The judge: one OpenAI model, reached the same way by both frameworks
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class JudgeSettings:
    model: str
    reasoning_effort: str | None
    max_output_tokens: int
    max_concurrent_requests: int
    embedding_model: str = DEFAULT_EMBEDDING_MODEL


@dataclass
class JudgeUsage:
    """What one framework cost, counted per request sent to OpenAI."""

    judge_calls: int = 0
    embedding_calls: int = 0
    failed_requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


class JudgeOutputCutOff(RuntimeError):
    """The model ran out of output tokens before finishing its answer."""


class OpenAIJudge:
    """Thin OpenAI Responses-API client shared by both framework adapters.

    ``gpt-6-luna`` rejects ``temperature``, so a reasoning effort is sent for
    reasoning models and ``temperature=0`` only when no effort is configured
    (for example ``gpt-4o-mini``). Every request is counted when it is sent,
    including failed ones and retries made by the frameworks or by this class;
    transport retries inside the OpenAI SDK are not counted.
    """

    def __init__(self, settings: JudgeSettings, api_key: str) -> None:
        self.settings = settings
        self.usage = JudgeUsage()
        self._usage_lock = threading.Lock()
        client_options: dict[str, Any] = {
            "api_key": api_key,
            "max_retries": 2,
            "timeout": REQUEST_TIMEOUT_SECONDS,
        }
        self._client = OpenAI(**client_options)
        self._async_client = AsyncOpenAI(**client_options)
        self._request_slots = asyncio.Semaphore(settings.max_concurrent_requests)

    def _request_arguments(
        self,
        prompt: str,
        token_budget: int,
        json_schema: type[BaseModel] | None,
        json_mode: bool,
    ) -> dict[str, Any]:
        arguments: dict[str, Any] = {
            "model": self.settings.model,
            "input": prompt,
            "max_output_tokens": token_budget,
        }
        if self.settings.reasoning_effort:
            arguments["reasoning"] = {"effort": self.settings.reasoning_effort}
        else:
            arguments["temperature"] = 0
        if json_schema is not None:
            # Non-strict: the model follows the schema without the strict-mode
            # restrictions (every field required, no defaults).
            arguments["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": json_schema.__name__,
                    "schema": json_schema.model_json_schema(),
                    "strict": False,
                }
            }
        elif json_mode and "json" in prompt.lower():
            arguments["text"] = {"format": {"type": "json_object"}}
        return arguments

    @contextmanager
    def _counted(self, counter: str) -> Iterator[None]:
        with self._usage_lock:
            setattr(self.usage, counter, getattr(self.usage, counter) + 1)
        try:
            yield
        except Exception:
            with self._usage_lock:
                self.usage.failed_requests += 1
            raise

    def _record_tokens(self, response: Any) -> None:
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        with self._usage_lock:
            self.usage.input_tokens += usage.input_tokens or 0
            self.usage.output_tokens += usage.output_tokens or 0

    def _text_of(self, response: Any) -> str:
        self._record_tokens(response)
        if response.status == "incomplete":
            reason = getattr(response.incomplete_details, "reason", "unknown")
            raise JudgeOutputCutOff(f"judge answer cut off ({reason})")
        text = response.output_text.strip()
        if not text:
            raise RuntimeError(f"judge returned no text (status={response.status})")
        return text

    def complete(
        self,
        prompt: str,
        *,
        json_schema: type[BaseModel] | None = None,
        json_mode: bool = False,
    ) -> str:
        for multiplier in TOKEN_BUDGET_MULTIPLIERS:
            budget = self.settings.max_output_tokens * multiplier
            arguments = self._request_arguments(prompt, budget, json_schema, json_mode)
            with self._counted("judge_calls"):
                response = self._client.responses.create(**arguments)
            try:
                return self._text_of(response)
            except JudgeOutputCutOff:
                if multiplier == TOKEN_BUDGET_MULTIPLIERS[-1]:
                    raise
        raise AssertionError("unreachable")

    async def acomplete(
        self,
        prompt: str,
        *,
        json_schema: type[BaseModel] | None = None,
        json_mode: bool = False,
    ) -> str:
        for multiplier in TOKEN_BUDGET_MULTIPLIERS:
            budget = self.settings.max_output_tokens * multiplier
            arguments = self._request_arguments(prompt, budget, json_schema, json_mode)
            async with self._request_slots:
                with self._counted("judge_calls"):
                    response = await self._async_client.responses.create(**arguments)
            try:
                return self._text_of(response)
            except JudgeOutputCutOff:
                if multiplier == TOKEN_BUDGET_MULTIPLIERS[-1]:
                    raise
        raise AssertionError("unreachable")

    def embed(self, texts: list[str]) -> list[list[float]]:
        with self._counted("embedding_calls"):
            response = self._client.embeddings.create(
                model=self.settings.embedding_model, input=texts
            )
        return [item.embedding for item in response.data]

    async def aembed(self, texts: list[str]) -> list[list[float]]:
        async with self._request_slots:
            with self._counted("embedding_calls"):
                response = await self._async_client.embeddings.create(
                    model=self.settings.embedding_model, input=texts
                )
        return [item.embedding for item in response.data]


# --------------------------------------------------------------------------
# Adapter 1: RAGAS  (BaseRagasLLM subclass + embeddings)
# --------------------------------------------------------------------------


def _as_llm_result(texts: list[str]) -> LLMResult:
    # RAGAS expects the n completions of one prompt inside one inner list.
    return LLMResult(generations=[[Generation(text=text) for text in texts]])


class RagasJudgeLLM(BaseRagasLLM):
    """RAGAS judge that ignores the ``temperature`` RAGAS asks for.

    RAGAS calls ``generate`` with ``n`` completions (``n=3`` for answer
    relevancy). The Responses API has no ``n``, so ``n`` separate requests are
    made. All RAGAS prompts ask for a JSON object, so JSON mode is switched on.
    """

    def __init__(self, judge: OpenAIJudge, run_config: RunConfig) -> None:
        super().__init__(run_config=run_config)
        self._judge = judge

    def generate_text(
        self,
        prompt: Any,
        n: int = 1,
        temperature: float | None = None,
        stop: list[str] | None = None,
        callbacks: Any = None,
    ) -> LLMResult:
        text = prompt.to_string()
        texts = [self._judge.complete(text, json_mode=True) for _ in range(n)]
        return _as_llm_result(texts)

    async def agenerate_text(
        self,
        prompt: Any,
        n: int = 1,
        temperature: float | None = None,
        stop: list[str] | None = None,
        callbacks: Any = None,
    ) -> LLMResult:
        text = prompt.to_string()
        texts = await asyncio.gather(
            *(self._judge.acomplete(text, json_mode=True) for _ in range(n))
        )
        return _as_llm_result(list(texts))

    def is_finished(self, response: LLMResult) -> bool:
        # An unfinished answer already raised JudgeOutputCutOff in the judge.
        return True


class RagasJudgeEmbeddings(BaseRagasEmbeddings):
    """OpenAI embeddings for RAGAS answer relevancy, counted by the judge."""

    def __init__(self, judge: OpenAIJudge, run_config: RunConfig) -> None:
        super().__init__()
        self._judge = judge
        self.run_config = run_config

    def embed_query(self, text: str) -> list[float]:
        return self._judge.embed([text])[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._judge.embed(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return (await self._judge.aembed([text]))[0]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self._judge.aembed(texts)


# --------------------------------------------------------------------------
# Adapter 2: DeepEval  (DeepEvalBaseLLM subclass)
# --------------------------------------------------------------------------


class DeepEvalJudgeLLM(DeepEvalBaseLLM):
    """DeepEval judge that never sends ``temperature``.

    DeepEval passes ``schema=<pydantic class>`` when a metric needs structured
    output. The schema goes to the API as a (non-strict) JSON-schema response
    format, the reply is validated into the schema class, and one more attempt
    is made if validation fails. Without a schema the raw text is returned.
    """

    def __init__(self, judge: OpenAIJudge) -> None:
        self._judge = judge
        super().__init__(model=judge.settings.model)

    def load_model(self, *args: Any, **kwargs: Any) -> OpenAIJudge:
        return self._judge

    def generate(self, prompt: str, schema: type[BaseModel] | None = None) -> Any:
        if schema is None:
            return self._judge.complete(prompt)
        last_error: Exception | None = None
        for _ in range(STRUCTURED_OUTPUT_ATTEMPTS):
            text = self._judge.complete(prompt, json_schema=schema)
            try:
                return schema.model_validate_json(text)
            except ValidationError as exc:
                last_error = exc
        raise ValueError(f"judge output did not match {schema.__name__}: {last_error}")

    async def a_generate(
        self, prompt: str, schema: type[BaseModel] | None = None
    ) -> Any:
        if schema is None:
            return await self._judge.acomplete(prompt)
        last_error: Exception | None = None
        for _ in range(STRUCTURED_OUTPUT_ATTEMPTS):
            text = await self._judge.acomplete(prompt, json_schema=schema)
            try:
                return schema.model_validate_json(text)
            except ValidationError as exc:
                last_error = exc
        raise ValueError(f"judge output did not match {schema.__name__}: {last_error}")

    def get_model_name(self, *args: Any, **kwargs: Any) -> str:
        return self._judge.settings.model


# --------------------------------------------------------------------------
# Scoring one case with each framework
# --------------------------------------------------------------------------


@dataclass
class MetricOutcome:
    """A score, or ``None`` plus the reason it could not be computed."""

    score: float | None
    error: str | None = None
    reason: str | None = None


def _describe_error(exc: BaseException) -> str:
    message = re.sub(r"\s+", " ", f"{type(exc).__name__}: {exc}").strip()
    return message[:MAX_ERROR_LENGTH]


def _outcome_from_score(score: Any, framework: str) -> MetricOutcome:
    if score is None:
        return MetricOutcome(None, error=f"{framework} returned no score")
    value = float(score)
    if value != value:  # NaN
        return MetricOutcome(None, error=f"{framework} returned NaN (nothing to score)")
    return MetricOutcome(value)


class RagasScorer:
    """Scores one case with the four RAGAS metrics, all in parallel."""

    def __init__(self, judge: OpenAIJudge, run_config: RunConfig) -> None:
        self._run_config = run_config
        self._llm = RagasJudgeLLM(judge, run_config)
        self._embeddings = RagasJudgeEmbeddings(judge, run_config)

    async def score_case(self, case: EvalCase) -> dict[str, MetricOutcome]:
        sample = SingleTurnSample(
            user_input=case.question,
            response=case.actual_answer,
            retrieved_contexts=list(case.retrieved_contexts),
            reference=case.reference,
        )
        metrics: dict[str, Any] = {
            "faithfulness": Faithfulness(llm=self._llm),
            "answer_relevancy": ResponseRelevancy(
                llm=self._llm, embeddings=self._embeddings
            ),
            "context_recall": LLMContextRecall(llm=self._llm),
            "context_precision": LLMContextPrecisionWithReference(llm=self._llm),
        }
        for metric in metrics.values():
            metric.init(self._run_config)
        outcomes = await asyncio.gather(
            *(self._score_metric(metric, sample) for metric in metrics.values())
        )
        return dict(zip(metrics, outcomes, strict=True))

    async def _score_metric(
        self, metric: Any, sample: SingleTurnSample
    ) -> MetricOutcome:
        try:
            score = await metric.single_turn_ascore(
                sample, timeout=self._run_config.timeout
            )
        except Exception as exc:  # a failed metric is data, not a crash
            return MetricOutcome(None, error=_describe_error(exc))
        return _outcome_from_score(score, "RAGAS")


class DeepEvalScorer:
    """Scores one case with the four DeepEval metrics, all in parallel.

    DeepEval metrics keep their result on the metric object, so a fresh set is
    built for every case.
    """

    def __init__(self, judge: OpenAIJudge) -> None:
        self._llm = DeepEvalJudgeLLM(judge)

    def _build_metrics(self) -> dict[str, Any]:
        options: dict[str, Any] = {
            "threshold": FAILURE_THRESHOLD,
            "model": self._llm,
            "include_reason": True,
            "async_mode": True,
            "eval_mode": "llm",
        }
        return {
            "faithfulness": FaithfulnessMetric(**options),
            "answer_relevancy": AnswerRelevancyMetric(**options),
            "context_recall": ContextualRecallMetric(**options),
            "context_precision": ContextualPrecisionMetric(**options),
        }

    async def score_case(self, case: EvalCase) -> dict[str, MetricOutcome]:
        test_case = LLMTestCase(
            input=case.question,
            actual_output=case.actual_answer,
            expected_output=case.reference,
            retrieval_context=list(case.retrieved_contexts),
        )
        metrics = self._build_metrics()
        outcomes = await asyncio.gather(
            *(self._score_metric(metric, test_case) for metric in metrics.values())
        )
        return dict(zip(metrics, outcomes, strict=True))

    @staticmethod
    async def _score_metric(metric: Any, test_case: LLMTestCase) -> MetricOutcome:
        try:
            score = await metric.a_measure(test_case, _show_indicator=False)
        except Exception as exc:  # a failed metric is data, not a crash
            return MetricOutcome(None, error=_describe_error(exc))
        outcome = _outcome_from_score(score, "DeepEval")
        outcome.reason = getattr(metric, "reason", None)
        return outcome


# --------------------------------------------------------------------------
# Running a framework over every case
# --------------------------------------------------------------------------


@dataclass
class StageResult:
    outcomes: dict[str, dict[str, MetricOutcome]] = field(default_factory=dict)
    wall_seconds: float = 0.0
    started_at: str = ""
    finished_at: str = ""
    usage: JudgeUsage = field(default_factory=JudgeUsage)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


async def run_stage(
    framework: str,
    cases: Sequence[EvalCase],
    score_case: Callable[[EvalCase], Awaitable[dict[str, MetricOutcome]]],
    judge: OpenAIJudge,
    concurrency: int,
) -> StageResult:
    """Score all cases with one framework; the two stages never overlap."""

    result = StageResult(started_at=_utc_now())
    slots = asyncio.Semaphore(concurrency)
    finished = 0

    async def worker(case: EvalCase) -> None:
        nonlocal finished
        async with slots:
            result.outcomes[case.case_id] = await score_case(case)
        finished += 1
        print(
            f"[{framework}] {finished}/{len(cases)} {case.case_id} done",
            file=sys.stderr,
            flush=True,
        )

    started = time.perf_counter()
    await asyncio.gather(*(worker(case) for case in cases))
    result.wall_seconds = round(time.perf_counter() - started, 2)
    result.finished_at = _utc_now()
    result.usage = judge.usage
    return result


# --------------------------------------------------------------------------
# Statistics (stdlib only)
# --------------------------------------------------------------------------


def _mean(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) < MIN_PAIRS_FOR_CORRELATION:
        return None
    try:
        return statistics.correlation(xs, ys)
    except statistics.StatisticsError:  # one side is constant
        return None


def _average_ranks(values: Sequence[float]) -> list[float]:
    """Rank values from 1, giving tied values the mean of their positions."""

    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and values[order[end + 1]] == values[order[start]]:
            end += 1
        shared_rank = (start + end) / 2 + 1
        for tied in order[start : end + 1]:
            ranks[tied] = shared_rank
        start = end + 1
    return ranks


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) < MIN_PAIRS_FOR_CORRELATION:
        return None
    return pearson(_average_ranks(xs), _average_ranks(ys))


def _rounded(value: float | None, digits: int = SCORE_DIGITS) -> float | None:
    return None if value is None else round(value, digits)


# --------------------------------------------------------------------------
# The report
# --------------------------------------------------------------------------


def _scores(
    stage: StageResult, cases: Sequence[EvalCase], metric: str
) -> dict[str, float | None]:
    return {case.case_id: stage.outcomes[case.case_id][metric].score for case in cases}


def _present(scores: dict[str, float | None]) -> list[float]:
    return [score for score in scores.values() if score is not None]


def _paired_scores(
    ragas_scores: dict[str, float | None],
    deepeval_scores: dict[str, float | None],
) -> tuple[list[float], list[float]]:
    ragas_values: list[float] = []
    deepeval_values: list[float] = []
    for case_id, ragas_value in ragas_scores.items():
        deepeval_value = deepeval_scores[case_id]
        if ragas_value is not None and deepeval_value is not None:
            # Stored precision, so a RAGAS 0.9999999999 ties with DeepEval's 1.0.
            ragas_values.append(round(ragas_value, SCORE_DIGITS))
            deepeval_values.append(round(deepeval_value, SCORE_DIGITS))
    return ragas_values, deepeval_values


def _is_failing(score: float | None) -> bool:
    # Compare the score as it is stored (4 decimals). RAGAS divides average
    # precision by (relevant + 1e-10), so a true 0.5 comes out as 0.49999999995
    # and a plain ``< 0.5`` would flag it.
    return score is not None and round(score, SCORE_DIGITS) < FAILURE_THRESHOLD


def _failing_ids(scores: dict[str, float | None]) -> list[str]:
    return [case_id for case_id, score in scores.items() if _is_failing(score)]


def _overlap(ragas_ids: list[str], deepeval_ids: list[str]) -> dict[str, list[str]]:
    ragas_set, deepeval_set = set(ragas_ids), set(deepeval_ids)
    return {
        "ragas": ragas_ids,
        "deepeval": deepeval_ids,
        "both": [case_id for case_id in ragas_ids if case_id in deepeval_set],
        "ragas_only": [i for i in ragas_ids if i not in deepeval_set],
        "deepeval_only": [i for i in deepeval_ids if i not in ragas_set],
    }


def _package_versions() -> dict[str, str]:
    names = (
        "ragas",
        "deepeval",
        "langchain",
        "langchain-core",
        "langchain-community",
        "langchain-openai",
        "openai",
        "pandas",
        "python-dotenv",
    )
    versions = {"python": platform.python_version()}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _case_row(case: EvalCase, stages: dict[str, StageResult]) -> dict[str, Any]:
    row: dict[str, Any] = {"id": case.case_id, "difficulty": case.difficulty}
    for framework, stage in stages.items():
        row[framework] = {}
        for metric, outcome in stage.outcomes[case.case_id].items():
            cell: dict[str, Any] = {
                "score": _rounded(outcome.score),
                "error": outcome.error,
            }
            if framework == "deepeval":
                cell["reason"] = outcome.reason
            row[framework][metric] = cell
    return row


def _metric_agreement(
    ragas_scores: dict[str, float | None],
    deepeval_scores: dict[str, float | None],
) -> dict[str, Any]:
    ragas_paired, deepeval_paired = _paired_scores(ragas_scores, deepeval_scores)
    differences = [d - r for r, d in zip(ragas_paired, deepeval_paired, strict=True)]
    return {
        "paired_n": len(ragas_paired),
        "pearson": _rounded(pearson(ragas_paired, deepeval_paired)),
        "spearman": _rounded(spearman(ragas_paired, deepeval_paired)),
        "mean_difference_deepeval_minus_ragas": _rounded(_mean(differences)),
        "mean_absolute_difference": _rounded(_mean([abs(d) for d in differences])),
        "deepeval_lower_count": sum(1 for d in differences if d < 0),
        "ragas_lower_count": sum(1 for d in differences if d > 0),
        "equal_count": sum(1 for d in differences if d == 0),
    }


def _judge_section(settings: JudgeSettings, judge_note: str | None) -> dict[str, Any]:
    return {
        "model": settings.model,
        "reasoning_effort": settings.reasoning_effort,
        "api": "openai-responses",
        "max_output_tokens": settings.max_output_tokens,
        "embedding_model_ragas_answer_relevancy": settings.embedding_model,
        "same_judge_for_both_frameworks": True,
        "adapters": {
            "ragas": "BaseRagasLLM subclass (RagasJudgeLLM); temperature ignored",
            "deepeval": "DeepEvalBaseLLM subclass (DeepEvalJudgeLLM); no temperature",
        },
        "note": judge_note,
    }


def _framework_section(stages: dict[str, StageResult]) -> dict[str, Any]:
    return {
        framework: {
            "wall_seconds": stage.wall_seconds,
            "started_at": stage.started_at,
            "finished_at": stage.finished_at,
            "judge_calls": stage.usage.judge_calls,
            "embedding_calls": stage.usage.embedding_calls,
            "failed_requests": stage.usage.failed_requests,
            "input_tokens": stage.usage.input_tokens,
            "output_tokens": stage.usage.output_tokens,
        }
        for framework, stage in stages.items()
    }


def build_report(
    cases: Sequence[EvalCase],
    stages: dict[str, StageResult],
    settings: JudgeSettings,
    run_details: dict[str, Any],
    judge_note: str | None,
) -> dict[str, Any]:
    averages: dict[str, Any] = {}
    agreement: dict[str, Any] = {}
    failing_per_metric: dict[str, Any] = {}
    for metric in METRIC_NAMES:
        ragas_scores = _scores(stages["ragas"], cases, metric)
        deepeval_scores = _scores(stages["deepeval"], cases, metric)
        averages[metric] = {
            "ragas": _rounded(_mean(_present(ragas_scores))),
            "ragas_n": len(_present(ragas_scores)),
            "deepeval": _rounded(_mean(_present(deepeval_scores))),
            "deepeval_n": len(_present(deepeval_scores)),
        }
        agreement[metric] = _metric_agreement(ragas_scores, deepeval_scores)
        failing_per_metric[metric] = _overlap(
            _failing_ids(ragas_scores), _failing_ids(deepeval_scores)
        )

    failing_any: dict[str, list[str]] = {name: [] for name in FRAMEWORK_NAMES}
    for case in cases:
        for framework, stage in stages.items():
            outcomes = stage.outcomes[case.case_id].values()
            if any(_is_failing(outcome.score) for outcome in outcomes):
                failing_any[framework].append(case.case_id)

    failed_evaluations = [
        {
            "id": case.case_id,
            "framework": framework,
            "metric": metric,
            "error": outcome.error,
        }
        for case in cases
        for framework, stage in stages.items()
        for metric, outcome in stage.outcomes[case.case_id].items()
        if outcome.score is None
    ]

    return {
        "schema_version": "1.0",
        "generated_at": _utc_now(),
        "judge": _judge_section(settings, judge_note),
        "versions": _package_versions(),
        "config": {
            "failure_threshold": FAILURE_THRESHOLD,
            "case_count": len(cases),
            **run_details,
        },
        "metric_pairs": {
            "faithfulness": ["ragas.Faithfulness", "deepeval.FaithfulnessMetric"],
            "answer_relevancy": [
                "ragas.ResponseRelevancy",
                "deepeval.AnswerRelevancyMetric",
            ],
            "context_recall": [
                "ragas.LLMContextRecall",
                "deepeval.ContextualRecallMetric",
            ],
            "context_precision": [
                "ragas.LLMContextPrecisionWithReference",
                "deepeval.ContextualPrecisionMetric",
            ],
        },
        "frameworks": _framework_section(stages),
        "cases": [_case_row(case, stages) for case in cases],
        "averages": averages,
        "agreement": agreement,
        "failing_cases": {
            "threshold": f"score < {FAILURE_THRESHOLD}",
            "per_metric": failing_per_metric,
            "any_metric": _overlap(failing_any["ragas"], failing_any["deepeval"]),
        },
        "failed_evaluations": failed_evaluations,
    }


# --------------------------------------------------------------------------
# Console output
# --------------------------------------------------------------------------


def _cell(value: float | None) -> str:
    return "  -" if value is None else f"{value:.2f}"


def _short(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _print_case_table(report: dict[str, Any]) -> None:
    print("\nPer case, ragas/deepeval ('-' = metric failed):")
    print(f"{'ID':<6}" + "".join(f"{name:<19}" for name in METRIC_NAMES))
    for row in report["cases"]:
        cells = ""
        for metric in METRIC_NAMES:
            ragas_cell = _cell(row["ragas"][metric]["score"])
            deepeval_cell = _cell(row["deepeval"][metric]["score"])
            cells += f"{ragas_cell}/{deepeval_cell}".ljust(19)
        print(f"{row['id']:<6}{cells}")


def _print_agreement_table(report: dict[str, Any]) -> None:
    print("\nAverages and agreement:")
    print(
        f"{'metric':<18}{'ragas':>8}{'deepeval':>10}{'pearson':>9}"
        f"{'spearman':>10}{'D-R diff':>10}{'pairs':>7}"
    )
    for metric in METRIC_NAMES:
        average, pair = report["averages"][metric], report["agreement"][metric]
        difference = pair["mean_difference_deepeval_minus_ragas"]
        print(
            f"{metric:<18}{_short(average['ragas']):>8}"
            f"{_short(average['deepeval']):>10}{_short(pair['pearson']):>9}"
            f"{_short(pair['spearman']):>10}{_short(difference):>10}"
            f"{pair['paired_n']:>7}"
        )


def print_summary(report: dict[str, Any]) -> None:
    judge = report["judge"]
    print(
        f"\nJudge: {judge['model']} (effort={judge['reasoning_effort']}) for both "
        f"frameworks | cases: {report['config']['case_count']} | "
        f"fail threshold: < {FAILURE_THRESHOLD}"
    )
    _print_case_table(report)
    _print_agreement_table(report)

    print("\nFailing cases (score < threshold), any metric:")
    failing = report["failing_cases"]["any_metric"]
    for key in ("ragas", "deepeval", "both", "ragas_only", "deepeval_only"):
        print(f"  {key:<14}{', '.join(failing[key]) or '-'}")

    print("\nCost:")
    for framework, info in report["frameworks"].items():
        print(
            f"  {framework:<9}{info['wall_seconds']:>8.1f}s "
            f"{info['judge_calls']:>5} judge calls "
            f"{info['embedding_calls']:>3} embedding calls "
            f"{info['failed_requests']:>3} failed "
            f"{info['input_tokens']:>8} in / {info['output_tokens']:>7} out tokens"
        )

    failed = report["failed_evaluations"]
    if failed:
        print(f"\nWARNING: {len(failed)} metric evaluation(s) failed (saved as null):")
        for item in failed[:10]:
            print(
                f"  {item['framework']}/{item['metric']} {item['id']}: "
                f"{item['error']}"
            )


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare RAGAS and DeepEval on the lab golden dataset (Ex. 3.4)."
    )
    artifacts = REPO_ROOT / "artifacts"
    parser.add_argument(
        "--golden", type=Path, default=REPO_ROOT / "golden_dataset.json"
    )
    parser.add_argument(
        "--answers", type=Path, default=artifacts / "actual_answers.json"
    )
    parser.add_argument(
        "--output", type=Path, default=artifacts / "framework_comparison.json"
    )
    parser.add_argument(
        "--ids",
        nargs="+",
        metavar="ID",
        help="Only run these case ids (space or comma separated), e.g. --ids E01 H03",
    )
    parser.add_argument(
        "--judge-model",
        default=os.environ.get("OPENAI_MODEL", "").strip(),
        help="Judge for BOTH frameworks (default: OPENAI_MODEL from .env)",
    )
    parser.add_argument(
        "--reasoning-effort",
        default=os.environ.get("OPENAI_REASONING_EFFORT", "").strip(),
        help="Judge reasoning effort; empty sends temperature=0 (default: from .env)",
    )
    parser.add_argument("--max-output-tokens", type=int, default=4000)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=3,
        help="Cases scored at the same time per framework",
    )
    parser.add_argument(
        "--max-concurrent-requests",
        type=int,
        default=8,
        help="Cap on in-flight judge requests",
    )
    parser.add_argument(
        "--judge-note",
        default="",
        help="Free text stored in the report, e.g. why a fallback judge was used",
    )
    args = parser.parse_args(argv)
    if args.ids:
        args.ids = [i for chunk in args.ids for i in chunk.split(",") if i]
    return args


async def run_comparison(
    cases: Sequence[EvalCase],
    settings: JudgeSettings,
    api_key: str,
    concurrency: int,
) -> dict[str, StageResult]:
    # Two judge objects with identical settings, so each framework's calls,
    # tokens and time are counted separately while the judging is identical.
    ragas_judge = OpenAIJudge(settings, api_key)
    deepeval_judge = OpenAIJudge(settings, api_key)
    ragas_run_config = RunConfig(timeout=180, max_retries=3, max_wait=10)
    ragas_scorer = RagasScorer(ragas_judge, ragas_run_config)
    deepeval_scorer = DeepEvalScorer(deepeval_judge)

    ragas_stage = await run_stage(
        "ragas", cases, ragas_scorer.score_case, ragas_judge, concurrency
    )
    deepeval_stage = await run_stage(
        "deepeval", cases, deepeval_scorer.score_case, deepeval_judge, concurrency
    )
    return {"ragas": ragas_stage, "deepeval": deepeval_stage}


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv(REPO_ROOT / ".env")
    args = parse_args(argv)

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        print("ERROR: OPENAI_API_KEY is missing; put it in the repository .env")
        return 2
    if not args.judge_model:
        print("ERROR: no judge model; set OPENAI_MODEL in .env or pass --judge-model")
        return 2
    try:
        corpus_id, cases = load_cases(args.golden, args.answers, args.ids)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 2

    settings = JudgeSettings(
        model=args.judge_model,
        reasoning_effort=args.reasoning_effort or None,
        max_output_tokens=args.max_output_tokens,
        max_concurrent_requests=args.max_concurrent_requests,
    )
    print(
        f"Comparing RAGAS vs DeepEval on {len(cases)} case(s), "
        f"judge={settings.model} (effort={settings.reasoning_effort})",
        file=sys.stderr,
    )
    stages = asyncio.run(run_comparison(cases, settings, api_key, args.concurrency))

    run_details: dict[str, Any] = {
        "corpus_id": corpus_id,
        "case_ids": [case.case_id for case in cases],
        "golden_path": _display_path(args.golden),
        "answers_path": _display_path(args.answers),
        "concurrency_cases": args.concurrency,
        "max_concurrent_requests": args.max_concurrent_requests,
    }
    report = build_report(cases, stages, settings, run_details, args.judge_note or None)

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print_summary(report)
    print(f"\nSaved comparison: {_display_path(output)}")
    return 1 if report["failed_evaluations"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
