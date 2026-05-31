from dataclasses import dataclass
from typing import Awaitable, Callable, Dict, List


Chunks = List[Dict]
Retrieve = Callable[[str, str], Chunks]
Generate = Callable[[str, Chunks], Awaitable[str]]
Score = Callable[[str, str, Chunks], int]
Rewrite = Callable[[str, str, Chunks], Awaitable[str]]


@dataclass
class QueryAttempt:
    retrieval_query: str
    answer: str
    chunks: Chunks
    confidence: int


@dataclass
class AgentResult:
    answer: str
    chunks: Chunks
    confidence: int
    attempts: int
    retried: bool
    selected_attempt: int


async def run_confidence_retry(
    question: str,
    specification_name: str,
    retrieve: Retrieve,
    generate: Generate,
    score: Score,
    rewrite: Rewrite,
    confidence_threshold: int = 70,
) -> AgentResult:
    """Run the normal RAG path and make one corrective attempt when needed."""
    chunks = retrieve(question, specification_name)
    if not chunks:
        return AgentResult("", [], 0, attempts=1, retried=False, selected_attempt=1)

    answer = await generate(question, chunks)
    if not answer:
        return AgentResult("", chunks, 0, attempts=1, retried=False, selected_attempt=1)

    first = QueryAttempt(
        retrieval_query=question,
        answer=answer,
        chunks=chunks,
        confidence=score(question, answer, chunks),
    )
    if first.confidence >= confidence_threshold:
        return _to_result(first, attempts=1, retried=False, selected_attempt=1)

    try:
        refined_query = (await rewrite(question, first.answer, first.chunks)).strip()
        if not refined_query or refined_query.casefold() == question.strip().casefold():
            return _to_result(first, attempts=1, retried=False, selected_attempt=1)

        retry_chunks = retrieve(refined_query, specification_name)
        if not retry_chunks:
            return _to_result(first, attempts=2, retried=True, selected_attempt=1)

        retry_answer = await generate(question, retry_chunks)
        if not retry_answer:
            return _to_result(first, attempts=2, retried=True, selected_attempt=1)

        second = QueryAttempt(
            retrieval_query=refined_query,
            answer=retry_answer,
            chunks=retry_chunks,
            confidence=score(question, retry_answer, retry_chunks),
        )
    except Exception:
        # Self-correction is best-effort; a usable first answer must not be lost.
        return _to_result(first, attempts=2, retried=True, selected_attempt=1)

    if second.confidence > first.confidence:
        return _to_result(second, attempts=2, retried=True, selected_attempt=2)
    return _to_result(first, attempts=2, retried=True, selected_attempt=1)


def _to_result(
    attempt: QueryAttempt,
    attempts: int,
    retried: bool,
    selected_attempt: int,
) -> AgentResult:
    return AgentResult(
        answer=attempt.answer,
        chunks=attempt.chunks,
        confidence=attempt.confidence,
        attempts=attempts,
        retried=retried,
        selected_attempt=selected_attempt,
    )
