from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

Chunks = list[dict]
Retrieve = Callable[[str, str], Chunks]
Generate = Callable[[str, Chunks], Awaitable[str]]
Score = Callable[[str, str, Chunks], int]
Rewrite = Callable[[str, str, Chunks], Awaitable[str]]
Decompose = Callable[[str], Awaitable[list[str]]]
Contextualize = Callable[[str, list[dict]], Awaitable[str]]
RouteSpecifications = Callable[[str, list[str]], Awaitable[list[str]]]


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
    subqueries: list[str] = field(default_factory=list)


async def resolve_follow_up(
    question: str,
    history: list[dict],
    contextualize: Contextualize,
    max_turns: int = 3,
) -> str:
    """Resolve conversational references while preserving a safe fallback."""
    normalized_history = []
    for turn in history[-max(1, max_turns):]:
        if not isinstance(turn, dict):
            continue
        previous_question = str(turn.get("question", "")).strip()[:1000]
        previous_answer = str(turn.get("answer", "")).strip()[:2000]
        if previous_question and previous_answer:
            normalized_history.append({
                "question": previous_question,
                "answer": previous_answer,
            })
    if not normalized_history:
        return question

    try:
        resolved = await contextualize(question, normalized_history)
    except Exception:
        return question
    resolved = " ".join(resolved.strip().strip('"\'').splitlines()[:1])[:1000]
    return resolved or question


async def select_specifications(
    question: str,
    available: list[str],
    route: RouteSpecifications,
    max_specs: int = 3,
) -> list[str]:
    """Select a bounded set of collections for a cross-spec query."""
    available = list(dict.fromkeys(name for name in available if name))
    limit = max(1, max_specs)
    if len(available) <= limit:
        return available

    try:
        proposed = await route(question, available)
    except Exception:
        return available[:limit]

    by_name = {name.casefold(): name for name in available}
    selected = []
    for name in proposed if isinstance(proposed, list) else []:
        if not isinstance(name, str):
            continue
        canonical = by_name.get(name.strip().casefold())
        if canonical and canonical not in selected:
            selected.append(canonical)
        if len(selected) >= limit:
            break
    return selected or available[:limit]


async def run_confidence_retry(
    question: str,
    specification_name: str,
    retrieve: Retrieve,
    generate: Generate,
    score: Score,
    rewrite: Rewrite,
    confidence_threshold: int = 70,
    decompose: Decompose | None = None,
    max_subqueries: int = 3,
    max_chunks: int = 15,
) -> AgentResult:
    """Run the normal RAG path and make one corrective attempt when needed."""
    subqueries = await _build_query_plan(question, decompose, max_subqueries)
    chunks = _retrieve_and_merge(
        subqueries, specification_name, retrieve, max_chunks=max_chunks
    )
    if not chunks:
        return AgentResult(
            "", [], 0, attempts=1, retried=False, selected_attempt=1,
            subqueries=subqueries,
        )

    answer = await generate(question, chunks)
    if not answer:
        return AgentResult(
            "", chunks, 0, attempts=1, retried=False, selected_attempt=1,
            subqueries=subqueries,
        )

    first = QueryAttempt(
        retrieval_query=question,
        answer=answer,
        chunks=chunks,
        confidence=score(question, answer, chunks),
    )
    if first.confidence >= confidence_threshold:
        return _to_result(
            first, attempts=1, retried=False, selected_attempt=1,
            subqueries=subqueries,
        )

    try:
        refined_query = (await rewrite(question, first.answer, first.chunks)).strip()
        if not refined_query or refined_query.casefold() == question.strip().casefold():
            return _to_result(
                first, attempts=1, retried=False, selected_attempt=1,
                subqueries=subqueries,
            )

        retry_chunks = retrieve(refined_query, specification_name) or []
        if len(retry_chunks) > max_chunks:
            retry_chunks = retry_chunks[:max_chunks]
        if not retry_chunks:
            return _to_result(
                first, attempts=2, retried=True, selected_attempt=1,
                subqueries=subqueries,
            )

        retry_answer = await generate(question, retry_chunks)
        if not retry_answer:
            return _to_result(
                first, attempts=2, retried=True, selected_attempt=1,
                subqueries=subqueries,
            )

        second = QueryAttempt(
            retrieval_query=refined_query,
            answer=retry_answer,
            chunks=retry_chunks,
            confidence=score(question, retry_answer, retry_chunks),
        )
    except Exception:
        # Self-correction is best-effort; a usable first answer must not be lost.
        return _to_result(
            first, attempts=2, retried=True, selected_attempt=1,
            subqueries=subqueries,
        )

    if second.confidence > first.confidence:
        return _to_result(
            second, attempts=2, retried=True, selected_attempt=2,
            subqueries=subqueries,
        )
    return _to_result(
        first, attempts=2, retried=True, selected_attempt=1,
        subqueries=subqueries,
    )


async def _build_query_plan(
    question: str,
    decompose: Decompose | None,
    max_subqueries: int,
) -> list[str]:
    if decompose is None:
        return [question]
    try:
        proposed = await decompose(question)
    except Exception:
        return [question]

    unique = []
    seen = set()
    for query in proposed if isinstance(proposed, list) else []:
        if not isinstance(query, str):
            continue
        query = " ".join(query.split()).strip()
        key = query.casefold()
        if query and key not in seen:
            seen.add(key)
            unique.append(query[:500])
        if len(unique) >= max(1, max_subqueries):
            break
    return unique or [question]


def _retrieve_and_merge(
    queries: list[str],
    specification_name: str,
    retrieve: Retrieve,
    max_chunks: int,
) -> Chunks:
    merged = []
    seen = set()
    per_query_limit = max(1, max_chunks // max(1, len(queries)))
    for query in queries:
        added_for_query = 0
        for chunk in retrieve(query, specification_name) or []:
            metadata = chunk.get("metadata", {})
            key = (
                metadata.get("collection_name")
                or metadata.get("specification_name", ""),
                metadata.get("method", ""),
                metadata.get("path", ""),
            )
            if not any(key):
                key = ("text", "", chunk.get("text", ""))
            if key in seen:
                continue
            seen.add(key)
            merged.append(chunk)
            added_for_query += 1
            if added_for_query >= per_query_limit or len(merged) >= max_chunks:
                break
    return merged


def _to_result(
    attempt: QueryAttempt,
    attempts: int,
    retried: bool,
    selected_attempt: int,
    subqueries: list[str],
) -> AgentResult:
    return AgentResult(
        answer=attempt.answer,
        chunks=attempt.chunks,
        confidence=attempt.confidence,
        attempts=attempts,
        retried=retried,
        selected_attempt=selected_attempt,
        subqueries=subqueries,
    )
