import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

Chunks = list[dict]
Retrieve = Callable[[str, str], Chunks]
Generate = Callable[[str, Chunks], Awaitable[str]]
Score = Callable[[str, str, Chunks], int]
Rewrite = Callable[[str, str, Chunks], Awaitable[str]]
Decompose = Callable[[str], Awaitable[list[str]]]
Contextualize = Callable[[str, list[dict]], Awaitable[str]]
RouteSpecifications = Callable[[str, list[str]], Awaitable[list[str]]]
GraphRetrieve = Callable[[str, list[str]], Chunks]


@dataclass
class AgentResult:
    answer: str
    chunks: Chunks
    confidence: int
    attempts: int
    retried: bool
    selected_attempt: int
    subqueries: list[str] = field(default_factory=list)
    resolved_question: str = ""
    selected_specs: list[str] = field(default_factory=list)


class AgentState(TypedDict, total=False):
    """Serializable request state passed between the LangGraph nodes."""

    original_question: str
    history: list[dict]
    requested_specification: str
    available_specifications: list[str]
    resolved_question: str
    selected_specifications: list[str]
    subqueries: list[str]
    retrieval_query: str
    chunks: Chunks
    answer: str
    confidence: int
    attempts: int
    retried: bool
    selected_attempt: int
    best_answer: str
    best_chunks: Chunks
    best_confidence: int
    stop_retry: bool


@dataclass(frozen=True)
class AgentContext:
    """Run-scoped dependencies and limits; these are not persisted in state."""

    retrieve: GraphRetrieve
    generate: Generate
    score: Score
    rewrite: Rewrite
    decompose: Decompose | None = None
    contextualize: Contextualize | None = None
    route: RouteSpecifications | None = None
    confidence_threshold: int = 70
    workflow_confidence_threshold: int | None = None
    max_subqueries: int = 3
    max_chunks: int = 15
    max_specs: int = 3
    max_attempts: int = 2
    all_specifications_value: str = "__all__"


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
    workflow_confidence_threshold: int | None = None,
    decompose: Decompose | None = None,
    max_subqueries: int = 3,
    max_chunks: int = 15,
) -> AgentResult:
    """Compatibility wrapper around the LangGraph confidence/retry flow."""

    async def contextualize(current_question: str, _history: list[dict]) -> str:
        return current_question

    async def route(_question: str, _available: list[str]) -> list[str]:
        return [specification_name]

    return await run_agent_graph(
        question=question,
        history=[],
        specification_name=specification_name,
        available_specifications=[specification_name],
        contextualize=contextualize,
        route=route,
        retrieve=lambda query, _selected: retrieve(query, specification_name),
        generate=generate,
        score=score,
        rewrite=rewrite,
        confidence_threshold=confidence_threshold,
        workflow_confidence_threshold=workflow_confidence_threshold,
        decompose=decompose,
        max_subqueries=max_subqueries,
        max_chunks=max_chunks,
    )


async def run_agent_graph(
    question: str,
    history: list[dict],
    specification_name: str,
    available_specifications: list[str],
    contextualize: Contextualize,
    route: RouteSpecifications,
    retrieve: GraphRetrieve,
    generate: Generate,
    score: Score,
    rewrite: Rewrite,
    confidence_threshold: int = 70,
    workflow_confidence_threshold: int | None = None,
    decompose: Decompose | None = None,
    max_subqueries: int = 3,
    max_chunks: int = 15,
    max_specs: int = 3,
    max_attempts: int = 2,
    all_specifications_value: str = "__all__",
) -> AgentResult:
    """Execute the complete query-agent StateGraph for one request."""
    context = AgentContext(
        contextualize=contextualize,
        route=route,
        retrieve=retrieve,
        generate=generate,
        score=score,
        rewrite=rewrite,
        decompose=decompose,
        confidence_threshold=confidence_threshold,
        workflow_confidence_threshold=workflow_confidence_threshold,
        max_subqueries=max_subqueries,
        max_chunks=max_chunks,
        max_specs=max_specs,
        max_attempts=max(1, max_attempts),
        all_specifications_value=all_specifications_value,
    )
    state = await get_agent_graph().ainvoke(
        {
            "original_question": question,
            "history": history,
            "requested_specification": specification_name,
            "available_specifications": available_specifications,
            "attempts": 1,
            "retried": False,
            "selected_attempt": 1,
            "best_answer": "",
            "best_chunks": [],
            "best_confidence": 0,
            "stop_retry": False,
        },
        context=context,
    )
    return AgentResult(
        answer=state.get("best_answer", ""),
        chunks=state.get("best_chunks", []),
        confidence=state.get("best_confidence", 0),
        attempts=state.get("attempts", 1),
        retried=state.get("retried", False),
        selected_attempt=state.get("selected_attempt", 1),
        subqueries=state.get("subqueries", [state.get("resolved_question", question)]),
        resolved_question=state.get("resolved_question", question),
        selected_specs=state.get("selected_specifications", [specification_name]),
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
        retrieved = _prioritize_operation_matches(
            query, retrieve(query, specification_name) or []
        )
        for chunk in retrieved:
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
            merged.extend(_tag_workflow_chunks([chunk], query, queries))
            added_for_query += 1
            if added_for_query >= per_query_limit or len(merged) >= max_chunks:
                break
    return merged


def _tag_workflow_chunks(
    chunks: Chunks,
    retrieval_query: str,
    workflow_steps: list[str],
) -> Chunks:
    """Attach request-local workflow context without mutating stored chunks."""
    if len(workflow_steps) <= 1:
        return chunks
    return [
        {
            **chunk,
            "metadata": {
                **chunk.get("metadata", {}),
                "retrieval_query": retrieval_query,
                "workflow_steps": list(workflow_steps),
            },
        }
        for chunk in chunks
    ]


def _prioritize_operation_matches(query: str, chunks: Chunks) -> Chunks:
    """Prefer endpoints whose method and summary match an explicit API action."""
    query_words = set(re.findall(r"[a-z]+", query.casefold()))
    action_methods = {
        "create": "POST",
        "attach": "POST",
        "add": "POST",
        "list": "GET",
        "get": "GET",
        "retrieve": "GET",
        "delete": "DELETE",
        "remove": "DELETE",
        "update": "PATCH",
    }
    requested_actions = query_words.intersection(action_methods)
    if not requested_actions:
        return chunks

    def priority(chunk: dict) -> int:
        metadata = chunk.get("metadata", {})
        summary_words = set(
            re.findall(r"[a-z]+", str(metadata.get("summary", "")).casefold())
        )
        method = str(metadata.get("method", "")).upper()
        action_score = 4 * len(requested_actions.intersection(summary_words))
        method_score = 2 if any(
            method == action_methods[action] for action in requested_actions
        ) else 0
        return action_score + method_score

    ranked = sorted(chunks, key=priority, reverse=True)
    if all(before is after for before, after in zip(chunks, ranked)):
        return chunks
    return ranked


def _workflow_adjusted_score(
    grounding_score: int,
    answer: str,
    workflow_steps: list[str],
) -> int:
    """Penalize grounded answers that omit requested workflow operations."""
    expected = len(workflow_steps)
    if expected <= 1:
        return grounding_score
    answered = {
        int(number)
        for number in re.findall(r"\bStep\s+(\d+)\b", answer, re.IGNORECASE)
        if 1 <= int(number) <= expected
    }
    completeness = len(answered) / expected
    return round(grounding_score * completeness)


def _score_with_workflow_completeness(
    score: Score,
    question: str,
    answer: str,
    chunks: Chunks,
    workflow_steps: list[str],
) -> int:
    confidence = score(question, answer, chunks)
    if getattr(score, "includes_workflow_completeness", False):
        return confidence
    return _workflow_adjusted_score(confidence, answer, workflow_steps)


async def _contextualize_node(
    state: AgentState, runtime: Runtime[AgentContext]
) -> dict:
    context = runtime.context
    if context.contextualize is None:
        resolved = state["original_question"]
    else:
        resolved = await resolve_follow_up(
            state["original_question"],
            state.get("history", []),
            context.contextualize,
        )
    return {"resolved_question": resolved}


async def _route_specifications_node(
    state: AgentState, runtime: Runtime[AgentContext]
) -> dict:
    context = runtime.context
    requested = state["requested_specification"]
    if requested != context.all_specifications_value:
        return {"selected_specifications": [requested]}

    if context.route is None:
        selected = state.get("available_specifications", [])[:context.max_specs]
    else:
        selected = await select_specifications(
            state["resolved_question"],
            state.get("available_specifications", []),
            context.route,
            max_specs=context.max_specs,
        )
    return {"selected_specifications": selected}


async def _decompose_node(
    state: AgentState, runtime: Runtime[AgentContext]
) -> dict:
    subqueries = await _build_query_plan(
        state["resolved_question"],
        runtime.context.decompose,
        runtime.context.max_subqueries,
    )
    return {"subqueries": subqueries}


async def _retrieve_node(
    state: AgentState, runtime: Runtime[AgentContext]
) -> dict:
    context = runtime.context
    selected = state.get("selected_specifications", [])
    subqueries = state["subqueries"]

    if state.get("attempts", 1) > 1:
        retrieval_query = state["retrieval_query"]
        chunks = context.retrieve(retrieval_query, selected) or []
        chunks = _tag_workflow_chunks(
            _prioritize_operation_matches(retrieval_query, chunks),
            retrieval_query,
            subqueries,
        )
        if len(chunks) > context.max_chunks:
            chunks = chunks[:context.max_chunks]
    else:
        chunks = _retrieve_and_merge(
            subqueries,
            state["requested_specification"],
            lambda query, _specification: context.retrieve(query, selected),
            max_chunks=context.max_chunks,
        )
    return {"chunks": chunks}


async def _route_after_retrieval(
    state: AgentState,
) -> Literal["generate", "finalize"]:
    return "generate" if state.get("chunks") else "finalize"


async def _generate_node(
    state: AgentState, runtime: Runtime[AgentContext]
) -> dict:
    try:
        answer = await runtime.context.generate(
            state["resolved_question"], state["chunks"]
        )
    except Exception:
        if state.get("attempts", 1) == 1:
            raise
        answer = ""

    update = {"answer": answer or ""}
    if state.get("attempts", 1) == 1 and not answer:
        update.update({"best_chunks": state["chunks"], "best_confidence": 0})
    return update


async def _route_after_generation(
    state: AgentState,
) -> Literal["confidence", "finalize"]:
    return "confidence" if state.get("answer") else "finalize"


async def _confidence_node(
    state: AgentState, runtime: Runtime[AgentContext]
) -> dict:
    context = runtime.context
    try:
        confidence = _score_with_workflow_completeness(
            context.score,
            state["resolved_question"],
            state["answer"],
            state["chunks"],
            state["subqueries"],
        )
    except Exception:
        if state.get("attempts", 1) == 1:
            raise
        return {"confidence": 0, "stop_retry": True}

    update = {"confidence": confidence}
    if not state.get("best_answer") or confidence > state.get("best_confidence", 0):
        update.update({
            "best_answer": state["answer"],
            "best_chunks": state["chunks"],
            "best_confidence": confidence,
            "selected_attempt": state.get("attempts", 1),
        })
    return update


async def _route_after_confidence(
    state: AgentState, runtime: Runtime[AgentContext]
) -> Literal["finalize", "prepare_workflow_retry", "rewrite"]:
    context = runtime.context
    threshold = (
        context.workflow_confidence_threshold
        if len(state["subqueries"]) > 1
        and context.workflow_confidence_threshold is not None
        else context.confidence_threshold
    )
    if (
        state.get("stop_retry")
        or state.get("confidence", 0) >= threshold
        or state.get("attempts", 1) >= context.max_attempts
    ):
        return "finalize"
    if len(state["subqueries"]) > 1:
        return "prepare_workflow_retry"
    return "rewrite"


async def _prepare_workflow_retry_node(state: AgentState) -> dict:
    # Keep the balanced evidence intact; only ask the model to regenerate.
    return {
        "attempts": state.get("attempts", 1) + 1,
        "retried": True,
        "answer": "",
    }


async def _rewrite_node(
    state: AgentState, runtime: Runtime[AgentContext]
) -> dict:
    try:
        refined_query = (
            await runtime.context.rewrite(
                state["resolved_question"],
                state["best_answer"],
                state["best_chunks"],
            )
        ).strip()
    except Exception:
        return {
            "attempts": state.get("attempts", 1) + 1,
            "retried": True,
            "stop_retry": True,
        }

    if (
        not refined_query
        or refined_query.casefold()
        == state["resolved_question"].strip().casefold()
    ):
        return {"stop_retry": True}
    return {
        "retrieval_query": refined_query,
        "attempts": state.get("attempts", 1) + 1,
        "retried": True,
        "stop_retry": False,
        "answer": "",
    }


async def _route_after_rewrite(
    state: AgentState,
) -> Literal["retrieve", "finalize"]:
    return "finalize" if state.get("stop_retry") else "retrieve"


async def _finalize_node(state: AgentState) -> dict:
    return {
        "answer": state.get("best_answer", ""),
        "chunks": state.get("best_chunks", state.get("chunks", [])),
        "confidence": state.get("best_confidence", 0),
    }


@lru_cache(maxsize=1)
def get_agent_graph():
    """Build and cache the request-level LangGraph once per process."""
    graph = StateGraph(AgentState, context_schema=AgentContext)
    graph.add_node("contextualize", _contextualize_node)
    graph.add_node("route_specifications", _route_specifications_node)
    graph.add_node("decompose", _decompose_node)
    graph.add_node("retrieve", _retrieve_node)
    graph.add_node("generate", _generate_node)
    graph.add_node("confidence", _confidence_node)
    graph.add_node("prepare_workflow_retry", _prepare_workflow_retry_node)
    graph.add_node("rewrite", _rewrite_node)
    graph.add_node("finalize", _finalize_node)

    graph.add_edge(START, "contextualize")
    graph.add_edge("contextualize", "route_specifications")
    graph.add_edge("route_specifications", "decompose")
    graph.add_edge("decompose", "retrieve")
    graph.add_conditional_edges("retrieve", _route_after_retrieval)
    graph.add_conditional_edges("generate", _route_after_generation)
    graph.add_conditional_edges("confidence", _route_after_confidence)
    graph.add_edge("prepare_workflow_retry", "generate")
    graph.add_conditional_edges("rewrite", _route_after_rewrite)
    graph.add_edge("finalize", END)
    return graph.compile()


def order_chunks_by_answer_citations(answer: str, chunks: Chunks) -> Chunks:
    """Place endpoints cited by the answer first, preserving all stable ties."""
    matches = re.findall(
        r"\*\*Endpoint:\*\*\s*`?\s*([A-Za-z]+)\s+([^`\s*]+)",
        answer,
    )
    cited = []
    for method, path in matches:
        key = (method.upper(), path.rstrip(".,;").rstrip("/") or "/")
        if key not in cited:
            cited.append(key)
    if not cited:
        return chunks

    citation_rank = {key: rank for rank, key in enumerate(cited)}

    def endpoint_key(chunk: dict) -> tuple[str, str]:
        metadata = chunk.get("metadata", {})
        path = str(metadata.get("path", "")).rstrip("/") or "/"
        return str(metadata.get("method", "")).upper(), path

    return sorted(
        chunks,
        key=lambda chunk: citation_rank.get(endpoint_key(chunk), len(cited)),
    )
