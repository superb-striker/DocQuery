import os
from contextvars import ContextVar

import httpx
from dotenv import load_dotenv
from loggerConfig import get_logger

logger = get_logger("llm")

# Environment
load_dotenv()

PROVIDERS = {
    "groq": {
        "label": "Groq",
        "api_url": "https://api.groq.com/openai/v1/chat/completions",
        "api_key_env": "GROQ_API_KEY",
        "model": "openai/gpt-oss-120b",
        # Keep Groq's free-tier token use deliberately lower.
        "max_chunks": 8,
    },
    "gemini": {
        "label": "Gemini",
        "api_url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        "api_key_env": "GEMINI_API_KEY",
        "model": "gemini-3.8-flash",
        "max_chunks": 15,
    },
    "huggingface": {
        "label": "Hugging Face",
        "api_url": "https://router.huggingface.co/v1/chat/completions",
        "api_key_env": "HF_TOKEN",
        "model": "meta-llama/Llama-3.1-8B-Instruct:novita",
        "max_chunks": 15,
    },
}

_LLM_METRICS: ContextVar[dict | None] = ContextVar("llm_metrics", default=None)


class LLMProviderError(RuntimeError):
    """A safe, user-facing error from the selected LLM provider."""


def get_provider_config(provider: str) -> dict:
    try:
        return PROVIDERS[provider]
    except KeyError as exc:
        supported = ", ".join(PROVIDERS)
        raise LLMProviderError(
            f"Unknown LLM provider '{provider}'. Choose one of: {supported}."
        ) from exc


def provider_catalog() -> list[dict]:
    """Return provider metadata without exposing API keys."""
    return [
        {
            "id": provider,
            "label": config["label"],
            "model": config["model"],
            "max_chunks": config["max_chunks"],
            "available": bool(os.getenv(config["api_key_env"])),
        }
        for provider, config in PROVIDERS.items()
    ]


def reset_llm_metrics() -> None:
    _LLM_METRICS.set({
        "calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    })


def get_llm_metrics() -> dict:
    return dict(_LLM_METRICS.get() or {})


def _record_usage(data: dict) -> None:
    current = dict(_LLM_METRICS.get() or {})
    usage = data.get("usage") or {}
    current["calls"] = current.get("calls", 0) + 1
    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
        current[field] = current.get(field, 0) + int(usage.get(field) or 0)
    _LLM_METRICS.set(current)


def build_llm_request(prompt: str, provider: str) -> tuple[str, dict, dict]:
    """Build an OpenAI-compatible request for a configured provider."""
    config = get_provider_config(provider)
    api_key = os.getenv(config["api_key_env"])
    if not api_key:
        raise LLMProviderError(
            f"{config['label']} is not configured. Add {config['api_key_env']} to .env."
        )

    payload = {
        "model": config["model"],
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 600,
        "temperature": 0.1,
        # Groq accepts at most four stop sequences; these work for every provider.
        "stop": ["USER QUESTION:", "[INST]", "SPEC SECTIONS:", "\n\nQUESTION"],
    }
    if provider == "gemini":
        payload["reasoning_effort"] = "low"
    return (
        config["api_url"],
        {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        payload,
    )

# Prompt Builder 
def build_prompt(query: str, chunks: list[dict]) -> str:
    if not query:
        logger.warning("Empty query passed to build_prompt.")
    if not chunks:
        logger.warning("No chunks provided to build_prompt.")
    try:
        context = "\n\n---\n\n".join(
            f"[Specification: "
            f"{c.get('metadata', {}).get('collection_name') or c.get('metadata', {}).get('specification_name', 'Unknown')}]\n"
            f"{c['text']}"
            for c in chunks
        )
    except KeyError as e:
        logger.error(f"Chunk missing 'text' field: {e}")
        context = ""
    prompt = f"""You are an API documentation assistant. Answer using ONLY the SPEC SECTIONS below.
    
            OUTPUT FORMAT - follow this exactly:
            **Endpoint:** [HTTP METHOD] [path]
            **Summary:** [one sentence from the spec]
            **Parameters:** [list only parameters explicitly named in the spec, or "None listed in spec"]
            **Request Body:** [list only fields explicitly named in the spec, or "Not specified in spec"]
            **Notes:** [any other relevant details from the spec only, do not reference other things in this spec.]

            If the spec sections do not contain enough information to fill a field, write "Not specified in spec" for that field.
            Do not add information from outside the spec. 

            SPEC SECTIONS:
            {context}

            QUESTION: {query}

            ANSWER:"""
    logger.debug("Prompt built successfully.")
    return prompt


def build_retrieval_rewrite_prompt(
    question: str,
    answer: str,
    chunks: list[dict],
) -> str:
    """Build a prompt that turns a weak first attempt into a better search query."""
    endpoints = "\n".join(
        f"- {chunk.get('metadata', {}).get('method', '')} "
        f"{chunk.get('metadata', {}).get('path', '')}: "
        f"{chunk.get('metadata', {}).get('summary', '')}"
        for chunk in chunks
    )
    return f"""You rewrite search queries for an OpenAPI documentation retriever.

The first retrieval produced a weakly grounded answer. Write one improved,
standalone search query that preserves the user's intent while adding useful API
terms, operation concepts, field names, or HTTP verbs. Avoid the endpoints that
are clearly irrelevant. Do not answer the question and do not explain your work.
Return only the rewritten search query on one line.

ORIGINAL USER QUESTION:
{question}

FIRST ANSWER:
{answer}

ENDPOINTS ALREADY RETRIEVED:
{endpoints}

REWRITTEN SEARCH QUERY:"""


def build_query_decomposition_prompt(question: str) -> str:
    """Build a prompt that plans retrieval for compound API questions."""
    return f"""You plan searches over OpenAPI endpoint documentation.

If the user's question requires distinct API operations, split it into separate,
standalone retrieval queries. If it is a simple question, keep it as one query.
Return only a valid JSON array of one to three strings. Preserve concrete API
nouns, field names, and constraints. Do not answer the question.

USER QUESTION:
{question}

JSON QUERY PLAN:"""


def build_contextualization_prompt(question: str, history: list[dict]) -> str:
    """Build a prompt that resolves references in a conversational follow-up."""
    conversation = "\n\n".join(
        f"Previous question: {turn['question']}\nPrevious answer: {turn['answer']}"
        for turn in history
    )
    return f"""Rewrite the latest API documentation question as a standalone query.

Use the conversation only to resolve references such as "that endpoint", "it",
or "those fields". Preserve the latest question's intent. If it already stands
alone or changes topic, return it unchanged. Do not answer the question and do
not add facts. Return only one standalone question on one line.

CONVERSATION:
{conversation}

LATEST QUESTION:
{question}

STANDALONE QUESTION:"""


def build_specification_routing_prompt(
    question: str,
    specification_names: list[str],
) -> str:
    """Build a prompt that selects relevant API specifications."""
    choices = "\n".join(f"- {name}" for name in specification_names)
    return f"""Select the API specifications relevant to the user's question.

Return only a valid JSON array containing one to three exact names from the
available list. Select multiple specifications when the user asks for a
comparison or a workflow spanning APIs. Do not answer the question.

AVAILABLE SPECIFICATIONS:
{choices}

USER QUESTION:
{question}

JSON SPECIFICATION SELECTION:"""

# LLM Query
async def query_llm(prompt: str, provider: str = "groq") -> str:
    if not prompt:
        raise LLMProviderError("Cannot query an LLM with an empty prompt.")
    config = get_provider_config(provider)
    api_url, headers, payload = build_llm_request(prompt, provider)
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            logger.info(
                f"Sending request to {config['label']} ({config['model']})"
            )
            response = await client.post(
                api_url,
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            try:
                data = response.json()
            except ValueError as exc:
                raise LLMProviderError(
                    f"{config['label']} returned an invalid response."
                ) from exc
            if data.get("error") or not data.get("choices"):
                raise LLMProviderError(
                    f"{config['label']} returned no answer. Try again in a moment."
                )
            try:
                answer = data["choices"][0]["message"]["content"].strip()
            except (KeyError, IndexError, TypeError) as e:
                raise LLMProviderError(
                    f"{config['label']} returned an unexpected response format."
                ) from e
            if not answer:
                raise LLMProviderError(
                    f"{config['label']} returned an empty answer. Try again in a moment."
                )
            _record_usage(data)
            logger.info(f"Received response from {config['label']}.")
            return answer
    except httpx.HTTPStatusError as e:
        status = e.response.status_code
        logger.error(f"{config['label']} HTTP error: {status}.")
        if status == 429:
            detail = "rate limit reached; wait briefly or try the other provider"
        elif status in (401, 403):
            detail = f"authentication failed; check {config['api_key_env']}"
        else:
            detail = f"request failed with HTTP {status}"
        raise LLMProviderError(f"{config['label']} {detail}.") from e
    except httpx.RequestError as e:
        logger.error(f"{config['label']} request failed: {e}.")
        raise LLMProviderError(
            f"Could not reach {config['label']}. Check your connection and try again."
        ) from e
